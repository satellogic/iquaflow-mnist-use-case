import sys
import os
import argparse
import json
import random
import torch
import glob
import cv2
import mlflow

import numpy as np
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from datetime import timedelta

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel


try:
    from torch.distributed.elastic.utils.data import ElasticDistributedSampler
except ModuleNotFoundError as e:
    print(str(e))
    from torchelastic.utils.data import ElasticDistributedSampler

def init_mlflow(opt, rank):

    if rank != 0:
        return

    mlflow.set_tracking_uri(opt.mlf_tracking_uri)
    mlflow.set_experiment(opt.mlf_experiment_name)

    for attr in ['batch_sz','n_epochs','learning_rate','gpus','seed','n_gpus']:
        if attr.startswith('.'):
            continue
        mlflow.log_param(attr, getattr(opt,attr))

    if opt.mlf_tags:
        try:
            mlf_tags = json.loads(opt.mlf_tags)
            mlflow.set_tags(mlf_tags)
        except Exception as e:
            print("invalid mlflow tags %s" % opt.mlf_tags)
            print(str(e))

def initialize_model(model: nn.Module, device_id: int):
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    model = model.cuda(device_id)
    cudnn.benchmark = True
    model = DistributedDataParallel(
        model, device_ids=[device_id], output_device=device_id
    )
    return model

#####################################################################
# DATASET
#####################################################################

class CustomDataset(Dataset):
    
    def __init__(
        self,
        root_dir= "mnist_png/training",
    ):
        self.root_dir = root_dir
        self.fn_lst = glob.glob(os.path.join(root_dir,'images','*.png'))

        # use only 10% of the data
        # random.shuffle(self.fn_lst)
        # self.fn_lst = self.fn_lst[:int(len(fn_lst)*0.1)]
    
        with open(os.path.join(root_dir,'annotations.json')) as json_file:
            self.annotjson = json.load(json_file)
            
    def __len__(self):
        return len(self.fn_lst)
    
    def __getitem__( self, ii ):
        
        arr = np.transpose( cv2.imread(self.fn_lst[ii]), [2,0,1] )
        
        img_json = [
            ij for ij in self.annotjson['images']
            if ij['file_name']==os.path.basename(self.fn_lst[ii])
        ][0]
        
        image_id = img_json['id']
        
        category_id = [
            an for an in self.annotjson['annotations'] 
            if an['image_id']==image_id
        ][0]['category_id']
        
        #one_hot_vect              = np.array([0 for i in range(10)])
        #one_hot_vect[category_id] = 1.0
        
        sample = {
            'img': torch.from_numpy(arr.astype(np.float32)),
            'target': torch.from_numpy(np.array(category_id).astype(np.float32)),

        }

        return sample

#####################################################################
# DATALOADER
#####################################################################

def get_dataloader(
    ds_dir = "mnist_png",
    batch_size=8,
    threads=1,
    distributed_sampler=True,
    seed=0
):
    dataset = CustomDataset(root_dir = ds_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        #shuffle=True,
        num_workers=threads,
        sampler=(ElasticDistributedSampler(dataset,seed) if distributed_sampler else None)
    )

    return dataloader


#####################################################################
# MODEL
#####################################################################

import torchvision.models as models
from torch import nn

class Regressor(nn.Module):
    
    def __init__(self, pretrained=True, n_channels=3 ,num_classes=1):
        """
        resnet18 architecture is designed with 3channels input
        """
        super(Regressor, self).__init__()
        resnet18 = models.resnet18(pretrained=pretrained)
        modules=list(resnet18.children())[:-1]
        
        if n_channels!=3:
            modules[0] = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.basenet = nn.Sequential(*modules)
        self.fc = nn.Linear(512, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        for i in range(len(self.basenet)):
            x = self.basenet[i](x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

#####################################################################
# TRAIN LOOP
#####################################################################


def train_loop(
    rank,
    model,
    train_dataloader,
    vali_dataloader,
    n_epochs = 300,
    weights_fn = './last.pt',
    optimizer = 'Here add something like > optim.Adam(model.parameters(), lr=0.0001)',
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    criterion = nn.MSELoss(),
    metrics_lst = [],
    metrics_func_lst = [],
    print_every_n_epochs = 10
):
    
    os.makedirs( os.path.dirname( weights_fn ) , exist_ok=True )
    
    model = model.to(device)
    
    mode_lst = ['train','vali']
    
    metrics_lst = ['loss'] + metrics_lst
    metrics_func_lst = [criterion] + metrics_func_lst

    metrics_dict = {
        mode:{ met:[] for met in metrics_lst }
        for mode in mode_lst
    }
    
    for epoch in range(n_epochs):

        for mode in mode_lst:

            if mode=='train':
                aux = model.train()
                dataloader = train_dataloader
            elif mode == "vali" and rank == 0:
                aux = model.eval()
                dataloader = vali_dataloader

            metrics_batch = { met:[] for met in metrics_lst }
            
            for sample in tqdm(dataloader):

                x = sample['img'].to( device )
                y = sample['target'].to( device )

                pred = model.forward(x)
                loss = criterion( pred , y.long())
                
                if mode=='train':
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                for f,met in zip( metrics_func_lst, metrics_lst ):
                    
                    if met=='loss':
                        metrics_batch[met].append( loss.item() )
                    else:
                        metrics_batch[met].append( f(pred,y).item() )

            for met in metrics_lst:
                
                metrics_dict[mode][met].append( np.mean(metrics_batch[met]) )
        
        if rank==0 and print_every_n_epochs:
            
            if epoch%print_every_n_epochs==0:
                
                print('*********************')
                print(f'epoch\t\t{epoch}')
                
                for mode in mode_lst:
                    
                    for met in metrics_lst:
                        
                        print(f'{mode}_{met}\t\t{metrics_dict[mode][met][-1]}')
            
        for mode in mode_lst:
                
            for met in ['loss']+metrics_lst:
                    
                curr_epoch_met = metrics_dict[mode][met][-1]
                print(f"{mode}{met}", curr_epoch_met, epoch)
                #writer.add_scalar(f"{mode}/{met}", curr_epoch_met, epoch)
                if rank==0:
                    mlflow.log_metric(f"{mode}_{met}", curr_epoch_met, epoch)

        if rank == 0:
            # Save weigths
            # s_dict = model.state_dict()
            # torch.save( s_dict , weights_fn )
            mlflow.pytorch.log_model(model, 'last.pth')

    results = {
        **{ 'train'+k:metrics_dict['train'][k] for k in metrics_dict['train']},
        **{ 'vali'+k:metrics_dict['vali'][k] for k in metrics_dict['vali']}
    }
    
    return results

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--weights', type=str, default='./data/mnist_last.pt', help='weights path')
    parser.add_argument('--trainds', type=str, default='./data/mnist_png/training', help='train dataset path')
    parser.add_argument('--valds', type=str, default='./data/mnist_png/validation', help='validation dataset path')
    parser.add_argument('--outputpath', type=str, default='./data', help='json config file')
    parser.add_argument('--batch_sz', type=int, default=64, help='batch_sz')
    parser.add_argument('--n_epochs', type=int, default=1, help='n_epochs')
    parser.add_argument('--threads', type=int, default=1, help='threads')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning_rate')
    #
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    parser.add_argument("--seed", default="12345", type=str, help="random seed")
    parser.add_argument("--n_gpus", type=int, default=int(os.environ.get("SM_NUM_GPUS", 1)))
    #
    parser.add_argument("--mlf_tracking_uri",type=str, default="https://mlflow.ml.analytics-dev.satellogic.team/")
    parser.add_argument("--mlf_experiment_name", type=str, default="MNIST")
    parser.add_argument("--mlf_tags", type=str, default="{}")
    parser.add_argument(
        "--dist-backend",
        default="nccl",
        choices=["nccl", "gloo"],
        type=str,
        help="distributed backend",
    )
    
    opt       = parser.parse_args()
    distrib   = "LOCAL_RANK" in os.environ
    device_id = int(( os.environ["LOCAL_RANK"] if "LOCAL_RANK" in os.environ else opt.gpus.split(',')[0] ))
    rank      = int(( os.environ["RANK"] if "RANK" in os.environ else '0' ))

    random.seed(int(opt.seed))
    np.random.seed(int(opt.seed))
    torch.manual_seed(int(opt.seed))
    torch.cuda.manual_seed_all(int(opt.seed))
    torch.backends.cudnn.deterministic = True

    init_mlflow(opt, rank)
    torch.cuda.set_device(device_id)

    print(f"==***==> set cuda device = {device_id}/{rank}")

    dist.init_process_group(
        backend=opt.dist_backend,
        init_method="env://",
        timeout=timedelta(seconds=10)
    )
    
    print('Getting dataloaders...')
    
    train_dataloader = get_dataloader(
        ds_dir              = opt.trainds,
        batch_size          = opt.batch_sz,
        threads             = opt.threads,
        distributed_sampler = distrib,
        seed                = int(opt.seed)
    )
    
    vali_dataloader = get_dataloader(
        ds_dir              = opt.valds,
        batch_size          = opt.batch_sz,
        threads             = opt.threads,
        distributed_sampler = False
    )
    
    print('Building model...')
    
    model = Regressor(
        pretrained  = True,
        num_classes = 10
    )
    
    if distrib:
        model = initialize_model(model,device_id)
    
    print('Start training...')
    
    results = train_loop(
        rank,
        model,
        train_dataloader,
        vali_dataloader,
        n_epochs             = opt.n_epochs,
        weights_fn           = opt.weights,
        optimizer            = optim.Adam(model.parameters(), lr=opt.learning_rate),
        device               = f'cuda:{device_id}',
        criterion            = nn.CrossEntropyLoss(),
        metrics_lst          = [],
        metrics_func_lst     = [],
        print_every_n_epochs = 1
    )
