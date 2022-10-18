import sys
import os
import argparse
import json
import random
import torch
import glob
import cv2

import numpy as np

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

#####################################################################
# DATASET
#####################################################################

class CustomDataset(Dataset):
    
    def __init__(
        self,
        root_dir= "mnist_png/training",
        seed=123,
        perc=0.01
    ):
        self.root_dir = root_dir
        fn_lst = glob.glob(os.path.join(root_dir,'images','*.png'))

        # use only 10% of the data
        random.seed(seed)
        random.shuffle(fn_lst)

        self.fn_lst = fn_lst[:int(len(fn_lst)*perc)]
    
        with open(os.path.join(root_dir,'annotations.json')) as json_file:
            self.annotjson = json.load(json_file)
            
    def __len__(self):
        return len(self.fn_lst)
    
    def __getitem__( self, ii ):
        
        arr = np.transpose( cv2.imread(self.fn_lst[ii]), [2,0,1] )/255
        
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
    num_workers=1,
    perc=0.01,
    seed=123
):
    dataset = CustomDataset(root_dir = ds_dir, perc=perc, seed=seed)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
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
            else:
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
                
        # Save weigths
        s_dict = model.state_dict()
        torch.save( s_dict , weights_fn )
        
        if print_every_n_epochs:
            
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
    
    results = {
        **{ 'train'+k:metrics_dict['train'][k] for k in metrics_dict['train']},
        **{ 'vali'+k:metrics_dict['vali'][k] for k in metrics_dict['vali']}
    }
    
    return results

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--weights', type=str, default='./course_data/mnist_last.pt', help='weights path')
    parser.add_argument('--trainds', type=str, default='./course_data/mnist_png/training', help='train dataset path')
    parser.add_argument('--valds', type=str, default='./course_data/mnist_png/validation', help='validation dataset path')
    parser.add_argument('--outputpath', type=str, default='./course_data', help='json config file')
    parser.add_argument('--batch_sz', type=int, default=64, help='batch_sz')
    parser.add_argument('--seed', type=int, default=123, help='seed')
    parser.add_argument('--n_epochs', type=int, default=15, help='n_epochs')
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning_rate')
    
    opt = parser.parse_args()
    
    print('Getting dataloaders...')
    
    train_dataloader = get_dataloader(
        ds_dir       = opt.trainds,
        batch_size   = opt.batch_sz,
        num_workers  = opt.num_workers,
        perc=0.1,
        seed = opt.seed
    )
    
    vali_dataloader = get_dataloader(
        ds_dir      = opt.valds,
        batch_size  = opt.batch_sz,
        num_workers = opt.num_workers,
        perc=0.1
    )
    
    print('Building model...')
    
    model = Regressor(
        pretrained  = True,
        num_classes = 10
    )
    
    print('Start training...')
    
    results = train_loop(
        model,
        train_dataloader,
        vali_dataloader,
        n_epochs             = opt.n_epochs,
        weights_fn           = opt.weights,
        optimizer            = optim.Adam(model.parameters(), lr=opt.learning_rate),
        device               = 'cuda' if torch.cuda.is_available() else 'cpu',
        criterion            = nn.CrossEntropyLoss(),
        metrics_lst          = [],
        metrics_func_lst     = [],
        print_every_n_epochs = 1
    )

    ####################################################################################
    #   Add some hyperparameters (key,value) in the results dictionary                 #
    #   Then save the results.json in the outpath                                      #
    
    results['batch_sz']      = opt.batch_sz
    results['n_epochs']      = opt.n_epochs
    results['learning_rate'] = opt.learning_rate
    results['learning_rate'] = opt.seed
    
    with open(os.path.join(opt.outputpath,'results.json'), 'w') as outfile:
        json.dump(results, outfile)
        
    ####################################################################################
    ####################################################################################
