
# Distributed usage

```
# PASTE AWS CREDENTIALS HERE

./download.sh

docker run --rm -it --gpus all --shm-size=32gb -v $(pwd):/workspace torche

pip install -r torch_elastic/requirements.txt && \
export MASTER_ADDR=127.0.0.1 && \
export MASTER_PORT=29501 && \
export RANK=0 && \
export LOCAL_RANK=0 && \
torchrun --standalone  \
    --nnodes=1  \
    --nproc_per_node=2  \
    torch_elastic/train.py \
        --gpus=0,1 \
        --n_gpus=2 \
        --n_epochs=1 \
        --threads=4 \
        --batch_sz=6000 \
        --dist-backend=nccl
```