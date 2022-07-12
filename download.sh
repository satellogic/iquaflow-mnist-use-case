#!/bin/bash

TO_PATH=data
python3 -c "import os; os.makedirs('$TO_PATH',exist_ok=True)"
wget https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-mnist-use-case/datasets/mnist_png.tar.gz -O $TO_PATH/mnist.tar.gz
chmod 775 $TO_PATH/mnist.tar.gz
tar xvzf $TO_PATH/mnist.tar.gz -C $TO_PATH
rm $TO_PATH/mnist.tar.gz