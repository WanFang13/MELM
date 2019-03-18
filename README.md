## New
A simplified version of MELM with context in PyTorch is released [\[here\]](https://github.com/vasgaowei/pytorch_MELM).

## Prerequisites

* Linux (tested on ubuntu 14.04LTS)
* NVIDIA GPU + CUDA CuDNN
* [Torch7](http://torch.ch/docs/getting-started.html)

## Getting started

1. Install the dependencies
    ```bash
    luarocks install hdf5 matio protobuf rapidjson loadcaffe xml
    ```
    
2. Download dataset, proposals and ImageNet pre-trained model

    Download VOC2007 from:
    http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    
    Download proposals from: 
    [https://dl.dropboxusercontent.com/s/orrt7o6bp6ae0tc/selective_search_data.tgz](https://github.com/rbgirshick/fast-rcnn)
    
    Download VGGF from:
    http://www.robots.ox.ac.uk/~vgg/software/deep_eval/releases/bvlc/VGG_CNN_F.caffemodel
    https://gist.githubusercontent.com/ksimonyan/a32c9063ec8e1118221a/raw/6a3b8af023bae65669a4ceccd7331a5e7767aa4e/VGG_CNN_F_deploy.prototxt
    
    The data folder has the following structure:
    ```bash
    $MELM/data/datasets/VOCdevkit_2007/
    $MELM/data/datasets/VOCdevkit_2007/VOCcode
    $MELM/data/datasets/VOCdevkit_2007/VOC2007
    $MELM/data/datasets/VOCdevkit_2007/...
    $MELM/data/datasets/proposals/
    $MELM/data/models/
    $MELM/data/results/
    ``` 
    
3. Install functions

    ```bash
    cd ./MELM
    export DIR=$(pwd)   
    
    cd $DIR/utils/c-cuda-functions
    sh install.sh
    
    cd $DIR/layers
    luarocks make
    ```
    
 4. Train and test
 
    ```bash
    cd $DIR
    sh Run_MELM.sh 0 VOC2007 VGGF SSW 0.1 None melm
    ```
    
## Acknowledgements

This work would not have been possible without prior work: Vadim Kantorov's [contextlocnet](https://github.com/vadimkantorov), Spyros Gidaris's [LocNet](http://github.com/gidariss/LocNet), Sergey Zagoruyko's [loadcaffe](http://github.com/szagoruyko/loadcaffe), Facebook FAIR's [fbnn/Optim.lua](http://github.com/facebook/fbnn/blob/master/fbnn/Optim.lua).
