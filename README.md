## Prerequisites

* Linux (tested on ubuntu 14.04LTS)
* NVIDIA GPU + CUDA CuDNN
* [Torch7](http://torch.ch/docs/getting-started.html)

## Getting started

1. install the dependencies
    ```bash
    luarocks install hdf5 matio protobuf rapidjson loadcaffe xml
    ```
    
2. Download dataset and proposals 
    ```bash
    cd ./MELM
    export DIR=$(pwd)
    
    cd $DIR/data/datasets
    # trainval
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    tar xvf  VOCtrainval_06-Nov-2007.tar
    # test
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    tar xvf VOCtest_06-Nov-2007.tar
    ```
    
    Proposals can be avalable from [https://dl.dropboxusercontent.com/s/orrt7o6bp6ae0tc/selective_search_data.tgz](https://github.com/rbgirshick/fast-rcnn)

3. install functions

    ```bash
    cd $DIR/utils/c-cuda-functions
    sh install.sh
    
    cd $DIR/layers
    luarocks make
    ```
    
 4. train and test
 
    ```bash
    sh Run_DeepMELM.sh 0 VOC2007 VGGF SSW 0.1 None melm
    ```
    
## Acknowledgements & Notes

This work would not have been possible without prior work: Vadim Kantorov's [contextlocnet](https://github.com/vadimkantorov), Spyros Gidaris's [LocNet](http://github.com/gidariss/LocNet), Sergey Zagoruyko's [loadcaffe](http://github.com/szagoruyko/loadcaffe), Facebook FAIR's [fbnn/Optim.lua](http://github.com/facebook/fbnn/blob/master/fbnn/Optim.lua).
