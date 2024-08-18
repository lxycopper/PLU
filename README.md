This repo contains an implementation of the following paper:
> **Proposal-Level Unsupervised Domain Adaptation for Open World Unbiased Detector**<br>


## Installation
1. Clone the repo
```
git clone https://github.com/lxycopper/PLU.git
cd PLU
```
2. Create conda environment and install dependencies
```
# create environment
conda create -n plu python=3.10
conda activate plu
pip install torch==1.7.0+cu110 torchvision==0.8.0+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

## install detectron2 according to the CUDA and torch versions
python -m pip install detectron2==0.5 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
python -m pip install -e ./

## install other packages
pip install reliability shortuuid

```
Other detectron2 versions can be found here: [detectron2](https://github.com/facebookresearch/detectron2/releases). We recommend directly using pre-built version. The pre-built package has to be used with corresponding version of CUDA and official PyTorch release.
## :open_book: Overview
![overall_structure](method-1.png)
![fixmatch](method-2.png)

## Dataset

   
## Quick Start 

You can run the code on a 4 GPU machine following the command:
```python
python tools/train_net.py --num-gpus 4 --config-file <Change to the appropriate config file> SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005
```
All config files can be found in: `configs/OWOD`

Alternatively, you can run the bash script `run.sh` file for a task workflow.
```
bash run.sh
```



## Possible Issues:
1. No existing key: rerun detectron2 installation
```
python -m pip install -e ./
```

2. cannot find "R-50.pkl"
```
pip install fvcore==0.1.1.dev200512
```

3. other installation issues, you may refer to [Common Installation Issues](https://github.com/JosephKJ/OWOD/blob/master/INSTALL.md)
   



## :fountain_pen: Citation

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
   @article{liu2023proposal,
    title={Proposal-Level Unsupervised Domain Adaptation for Open World Unbiased Detector},
    author={Liu, Xuanyi and Yue, Zhongqi and Hua, Xian-Sheng},
    journal={arXiv preprint arXiv:2311.02342},
    year={2023}
  }
  ```


## Acknowledgement
This project is built using the following open source repositories: [ORE](https://github.com/JosephKJ/OWOD)

