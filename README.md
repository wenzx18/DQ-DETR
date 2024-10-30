# DQ-DETR: DETR with Dynamic Query for Tiny Object Detection

![method](./figure/model_final_V4.pdf)

* This repository is an official implementation of the paper DQ-DETR: DETR with Dynamic Query for Tiny Object Detection.
* The original repository link was https://github.com/Katie0723/DQ-DETR. Here is the updated link.


## News
[2024/7/1]: **DQ-DETR** has been accepted by **ECCV 2024**. 🔥🔥🔥

[2024/5/3]: **DNTR** has been accepted by **TGRS 2024**. 🔥🔥🔥


## Installation -- Compiling CUDA operators
* The code are built upon the official [DINO DETR](https://github.com/IDEA-Research/DINO) repository.

```sh
conda create -n dqdetr python=3.9 --y
conda activate dqdetr
bash install.sh
```

<!-- # bash scripts/DQ_eval.sh /nfs/home/hoiliu/dqdetr/weights/dqdetr_best305.pth -->
## Eval models
```sh
bash scripts/DQ_eval.sh /path/to/your/dataset /path/to/your/checkpoint
```

## Trained Model
* Changed the pretrained model path in DQ.sh
```sh
CUDA_VISIBLE_DEVICES=5,6,7 bash scripts/DQ.sh /path/to/your/dataset
```

## Other works on Tiny Object Detection 
* Title: A DeNoising FPN With Transformer R-CNN for Tiny Object Detection.
Author: Hou-I Liu and Yu-Wen Tseng and Kai-Cheng Chang and Pin-Jyun Wang and Hong-Han Shuai, and Wen-Huang Cheng. 
Journal: IEEE Transactions on Geoscience and Remote Sensing (**TGRS 2024**).
[paper](https://arxiv.org/abs/2406.05755) | [code](https://github.com/hoiliu-0801/DNTR) 

## Citation
```bibtex
@article{huang2024dq,
  title={Dq-detr: Detr with dynamic query for tiny object detection},
  author={Huang, Yi-Xin and Liu, Hou-I and Shuai, Hong-Han and Cheng, Wen-Huang},
  journal={arXiv preprint arXiv:2404.03507},
  year={2024}
}

@ARTICLE{10518058,
  author={Liu, Hou-I and Tseng, Yu-Wen and Chang, Kai-Cheng and Wang, Pin-Jyun and Shuai, Hong-Han and Cheng, Wen-Huang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={A DeNoising FPN With Transformer R-CNN for Tiny Object Detection}, 
  year={2024},
  volume={62},
  number={},
  pages={1-15},
}
