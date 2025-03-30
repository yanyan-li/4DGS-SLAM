[comment]: <> (# 4D Gaussian Splatting SLAM)

<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center"> 4D Gaussian Splatting SLAM
  </h1>
  <p align="center">
    <p align="center">
    <strong>Yanyan Li</strong> ·
    <strong>Youxu Fang</strong> ·
    <strong>Zunjie Zhu</strong> ·
    <strong>Kunyi Li</strong> ·
    <strong>Yong Ding</strong> ·
    <strong>Federico Tombari</strong>
    </p>
  </p>




[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center"><a href="https://arxiv.org/pdf/2503.16710">Paper</a> | <a href="https://yanyan-li.github.io/project/gs/4dgsslam.html">Project Page</a></h3>
  <div align="center"></div>

<p align="center">
  <a href="">
    <img src="./imgs/teaser.png" alt="teaser" width="100%">
  </a>

  <a href="">
    <img src="./imgs/4DGSSLAM.gif" alt="teaser" width="100%">
  </a>
</p>


<br>



# 1.Installation
 
 
```
git clone https://github.com/yanyan-li/4DGS-SLAM.git
cd 4DGS-SLAM
```

Setup the environment.
```
conda create -n 4dgs-slam python=3.8
conda activate 4dgs-slam
# CUDA 11.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

The simple-knn and diff-gaussian-rasterization libraries use the ones provided by MonoGS.
```
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
```

Use torch-batch-svd speed up (Optional)
```
git clone https://github.com/KinglittleQ/torch-batch-svd
cd torch-batch-svd
python setup.py install
```

# 2.Pretrained Models

Download YOLOv9e-seg
```bash
cd 4DGS-SLAM/pretrained
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e-seg.pt
```
Or download it directly from https://docs.ultralytics.com/models/yolov9/

Download RAFT

The model **raft-things.pth** used in this system can be obtained directly from https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT


# 3.Datasets

### TUM-RGBD dataset
```bash
bash scripts/download_tum_dynamic.sh
```

### BONN dataset

# 4.Testing

### Dynamic rendering
```bash
python slam.py --config configs/rgbd/tum/fr3_sitting.yaml --eval --dynamic
```
### Adjust the frequency for image saving
```bash
python slam.py --config configs/rgbd/tum/fr3_sitting.yaml --eval --dynamic --interval 50
```



# 5.Acknowledgement
This work incorporates many open-source codes. We extend our gratitude to the authors of the software.
- [MonoGS](https://github.com/muskie82/MonoGS)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [GeoGaussian](https://github.com/yanyan-li/GeoGaussian)
- [SC-GS](https://github.com/CVMI-Lab/SC-GS)
- [Open3D](https://github.com/isl-org/Open3D)


# 6.License



# 7.Citation
If you find this code/work useful for your own research, please consider citing:
```
@article{li20254d,
  title={4{D} {G}aussian {S}platting {SLAM}},
  author={Li, Yanyan and Fang, Youxu and Zhu, Zunjie and Li, Kunyi and Ding, Yong and Tombari, Federico},
  journal={arXiv preprint arXiv:2503.16710},
  year={2025}
}
```












