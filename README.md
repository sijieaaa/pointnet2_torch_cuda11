# pointnet2_torch_cuda11

- A new implementation for PointNet++, https://arxiv.org/abs/1706.02413

- This is based on votenet/pointnet2/, https://github.com/facebookresearch/votenet/tree/main/pointnet2

- We fix some bugs of the above old pointnet2, and create this new pointnet2. 

- This pointnet2 works well in the following envs (not limit to these):

```
pytorch 1.5+
CUDA 11.3/11.6
python 3.6/3.7/3.8
ubuntu 18.04/20.04
g++ 9.4.0 gcc 9.4.0 cmake 3.23.0 (ubuntu 20.04)
g++ 7.5.0 gcc 7.5.0 cmake 3.23.2 (ubuntu 18.04)

```

- Usage:

```
git clone https://github.com/sijieaaa/pointnet2_torch_cuda11.git
cd pointnet2_torch_cuda11
python setup.py install
```



