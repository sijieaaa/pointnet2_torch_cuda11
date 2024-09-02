# pointnet2_torch_cuda11

- A new implementation for PointNet++, https://arxiv.org/abs/1706.02413

- This is based on votenet/pointnet2/, https://github.com/facebookresearch/votenet/tree/main/pointnet2
https://github.com/erikwijmans/Pointnet2_PyTorch/tree/master/pointnet2_ops_lib

- We fix some bugs of the above old pointnet2, and create this new pointnet2. 

- This pointnet2 works well in the following envs (include but not limit to these):

```
pytorch 1.5.0+
CUDA 11.3/11.6
python 3.6/3.7/3.8
ubuntu 18.04/20.04
g++ 7.5.0 gcc 7.5.0 cmake 3.23.2 (ubuntu 18.04)
g++ 9.4.0 gcc 9.4.0 cmake 3.23.0 (ubuntu 20.04)
```

- You need to make sure PyTorch is installed correctly with the corresponding NVCC version.
```
nvcc --version
```
```
>>> torch.version.cuda
'11.8'
```


- Install:

```
git clone https://github.com/sijieaaa/pointnet2_torch_cuda11.git
cd pointnet2_torch_cuda11
python setup.py install
```

- Example (FPS):

```
import torch
import numpy as np
import pointnet2_utils
import matplotlib.pyplot as plt
 
def test_furthest_point_sampling():
    b = 10
    c = 2
    N = 1000
    n = 10
    pool = torch.randn(b, N, c, requires_grad=True).float().cuda()
    indices = pointnet2_utils.furthest_point_sample(pool, n).long()
 
    indices = indices.detach().cpu() # [b,n]
    pool = pool.detach().cpu() # [b,c,m]
    sampled = torch.gather(pool, 1, indices.unsqueeze(-1).expand(b, n, c))
    sampled = sampled.numpy()
    pool = pool.numpy()
    print(indices[0])
 
    plt.figure()
    plt.plot(pool[1,:,0], pool[1,:,1], 'b.')
    plt.plot(sampled[1,:,0], sampled[1,:,1], 'r.')
    plt.savefig('furthest_point_sample.png')
 
if __name__=='__main__':
    test_furthest_point_sampling()
```

