import torch
import numpy as np
 
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
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
