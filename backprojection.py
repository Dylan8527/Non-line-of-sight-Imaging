'''
Author: Dylan8527 vvm8933@gmail.com
Date: 2022-11-23 18:21:45
LastEditors: Dylan8527 vvm8933@gmail.com
LastEditTime: 2022-11-25 10:21:28
FilePath: \cs276_assignment2_v1.0\backprojection.py
Description: 

Copyright (c) 2022 by Dylan8527 vvm8933@gmail.com, All Rights Reserved. 
'''
from inputdata import Dataset
from boundingbox import BoundingBox
import numpy as np
import tqdm 
import torch
# import taichi as ti


# @ti.func
def check_intersection(radiance, r, o, pos):
    signs = torch.sign(torch.norm(pos - o, dim=-1) - r)
    signs = torch.sum(signs, dim=-1)
    # print(signs.shape, radiance.shape)
    # increments = torch.stack([torch.where(torch.abs(signs[i])!=8, radiance[i], 0) for i in range(radiance.shape[0])], dim=0
    increments = torch.where(torch.abs(signs)!=8, radiance, 0)  
    # increments = torch.sum(increments, dim=0)
    return increments

def solve(dataset: Dataset, aabb: BoundingBox)->np.array:
    # get the wall position
    walls = aabb.get_walls()
    # get the wall radiance at different time t 0~1023
    imgs = dataset.get_rect_data()
    imgs = torch.from_numpy(imgs).cuda()
    eight_corners = torch.from_numpy(aabb.eight_corners).cuda()
    grid = torch.from_numpy(aabb.grid)
    densities = torch.zeros_like(grid[..., 0]).cuda()
    o = []
    for wall in walls:
        o.append(grid[wall])
    o = torch.stack(o, dim=0).cuda()
    from tqdm import tqdm
    for t in tqdm(range(imgs.shape[0]), desc="Backprojection: "):
        img = imgs[t].reshape(-1) # the wall at time t 
        r = aabb.calc_r(t)
        densities += backprojection(img, r, o, eight_corners, densities)
    return densities.cpu().numpy()

# @ti.kernel
def backprojection(img, r: float, o, eight_corners, densities):
    zeros = torch.zeros_like(densities)
    if img.max() == 0:
        return zeros
        
    # we slice img into different pieces 
    # slices = np.arange(img.shape[0], step=8)
    # for i in range(len(slices)):
    #     st = slices[i]
    #     if i == len(slices) - 1:
    #         ed =  img.shape[0]
    #     else:
    #         ed = slices[i+1]
    #     radiance = img[st:ed]
    #     origin = o[st:ed][:, None, None, None, None, :]
    #     multiple_corners = eight_corners[None, ...]
    #     increments = check_intersection(radiance, r, origin, multiple_corners)
    #     zeros += increments

    for i in range(img.shape[0]):
        radiance = img[i] # float
        increments = check_intersection(radiance, r, o[i], eight_corners)
        zeros += increments
    return zeros
