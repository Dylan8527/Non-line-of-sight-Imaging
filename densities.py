'''
Author: Dylan8527 vvm8933@gmail.com
Date: 2022-11-25 10:16:58
LastEditors: Dylan8527 vvm8933@gmail.com
LastEditTime: 2022-11-25 10:18:18
FilePath: \cs276_assignment2_v1.0\densities.py
Description: 

Copyright (c) 2022 by Dylan8527 vvm8933@gmail.com, All Rights Reserved. 
'''
from inputdata import Dataset
from boundingbox import BoundingBox
from backprojection import solve

def generate_densities():
    downsampling_factors = [0.125, 0.125, 0.25, 0.25]
    density_paths = ['./densities32_sliceaabb.npy','./densities32_fullaabb.npy', \
                    './densities64_sliceaabb.npy', './densities64_fullaabb.npy']
    fullboundingboxs = [False, True, False, True]
    for i in range(len(density_paths)):
        density_path = density_paths[i]
        downsampling_factor = downsampling_factors[i]
        fullboundingbox = fullboundingboxs[i]
        dataset = Dataset(path="data_bunny.mat", downsample_factor=downsampling_factor)
        dataset.info()
        aabb = BoundingBox(width=dataset.get_width(), wall_resolution=dataset.get_wall_resolution(), bin_resolution=dataset.get_bin_resolution(), c=dataset.get_c(), maxt=dataset.get_maxt(), fullboundingbox=fullboundingbox)
        aabb.info()
        aabb.densities = solve(dataset=dataset, aabb=aabb)
        aabb.save_densities(density_path)

if __name__ == '__main__':
    generate_densities()