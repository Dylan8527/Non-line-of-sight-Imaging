'''
Author: Dylan8527 vvm8933@gmail.com
Date: 2022-11-23 16:22:49
LastEditors: Dylan8527 vvm8933@gmail.com
LastEditTime: 2022-11-25 10:11:38
FilePath: \assignment2\boundingbox.py
Description: 

Copyright (c) 2022 by Dylan8527 vvm8933@gmail.com, All Rights Reserved. 
'''

import numpy as np

class BoundingBox:
    def __init__(self, width: float, wall_resolution: int, bin_resolution:float, c:float, maxt:float, fullboundingbox=False):
        self.bin_resolution = bin_resolution
        self.c = c
        self.maxr = bin_resolution * maxt * c / 2
        interval = width / wall_resolution
        self.interval = interval
        self.fullboundingbox = fullboundingbox
        # AABB (Axis Aligned Bounding Box)
        self.min = [0.0, 0.0, 0.0]
        if fullboundingbox:
            self.max = [width + self.maxr * 2, self.maxr * 25 / 34, width + self.maxr * 2]
        else:
            self.max = [width, self.maxr * 25 / 34, width]
        x = np.arange(self.min[0], self.max[0], interval)
        y = np.arange(self.min[1], self.max[1], interval)
        z = np.arange(self.min[2], self.max[2], interval)
        # print(x.shape)
        # print(y.shape)
        # print(z.shape)
        # generate voxel grid (position)
        self.grid = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
        self.eight_corners = self.get_eight_corner(self.grid)
        # generate density grid (density)
        self.densities = np.zeros_like(self.grid[..., 0])
        # Wall
        if fullboundingbox:
            self.wall_min = self.grid[int(self.maxr/interval), 0 , int(self.maxr/interval)]
            self.wall_max = self.grid[int(self.maxr/interval)+int(width/interval), 0, int(self.maxr/interval)+int(width/interval)]
            self.wall_min_index = (int(self.maxr/interval), 0 , int(self.maxr/interval))
            self.wall_max_index = (int(self.maxr/interval)+int(width/interval), 0, int(self.maxr/interval)+int(width/interval))
        else:
            self.wall_min = self.grid[0, 0, 0]
            self.wall_max = self.grid[-1, -1, -1]
            self.wall_min_index = (0, 0, 0)
            self.wall_max_index = (int(width/interval), 0, int(width/interval))
        
        # print(self.grid)
        # print(self.grid.shape) 
        # print(self.wall_min)
        # print(self.wall_max)
        # print((self.wall_max - self.wall_min))
    
    def calc_r(self, t: float):
        return self.bin_resolution * t * self.c / 2

    def get_eight_corner(self, pos: np.array(3)):
        # pos: position of the voxel
        # return: eight corner of the voxel
        return (np.array([
            [-self.interval, -self.interval, -self.interval],
            [self.interval, -self.interval, -self.interval],
            [-self.interval, self.interval, -self.interval],
            [self.interval, self.interval, -self.interval],
            [-self.interval, -self.interval, self.interval],
            [self.interval, -self.interval, self.interval],
            [-self.interval, self.interval, self.interval],
            [self.interval, self.interval, self.interval],
        ]) * 0.5)[None, None, None, ...]+ pos[..., None, :]

    def get_walls(self):
        xx = (self.wall_min_index[0], self.wall_max_index[0])
        zz = (self.wall_min_index[2], self.wall_max_index[2])
        walls = []
        for x in range(*xx):
                for z in range(*zz):
                    o = (x, 0, z)
                    walls.append(o)
        return walls
        
    def get_wall_min(self):
        return self.wall_min
    
    def get_wall_max(self):
        return self.wall_max
    
    def get_min(self):
        return self.min
    
    def get_max(self):
        return self.max
    
    def get_grid(self):
        return self.grid

    def get_eight_corners(self):
        return self.eight_corners

    def get_densities(self):
        return self.densities

    def info(self):
        print('aabb_min: ', self.min)
        print('aabb_max: ', self.max)
        print('wall_min: ', self.wall_min)
        print('wall_max: ', self.wall_max)
        print('grid.shape(position): ', self.grid.shape)
        print('eight_corners.shape: ', self.eight_corners.shape)
        print('densities.shape(density): ', self.densities.shape)

    def save_densities(self, path='densities.npy'):
        np.save(path, self.densities)

if __name__ == "__main__":
    box = BoundingBox(width=1.0, wall_resolution=256, bin_resolution=8e-12, c=3e8, maxt=1024)
