'''
Author: Dylan8527 vvm8933@gmail.com
Date: 2022-11-23 19:42:29
LastEditors: Dylan8527 vvm8933@gmail.com
LastEditTime: 2022-11-26 22:10:11
FilePath: \cs276_assignment2_v1.0\visualize.py
Description: 

Copyright (c) 2022 by Dylan8527 vvm8933@gmail.com, All Rights Reserved. 
'''
from inputdata import Dataset
from boundingbox import BoundingBox
from backprojection import solve
from filtering import filtering

import os
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import taichi as ti
DOWNFACTOR = .25
DENSITIES_PATH = './densities64_fullaabb.npy'

def make_video(densities: np.array, path='outputdata.avi'):
    # visulize along y axis
    basepath = './output/results'
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    import cv2
    imgs = (densities - np.min(densities)) / (np.max(densities) - np.min(densities)) * 255
    imgs = np.array(imgs, dtype=np.uint8)[..., None]
    imgs = np.concatenate([imgs, imgs, imgs], axis=-1)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, 1, (imgs.shape[2], imgs.shape[0]))
    for i in range(imgs.shape[1]):
        img = np.ascontiguousarray(imgs[:, i, :, :])
        img = np.transpose(img, (1, 0, 2))
        # plt.imsave(os.path.join(basepath, f'{DOWNFACTOR}_{i}.png'), img, cmap='gray')
        out.write(img)
    out.release()

def show_different_downsampling_inputdata():
    downsample_factors = [1., 0.5, 0.25, 0.125]
    basepath = './output/downsample_inputdata'
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    for downsample_factor in downsample_factors:
        dataset = Dataset(downsample_factor=downsample_factor)
        imgs = dataset.get_rect_data()
        imgs = np.transpose(imgs, (2, 0, 1))
        print(imgs.shape)
        # make_video(imgs, os.path.join(basepath, f'downsample_{downsample_factor}.avi'))
        # save the image at time 360
        plt.imsave(os.path.join(basepath, f'downsample_{downsample_factor}.png'), imgs[:, 360, :], cmap='gray')

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_inputdata():
    downsampling_factors = [1.0, 0.5, 0.25, 0.125]
    pic_basepath = './pictures'
    video_basepath = './videos'
    for downsampling_factor in downsampling_factors:
        pic_inputdata_path = os.path.join(pic_basepath, 'inputdata')
        video_inputdata_path = os.path.join(video_basepath, 'inputdata')
        dataset = Dataset(path="data_bunny.mat", downsample_factor=downsampling_factor)
        # 1.1 save input data video
        video_path = os.path.join(video_inputdata_path, f'{downsampling_factor}')
        check_dir(video_path)
        dataset.make_video(os.path.join(video_path, 'inputdata.avi'))

        # 1.2 save input data picture
        pic_path = os.path.join(pic_inputdata_path, f'{downsampling_factor}')
        check_dir(pic_path)
        imgs = dataset.get_rect_data()
        # for img in imgs:
        for t in range(imgs.shape[0]):
            plt.imsave(os.path.join(pic_path, f'{t}.png'), imgs[t], cmap='gray')

def normalize(img: np.ndarray):
    return (img - img.min()) / (img.max() - img.min() + 1e-5)

def save_outputdata():
    density_paths = ['./densities32_sliceaabb.npy','./densities32_fullaabb.npy', \
                    './densities64_sliceaabb.npy', './densities64_fullaabb.npy', \
                    './densities128_sliceaabb.npy']
    types = ['32_sliceaabb', '32_fullaabb', \
             '64_sliceaabb', '64_fullaabb', \
             '128_sliceaabb']
    pic_basepath = './pictures'
    video_basepath = './videos'
    for i in range(len(density_paths)):
        density_path = density_paths[i]
        typename = types[i]
        # without filtering
        densities = np.load(density_path)
        ling = np.zeros((densities.shape[0], 3, densities.shape[2]))
        densities = np.concatenate((ling, densities), axis=1)
        pic_outputdata_path = os.path.join(pic_basepath, 'outputdata_without_filtering')
        video_outputdata_path = os.path.join(video_basepath, 'outputdata_without_filtering')
        # 2.1 save output data video (without filtering)
        video_path = os.path.join(video_outputdata_path, f'{typename}')
        check_dir(video_path)
        make_video(densities, os.path.join(video_path, 'outputdata.avi'))
        
        # 2.2 save output data picture (without filtering)
        pic_path = os.path.join(pic_outputdata_path, f'{typename}')
        check_dir(pic_path)
        for y in range(densities.shape[1]):
            plt.imsave(os.path.join(pic_path, f'without_filtering_{typename}_{y}.png'), normalize(densities[:, y, :]).T, cmap='gray')

        # filtering
        densities = filtering(densities)
        pic_outputdata_path = os.path.join(pic_basepath, 'outputdata_with_filtering')
        video_outputdata_path = os.path.join(video_basepath, 'outputdata_with_filtering')
        # 2.1 save output data video (without filtering)
        video_path = os.path.join(video_outputdata_path, f'{typename}')
        check_dir(video_path)
        make_video(densities, os.path.join(video_path, 'outputdata.avi'))
        
        # 2.2 save output data picture
        pic_path = os.path.join(pic_outputdata_path, f'{typename}')
        check_dir(pic_path)
        for y in range(densities.shape[1]):
            plt.imsave(os.path.join(pic_path, f'with_filtering_{typename}_{y}.png'), normalize(densities[:, y, :]).T, cmap='gray')

def record_results():
    # 1. save input data at different downsampling factor
    # save_inputdata()
    # 2. save output data at different resolution and bounding box
    save_outputdata()

if __name__ == "__main__":
    record_results()
    exit()
    # 1. Load scene
    # 1.1 data
    print("---------------------------------")
    dataset = Dataset(path="data_bunny.mat", downsample_factor=DOWNFACTOR)
    # dataset.make_video()
    dataset.info()
    # 1.2 bounding box
    print("---------------------------------")
    aabb = BoundingBox(width=dataset.get_width(), wall_resolution=dataset.get_wall_resolution(), bin_resolution=dataset.get_bin_resolution(), c=dataset.get_c(), maxt=dataset.get_maxt())
    aabb.info()
    # 2. Backprojection
    print("---------------------------------")
    if not os.path.exists(DENSITIES_PATH):
        # ti.init(arch=ti.gpu, dynamic_index=True)
        aabb.densities = solve(dataset=dataset, aabb=aabb)
        aabb.save_densities(DENSITIES_PATH)

    # 3. Laplacian filtering
    # print("---------------------------------")
    densities = np.load(DENSITIES_PATH)
    ling = np.zeros((densities.shape[0], 3, densities.shape[2]))
    densities = np.concatenate((ling, densities), axis=1)
    
    print(densities.max(), densities.sum()/(np.abs(np.sign(densities))).sum(), densities.min())
    #TODO: implement laplacian filtering after thresholding
    densities = filtering(densities)
    print(densities.max(), densities.sum()/(np.abs(np.sign(densities))).sum(), densities.min())

    # make_video(densities)
    # exit()
    #---------------------------------#
    # 4. for visualization
    verts, faces, normals, values = measure.marching_cubes(densities, level=50.0)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlim(0, densities.shape[0])  
    ax.set_ylim(0, densities.shape[1])  
    ax.set_zlim(0, densities.shape[2]) 

    # plt.tight_layout()
    plt.show()