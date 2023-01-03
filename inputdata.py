'''
Author: Dylan8527 vvm8933@gmail.com
Date: 2022-11-22 23:09:34
LastEditors: Dylan8527 vvm8933@gmail.com
LastEditTime: 2022-11-25 10:09:36
FilePath: \assignment2\inputdata.py
Description: 

Copyright (c) 2022 by Dylan8527 vvm8933@gmail.com, All Rights Reserved. 
'''
# load 'data_bunny.mat'
import scipy.io as sio

def load_data(path='data_bunny.mat'):
    data = sio.loadmat(path)
    # There's a mistake in the data. The width given should be 1 meter, not 0.5 meter.
    data['width'] = float(data['width']) * 2
    data['bin_resolution'] = 8e-12
    data['c'] = 3e8
    return data

def make_video(data: dict, path='inputdata.avi'):
    # data.size() = [x, y, t]
    import cv2
    import numpy as np
    imgs = np.array(data) * 1e5
    imgs = np.array(imgs, dtype=np.uint8)[..., None]
    imgs = np.concatenate([imgs, imgs, imgs], axis=-1)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, 30, (imgs.shape[2], imgs.shape[1]))
    for i in range(imgs.shape[0]):
        out.write(imgs[i])
    out.release()

class Dataset:
    def __init__(self, path='data_bunny.mat', downsample_factor=1.):
        self.data = load_data(path)
        # downsample the data if need
        if downsample_factor != 1.:
            import cv2
            import numpy as np
            downsize = (int(self.data['rect_data'].shape[0] * downsample_factor), int(self.data['rect_data'].shape[1] * downsample_factor))
            downsample_data = np.zeros(downsize+(self.data['rect_data'].shape[2],))
            for i in range(self.data['rect_data'].shape[2]):
                downsample_data[:, :, i] = (cv2.resize(self.data['rect_data'][:,:,i], None, fx=downsample_factor, fy=downsample_factor, interpolation=cv2.INTER_AREA))
            self.data['rect_data'] = np.array(downsample_data)
        # change the first dimension to the last dimension
        self.data['rect_data'] = self.data['rect_data'].transpose(2, 0, 1)
        self.wall_resolution = self.data['rect_data'].shape[1]
        self.maxt = self.data['rect_data'].shape[0]
        
    def info(self):
        print('keys: ', self.data.keys())
        print('width: ', self.data['width'])
        print('wall_resolution: ', self.wall_resolution)
        print('maxt: ', self.maxt)
        print('bin_resolution: ', self.data['bin_resolution'])
        print('c: ', self.data['c'])
        print('rect_data.shape: ', self.data['rect_data'].shape)
        print('rect_data.sum: ', self.data['rect_data'].sum())
        print('rect_data.max: ', self.data['rect_data'].max())
        print('rect_data.min: ', self.data['rect_data'].min())
        
    
    def __len__(self):
        return self.data['rect_data'].shape[0]
    
    def __getitem__(self, idx):
        return self.data['rect_data'][:,:,idx]

    def get_wall_resolution(self):
        return self.wall_resolution

    def get_maxt(self):
        return self.maxt

    def get_data(self):
        return self.data
    
    def get_rect_data(self):
        return self.data['rect_data']
    
    def get_width(self):
        return self.data['width']
    
    def get_bin_resolution(self):
        return self.data['bin_resolution']
    
    def get_c(self):
        return self.data['c']
    
    def make_video(self, path='inputdata.avi'):
        make_video(self.data['rect_data'], path=path)

if __name__ == "__main__":
    data = load_data()
    print(data.keys())
    print(data['rect_data'].shape)
    print(data['width'])
    print(data['rect_data'].sum(), data['rect_data'].max(), data['rect_data'].min())
    make_video(data)
    # dataset = Dataset(data)
    # dataset.plot()