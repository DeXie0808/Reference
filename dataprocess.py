import h5py
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pylab import *
import os
from config import cfg
from easydict import EasyDict


class LoadData():
    def __init__(self, h5_path, datapath):
        F1 = h5py.File(os.path.join(h5_path, 'Image.h5'), 'r')
        self.train_x_list = []
        self.retrieval_x_list = []
        self.query_x_list = []
        train_x_list = F1['ImgTrain'][:]

        for trainidx in train_x_list:
            newpath = os.path.join(datapath, trainidx)
            self.train_x_list.append(newpath)
        self.train_x_list = np.array(self.train_x_list)
        retrieval_x_list = F1['ImgDataBase'][:]

        for retrievaligx in retrieval_x_list:
            newpath = os.path.join(datapath, retrievaligx)
            self.retrieval_x_list.append(newpath)
        self.retrieval_x_list = np.array(self.retrieval_x_list)
        query_x_list = F1['ImgQuery'][:]

        for queryidx in query_x_list:
            newpath = os.path.join(datapath, queryidx)
            self.query_x_list.append(newpath)
        self.query_x_list = np.array(self.query_x_list)
        F1.close()

        F2 = h5py.File(os.path.join(h5_path, 'Text.h5'), 'r')
        self.train_y = F2['TagTrain'][:]
        self.retrieval_y = F2['TagDataBase'][:]
        self.query_y = F2['TagQuery'][:]
        F2.close()
        F3 = h5py.File(os.path.join(h5_path, 'Label.h5'), 'r')
        self.train_L = F3['LabTrain'][:]
        self.retrieval_L = F3['LabDataBase'][:]
        self.query_L = F3['LabQuery'][:]
        F3.close()

    def __call__(self, *args, **kwargs):
        pass

def LoadImg(pathList, crop_size = 224):
    if type(pathList) is not np.ndarray:
        pathList = np.array([pathList])
    else:
        pathList = pathList
    ImgSelect = np.ndarray([len(pathList), crop_size, crop_size, 3])
    count = 0
    for idx in range(len(pathList)):
        img = Image.open(pathList[idx])
        xsize, ysize = img.size
        nulArray = np.zeros([crop_size,crop_size,3])
        seldim = max(xsize, ysize)
        rate = float(crop_size) / seldim
        nxsize = int(xsize * rate)
        nysize = int(ysize * rate)
        if nxsize %2 != 0:
            nxsize = int(xsize*rate) + 1
        if nysize %2 != 0:
            nysize = int(ysize*rate) + 1
        img = img.resize((nxsize, nysize))
        nxsize, nysize = img.size
        img = img.convert("RGB")
        img = array(img)
        nulArray[int(112-nysize/2) :int(112+nysize/2), int(112-nxsize/2) :int(112+nxsize/2), :] = img
        if nulArray.shape[2] != 3:
            print('This image is not a rgb picture: {0}'.format(pathList[idx]))
            print('The shape of this image is {0}'.format(nulArray.shape))
            ImgSelect[count, :, :, :] = nulArray[:, :, 0:3]
            count += 1
        else:
            ImgSelect[count, :, :, :] = nulArray
            count += 1
    return ImgSelect


def split_data(data):
    dataset = EasyDict()
    loaddata = LoadData(h5_path=data.h5_path, datapath=data.data_path)
    dataset.train_x = loaddata.train_x_list
    dataset.train_y = loaddata.train_y.astype(float)
    dataset.train_L = loaddata.train_L.astype(float)
    dataset.query_x = loaddata.query_x_list
    dataset.query_y = loaddata.query_y.astype(float)
    dataset.query_L = loaddata.query_L.astype(float)
    dataset.retrieval_x = loaddata.retrieval_x_list
    dataset.retrieval_y = loaddata.retrieval_y.astype(float)
    dataset.retrieval_L = loaddata.retrieval_L.astype(float)
    return dataset

if __name__ == '__main__':
    dataset = split_data(cfg.data)
    print(dataset.keys())