import numpy as np
from glob import glob

import torch
from natsort import natsorted
import h5py

from torch.utils.data import Dataset

from preprocessing import preprocessing_steps
from transforms import get_transforms
from utils.rectangular_masks import RectangularMasks



class GMDChargeLabelsSeededDataset(Dataset):

    def __init__(self,
                 main_path: str,
                 partnumbers: list,
                 file_head_name: str,
                 data_shape: list,
                 len_data: int,
                 max_nr_images: int,
                 preprocessing: list,
                 preprocessing_kwargs: dir,
                 transformations: list,
                 transformation_kwargs: dir,
                 mask_kwargs: dir,
                 h5paths: {}
                 ):

        self.main_path = main_path
        self.partnumbers = partnumbers
        self.file_head_name = file_head_name
        self.data_shapes = data_shape
        self.len_data = len_data
        self.max_nr_images = max_nr_images
        self.preprocessing = preprocessing
        self.preprocessing_kwargs = preprocessing_kwargs
        self.transformations = transformations
        self.transformation_kwargs = transformation_kwargs
        self.h5_paths = h5paths

        self.polarix = None
        self.gmd = None
        self.charge = None
        self.filelist = None
        self.transformlist = None

        self.fill_data()
        self.build_transformation()
        self.maskcreator = RectangularMasks(**mask_kwargs)

        assert self.gmd.shape[0] == self.polarix.shape[0]

        self.len = self.gmd.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        data = self.transformlist(self.polarix[index])
        mask = self.maskcreator.get_mask()
        return data, mask

    def fill_data(self):

        filelist = []
        for partnumber in self.partnumbers:
            filelist.extend(glob(self.main_path + self.file_head_name + str(partnumber) + '*.h5'))

        self.filelist = natsorted(filelist)
        print(str(len(filelist)) + 'files found')


        ## fill polarix images and other data
        im_high = self.data_shapes[0]
        im_arr_all = np.zeros((1, im_high, self.data_shapes[1]))
        gmd_all = np.zeros(1)
        charge_all = np.zeros(1)

        stop = False

        for filestr in self.filelist:

            file = h5py.File(filestr, 'r')
            path = self.h5_paths['polarix']
            if (path in file) * (not stop):
                im_arr_raw = file[path][:self.max_nr_images]
                if self.max_nr_images:
                    stop = True
                    self.filelist = [filestr]
                    print('stop reading files, as max number is set')


                im_arr, ind_all = preprocessing_steps(im_arr_raw, self.preprocessing, self.preprocessing_kwargs, runnumber=-1)
                im_arr_all = np.vstack((im_arr_all, im_arr))


                h5_gmdpath = self.h5_paths['gmd']
                h5_chargepath = self.h5_paths['charge']


                gmd = np.mean(np.array(file[h5_gmdpath]),axis = 1)[:self.max_nr_images] ##here we have 4 gmd values.. I have to find out with mean? is good
                charge = np.array(file[h5_chargepath])[:self.max_nr_images]

                assert np.all(charge > 0)

                gmd_all = np.concatenate((gmd_all, gmd))
                charge_all = np.concatenate((charge_all, charge))

            else:
                if not stop:
                    print(f'Warning: Skip file {filestr}, no polarix data found')

            file.close()

        self.polarix = im_arr_all[1:]
        self.gmd = gmd_all[1:]
        self.charge = charge_all[1:]

    def build_transformation(self):
        self.transformlist = get_transforms(self.transformations,self.transformation_kwargs)
