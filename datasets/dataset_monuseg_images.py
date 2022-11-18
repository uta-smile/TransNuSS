import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pickle
import random


class Monuseg_image_dataset(Dataset):
    def __init__(self, data_path):
        dataset_name = 'monuseg_tiles_512x512'
        dataset_path = os.path.join(data_path, dataset_name)

        wsi_folders = sorted(os.listdir(dataset_path))

        self.image_filepaths = []

        for wsi_folder in wsi_folders:
            wsi_folder_path = os.path.join(dataset_path, wsi_folder)

            if os.path.isdir(wsi_folder_path):
                scale_folders = sorted(os.listdir(wsi_folder_path))

                for scale_folder in scale_folders:
                    scale_folder_path = os.path.join(wsi_folder_path, scale_folder)
                    image_filenames = sorted(os.listdir(scale_folder_path))

                    for image_filename in image_filenames:
                        image_filepath = os.path.join(scale_folder_path, image_filename)
                        self.image_filepaths.append(image_filepath)

        mean_std_file = os.path.join(dataset_path, 'monuseg_tiles_mean_std.txt')

        with open(mean_std_file, 'rb') as handle:
            data_mean_std = pickle.loads(handle.read())

        self.mean_value = data_mean_std['mean_train_images']
        self.std_value = data_mean_std['std_train_images']

    def __len__(self):
        return len(self.image_filepaths)

    def get_scaled_image(self, image, scale):
        width, height = image.size

        crop_width = width // scale
        crop_height = height // scale

        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)

        image_scaled = image.crop((left, top, left + crop_width, top + crop_height))
        image_scaled = image_scaled.resize((512, 512), Image.BICUBIC)

        return image_scaled, left, top, left + crop_width, top + crop_height

    def __getitem__(self, index):
        image_filepath = self.image_filepaths[index]
        image_filename = image_filepath.split('/')[-1]

        image_original = Image.open(image_filepath)
        image = np.array(image_original).astype('float32')
        image = ((image - self.mean_value) / self.std_value)
        image = image.transpose((2, 0, 1))

        scales = [1.0, 1.25, 1.5, 1.75, 2.0]
        scale_label = random.randint(0, 4)
        scale = scales[scale_label]

        image_scaled, left, top, right, bottom = self.get_scaled_image(image_original, scale)
        image_scaled = np.array(image_scaled).astype('float32')
        image_scaled = ((image_scaled - self.mean_value) / self.std_value)
        image_scaled = image_scaled.transpose((2, 0, 1))

        sample = {'image_normal': image,}
        sample['case_name_normal'] = image_filename
        sample['image_scaled'] = image_scaled
        sample['case_name_scale'] = image_filename
        sample['scale_label'] = scale_label
        sample['crop_coord_left'] = left
        sample['crop_coord_top'] = top
        sample['crop_coord_right'] = right
        sample['crop_coord_bottom'] = bottom

        return sample
