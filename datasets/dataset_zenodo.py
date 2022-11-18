import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pickle


class Zenodo_dataset(Dataset):
    def __init__(self, data_path, split):
        self.split = split
        dataset_name = 'zenodo'
        dataset_path = os.path.join(data_path, dataset_name)

        mean_std_file = os.path.join(dataset_path, dataset_name + '_mean_std.txt')

        with open(mean_std_file, 'rb') as handle:
            data_mean_std = pickle.loads(handle.read())

        if split == 'train':
            self.image_dir = os.path.join(dataset_path, 'train_images')
            self.mean_value = data_mean_std['mean_train_images']
            self.std_value = data_mean_std['std_train_images']
        elif split == 'validation':
            self.image_dir = os.path.join(dataset_path, 'validation_images')
            self.mean_value = data_mean_std['mean_val_images']
            self.std_value = data_mean_std['std_val_images']
        elif split == 'test':
            self.image_dir = os.path.join(dataset_path, 'test_images')
            self.mean_value = data_mean_std['mean_val_images']
            self.std_value = data_mean_std['std_val_images']

        self.image_files = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_filename = self.image_files[index]
        image_filepath = os.path.join(self.image_dir, image_filename)

        mask_dir = self.image_dir.replace("_images", "_masks")
        mask_filename = image_filename.replace("_image.png", "_mask.png")
        # mask_filename = image_filename.replace(".tif", ".png") # for psb dataset
        mask_filepath = os.path.join(mask_dir, mask_filename)

        image = Image.open(image_filepath)
        image = np.array(image).astype('float32')
        image = ((image - self.mean_value) / self.std_value)
        image = image.transpose((2, 0, 1))

        label = Image.open(mask_filepath)
        label = np.array(label.convert('1')).astype('uint8')

        sample = {'image': image, 'label': label}
        sample['case_name'] = self.image_files[index]

        return sample
