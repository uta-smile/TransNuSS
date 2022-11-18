# author: Mohammad Minhazul Haq
# created on: February 12, 2022

import os
from PIL import Image
from skimage.color import rgb2hsv
import numpy as np
import pickle

DATASET_PATH = 'monuseg_tiles_512x512'


def remove_background_patches():
    wsi_folders = sorted(os.listdir(DATASET_PATH))

    for wsi_folder in wsi_folders:
        wsi_folder_path = os.path.join(DATASET_PATH, wsi_folder)
        scale_folders = sorted(os.listdir(wsi_folder_path))

        for scale_folder in scale_folders:
            scale_folder_path = os.path.join(wsi_folder_path, scale_folder)
            print('processing ', scale_folder_path)

            image_filenames = sorted(os.listdir(scale_folder_path))

            for image_filename in image_filenames:
                image_filepath = os.path.join(scale_folder_path, image_filename)
                image = Image.open(image_filepath)
                image_hsv = rgb2hsv(image)

                mask_s = (image_hsv[:, :, 1] >= 0.05)
                mask_s_val = (np.sum(mask_s) / (512 * 512))

                mask_v = (image_hsv[:, :, 2] <= 0.9)
                mask_v_val = (np.sum(mask_v) / (512 * 512))

                if image.size != (512, 512):
                    os.remove(image_filepath)
                elif mask_s_val < 0.7 and mask_v_val < 0.7:
                    os.remove(image_filepath)


def compute_mean_std():
    dataset_name = 'monuseg_tiles_512x512'
    dataset_path = os.path.join('data', dataset_name)

    means_train_images = []
    stds_train_images = []
    total_images = 0

    mean_std_filename = 'monuseg_tiles_mean_std.txt'

    wsi_folders = sorted(os.listdir(dataset_path))

    for wsi_folder in wsi_folders:
        print(wsi_folder, ' processing...')

        wsi_folder_path = os.path.join(dataset_path, wsi_folder)
        scale_folders = sorted(os.listdir(wsi_folder_path))

        for scale_folder in scale_folders:
            scale_folder_path = os.path.join(wsi_folder_path, scale_folder)
            image_filenames = sorted(os.listdir(scale_folder_path))

            for image_filename in image_filenames:
                image_filepath = os.path.join(scale_folder_path, image_filename)
                image = np.array(Image.open(image_filepath))

                avg = np.mean(image)
                std = np.std(image)

                means_train_images.append(avg)
                stds_train_images.append(std)

    # save mean, std to file
    data_to_save = {'mean_train_images': np.mean(means_train_images),
                    'std_train_images': np.mean(stds_train_images)}

    with open(os.path.join(dataset_path, mean_std_filename), 'wb') as handle:
        pickle.dump(data_to_save, handle)

    print('Total images: ', total_images)
    print('success')


if __name__ == '__main__':
    remove_background_patches()
    compute_mean_std()
