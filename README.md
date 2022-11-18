# TransNuSS
This repo holds code for [Self-supervised Pre-training for Nuclei Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-16434-7_30)

## Usage

### 1. Download Google pre-trained ViT models
Download "R50-ViT-B_16" from https://console.cloud.google.com/storage/vit_models/.
Put the downloaded model weights file at "model/vit_checkpoint/imagenet21k/".

### 2. Requirements
- Python 3.6.13
- PyTorch 1.10.2

### 3. Dataset

#### Pre-training dataset
1. Download the *.svs files (which are listed in wsi_list.txt) from https://portal.gdc.cancer.gov/.
These are the Whole Slide Images (WSI).
2. Put the downloaded WSIs at data/MoNuSeg_WSI/
3. Then, run "preprocessing/make_tiles.py" to extract patches from the downloaded WSIs. This 
will save the extracted patches in "monuseg_tiles_512x512" folder.
4. Run "MoNuSeg_dataset_builder.py"

#### Fine-tuning dataset
1. Download [TNBC dataset](https://zenodo.org/record/1175282#.YMisCTZKgow)
2. Put the downloaded dataset at "data/zenodo/"
3. Split the images and masks into "train_images", "train_masks", "validation_images", 
"validation_masks", "test_images", and "test_masks" folder.

### 4. Train
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --root_path data --batch_size 2 --vit_name R50-ViT-B_16
```

### 5. Test
```bash
python test.py --vit_name R50-ViT-B_16
```

### 6. Evaluate
```bash
python evaluate.py
```

## Citations
```bibtex
@inproceedings{haq2022self,
  title={Self-supervised Pre-training for Nuclei Segmentation},
  author={Haq, Mohammad Minhazul and Huang, Junzhou},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={303--313},
  year={2022},
  organization={Springer}
}
```
