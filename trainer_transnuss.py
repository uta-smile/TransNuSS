import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from datasets.dataset_monuseg_images import Monuseg_image_dataset
from datasets.dataset_zenodo import Zenodo_dataset
from evaluate import compute_iou, compute_dice_score
from random import shuffle


def validate(validation_loader, model):
    iou_scores = []
    dice_scores = []

    for i_batch, sampled_batch in enumerate(validation_loader):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        image_batch = image_batch.cuda()

        pred, _, _, _, _ = model(image_batch)

        label = label_batch.data.cpu().numpy()
        label = label.squeeze()
        label_binarized = np.zeros_like(label)
        label_binarized[label > 0] = 1

        pred = pred.data.cpu().numpy()
        pred = pred.squeeze()
        pred_binarized = np.zeros_like(pred)
        threshold = 0.5
        pred_binarized[pred > threshold] = 1

        iou_score = compute_iou(label, pred_binarized)
        iou_scores.append(iou_score)

        dice_score = compute_dice_score(label, pred_binarized)
        dice_scores.append(dice_score)

    return np.mean(iou_scores), np.mean(dice_scores)


def region_triplet_loss(z_a, z_p, z_n):
    m1 = 0.1

    d_za_zp = (torch.sum((z_a - z_p)**2))
    d_za_zn = (torch.sum((z_a - z_n)**2))

    loss_val = torch.max(torch.tensor(0.0).cuda(), d_za_zp - d_za_zn + m1)

    return loss_val


# https://github.com/pytorch/pytorch/issues/1249
def dice_coef_loss(y_pred, y_true):
    smooth = 1.

    iflat = y_pred.view(-1)
    tflat = y_true.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


class predict_mlp(nn.Module):
    def __init__(self):
        super(predict_mlp, self).__init__()

        number_of_feature = 768

        self.fc1 = nn.Linear(number_of_feature*8, number_of_feature*2)
        self.fc2 = nn.Linear(number_of_feature*2, number_of_feature)

    def forward(self, nei_features):
        x = self.fc1(nei_features)
        x = self.fc2(x)

        return x


def transnuss_trainer(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = 0.01

    db_pretrain = Monuseg_image_dataset(data_path=args.root_path)
    print("The length of pretraining dataset is: {}".format(len(db_pretrain)))

    db_train_lab = Zenodo_dataset(data_path=args.root_path, split="train")
    print("The length of training dataset is: {}".format(len(db_train_lab)))

    db_val = Zenodo_dataset(data_path=args.root_path, split="validation")
    print("The length of validation dataset is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    pretrain_loader = DataLoader(db_pretrain, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn)
    train_loader = DataLoader(db_train_lab, batch_size=1, shuffle=True, num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn)
    validation_loader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn)

    # predict-net
    predict_net = predict_mlp()

    # scale-net
    scale_net = torchvision.models.resnet34(pretrained=True)
    num_ftrs_resnet34 = scale_net.fc.in_features
    scale_net.fc = torch.nn.Linear(in_features=num_ftrs_resnet34, out_features=5)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
        scale_net = nn.DataParallel(scale_net)

    model.train()

    predict_net.train()
    predict_net.cuda()

    scale_net.train()
    scale_net.cuda()

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_predict_net = optim.SGD(predict_net.parameters(), lr=0.001)
    optimizer_scale_net = torch.optim.Adam(scale_net.parameters(), lr=0.0001)

    mce_loss = torch.nn.CrossEntropyLoss()
    mae_loss = torch.nn.L1Loss()

    writer = SummaryWriter(snapshot_path + '/log')

    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))

    iterator = tqdm(range(max_epoch), ncols=70)

    best_val_iou = 0.0
    best_val_iou_epoch = 0

    batch_size = args.batch_size

    for epoch_num in iterator:
        # self-supervised pre-training
        if epoch_num < 20:
            model.train()
            predict_net.train()
            scale_net.train()

            for iter, sampled_batch in enumerate(pretrain_loader):
                image_batch_normal = sampled_batch['image_normal']
                image_batch_normal = image_batch_normal.cuda()
                _, _, height, width = image_batch_normal.size()

                image_batch_scaled = sampled_batch['image_scaled']
                image_batch_scaled = image_batch_scaled.cuda()
                scale_label = sampled_batch['scale_label'].cuda()

                output_n, _, feat_n, feat_n_r, _ = model(image_batch_normal)
                output_s, _, feat_s, feat_s_r, _ = model(image_batch_scaled)
                _, _, height_n, width_n = feat_n_r.size()
                _, _, height_s, width_s = feat_s_r.size()

                # region-level matching and corresponding losses
                loss_feature_prediction_n_batch = torch.tensor(0.0).cuda(0)
                loss_feature_prediction_s_batch = torch.tensor(0.0).cuda(0)
                loss_region_matching_batch = torch.tensor(0.0).cuda(0)

                for b in range(batch_size):
                    # normal image features
                    loss_feature_prediction_n = torch.tensor(0.0).cuda(0)
                    difficulty_n = np.zeros((height_n, width_n))

                    for r in range(1, height_n - 1):
                        for c in range(1, width_n - 1):
                            nei_features_n = torch.cat(
                                [feat_n_r[b, :, r - 1, c - 1], feat_n_r[b, :, r - 1, c], feat_n_r[b, :, r - 1, c + 1],
                                 feat_n_r[b, :, r, c - 1], feat_n_r[b, :, r, c + 1],
                                 feat_n_r[b, :, r + 1, c - 1], feat_n_r[b, :, r + 1, c], feat_n_r[b, :, r + 1, c + 1]])

                            # nei_features_n = nei_features_n.detach()
                            predicted_feat_n = predict_net(nei_features_n)

                            mae_predicted_n = mae_loss(predicted_feat_n, feat_n_r[b, :, r, c])
                            loss_feature_prediction_n += mae_predicted_n

                            difficulty_n[r, c] = mae_predicted_n

                    loss_feature_prediction_n /= ((height_n-2) * (width_n-2))
                    loss_feature_prediction_n_batch += loss_feature_prediction_n
                    difficulty_n = (difficulty_n - difficulty_n.min()) / (difficulty_n.max() - difficulty_n.min())

                    # normal foreground
                    threshold_n_fg = np.percentile(difficulty_n[1:height_n - 2, 1:width_n - 2], 95)
                    foreground_n = (difficulty_n >= threshold_n_fg)
                    indices_fg_normal = []

                    for r in range(1, height_n - 1):
                        for c in range(1, width_n - 1):
                            if foreground_n[r, c] == True:
                                indices_fg_normal.append((r, c))

                    shuffle(indices_fg_normal)

                    # normal background
                    threshold_n_bg = np.percentile(difficulty_n[1:height_n - 2, 1:width_n - 2], 5)
                    background_n = (difficulty_n <= threshold_n_bg) & (difficulty_n > 0)
                    indices_bg_normal = []

                    for r in range(1, height_n - 1):
                        for c in range(1, width_n - 1):
                            if background_n[r, c] == True:
                                indices_bg_normal.append((r, c))

                    shuffle(indices_bg_normal)

                    # scaled image features
                    loss_feature_prediction_s = torch.tensor(0.0).cuda(0)
                    difficulty_s = np.zeros((height_n, width_n))

                    for r in range(1, height_s - 1):
                        for c in range(1, width_s - 1):
                            nei_features_s = torch.cat(
                                [feat_s_r[b, :, r - 1, c - 1], feat_s_r[b, :, r - 1, c], feat_s_r[b, :, r - 1, c + 1],
                                 feat_s_r[b, :, r, c - 1], feat_s_r[b, :, r, c + 1],
                                 feat_s_r[b, :, r + 1, c - 1], feat_s_r[b, :, r + 1, c], feat_s_r[b, :, r + 1, c + 1]])

                            # nei_features_s = nei_features_s.detach()
                            predicted_feat_s = predict_net(nei_features_s)

                            mae_predicted_s = mae_loss(predicted_feat_s, feat_s_r[b, :, r, c])
                            loss_feature_prediction_s += mae_predicted_s

                            difficulty_s[r, c] = mae_predicted_s

                    loss_feature_prediction_s /= ((height_s-2) * (width_s-2))
                    loss_feature_prediction_s_batch += loss_feature_prediction_s
                    difficulty_s = (difficulty_s - difficulty_s.min()) / (difficulty_s.max() - difficulty_s.min())

                    # scaled foreground
                    threshold_s_fg = np.percentile(difficulty_s[1:height_s - 2, 1:width_s - 2], 95)
                    foreground_s = (difficulty_s >= threshold_s_fg)
                    indices_fg_scaled = []

                    for r in range(1, height_s - 1):
                        for c in range(1, width_s - 1):
                            if foreground_s[r, c] == True:
                                indices_fg_scaled.append((r, c))

                    shuffle(indices_fg_scaled)

                    # scaled background
                    threshold_s_bg = np.percentile(difficulty_s[1:height_s - 2, 1:width_s - 2], 5)
                    background_s = (difficulty_s <= threshold_s_bg) & (difficulty_s > 0)
                    indices_bg_scaled = []

                    for r in range(1, height_s - 1):
                        for c in range(1, width_s - 1):
                            if background_s[r, c] == True:
                                indices_bg_scaled.append((r, c))

                    shuffle(indices_bg_scaled)

                    # region-matching loss
                    loss_region_matching = torch.tensor(0.0).cuda(0)
                    len_indices = [len(indices_fg_normal), len(indices_bg_normal), len(indices_fg_scaled), len(indices_bg_scaled)]
                    len_indices_min = min(len_indices)
                    m_region = min(32, len_indices_min)

                    for m in range(m_region):
                        r_n, c_n = indices_fg_normal[m]
                        r_s, c_s = indices_fg_scaled[m]
                        prob_bg = random.random()

                        if prob_bg <= 0.5:
                            r_bg, c_bg = indices_bg_normal[m]
                            feat_bg = feat_n_r[b, :, r_bg, c_bg]
                        else:
                            r_bg, c_bg = indices_bg_scaled[m]
                            feat_bg = feat_s_r[b, :, r_bg, c_bg]

                        loss_region_matching += region_triplet_loss(feat_n_r[b, :, r_n, c_n], feat_s_r[b, :, r_s, c_s], feat_bg)

                    loss_region_matching /= float(m)
                    loss_region_matching_batch += loss_region_matching

                loss_feature_prediction_n_batch /= float(batch_size)
                loss_feature_prediction_s_batch /= float(batch_size)
                loss_region_matching_batch /= float(batch_size)

                # scale-loss
                J_scaled = image_batch_scaled * output_s
                scale_pred = scale_net(J_scaled)
                loss_scale = mce_loss(scale_pred, scale_label)

                loss = loss_region_matching_batch + (0.5 * loss_scale) + \
                       loss_feature_prediction_n_batch + loss_feature_prediction_s_batch

                optimizer.zero_grad()
                optimizer_predict_net.zero_grad()
                optimizer_scale_net.zero_grad()

                loss.mean().backward()

                optimizer.step()
                optimizer_predict_net.step()
                optimizer_scale_net.step()

                lr_ = base_lr * (1.0 - (iter+1) / max_iterations) ** 0.9

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                logging.info('epoch: %d, iteration: %d, loss_region_matching: %f, loss_scale: %f, '
                             'loss_feat_pred_normal: %f, loss_feat_pred_scaled: %f'
                             % (epoch_num+1, iter+1, loss_region_matching_batch.mean().item(), loss_scale.mean().item(),
                                loss_feature_prediction_n_batch.mean().item(), loss_feature_prediction_s_batch.mean().item()))

        # fine-tuning
        else:
            model.train()

            for iter, sampled_batch in enumerate(train_loader):
                image_batch_normal, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch_normal, label_batch = image_batch_normal.cuda(), label_batch.cuda()

                output_n, _, _, _, _ = model(image_batch_normal)

                loss_dice = dice_coef_loss(output_n, label_batch)
                loss = loss_dice

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lr_ = base_lr * (1.0 - (iter+1) / max_iterations) ** 0.9

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                writer.add_scalar('info/lr', lr_, iter+1)
                writer.add_scalar('info/total_loss', loss, iter+1)

                logging.info('epoch: %d, iteration: %d, loss_dice: %f' % (epoch_num+1, iter+1, loss_dice.item()))

        save_model_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num + 1) + '.pth')
        torch.save(model.state_dict(), save_model_path)

        logging.info("save model to {}".format(save_model_path))

        # validation
        model.eval()
        scale_net.eval()
        predict_net.eval()

        print('validating...')
        val_iou_score, val_dice_score = validate(validation_loader, model)

        if val_iou_score > best_val_iou:
            best_val_iou = val_iou_score
            best_val_iou_epoch = epoch_num + 1

            best_model_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)

            logging.info("save best model to {}".format(best_model_path))

        print('epoch:{0:3d}, val_iou_score: {1:.4f}, val_dice_score: {2:.4f}'.format(epoch_num+1, val_iou_score, val_dice_score))
        print('best_val_iou:{0:.4f}, best_val_iou_epoch{1:3d}'.format(best_val_iou, best_val_iou_epoch))

    writer.close()
    return "Training Finished!"
