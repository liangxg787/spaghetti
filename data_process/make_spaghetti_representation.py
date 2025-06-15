# -*- coding: UTF-8 -*-
"""
@Time : 15/06/2025 17:16
@Author : Xiaoguang Liang
@File : make_spaghetti_representation.py
@Project : spaghetti
"""
import os
import numpy as np
from pathlib import Path

import torch
from h5py import File
from tqdm import tqdm
from loguru import logger
from spaghetti import options
from spaghetti.shape_inversion import MeshProjectionMid


EPOCHS = 150


def merge_zh_step_a_v2(gmms):
    b, gp, g, _ = gmms[0].shape
    mu, p, phi, eigen = [item.view(b, gp * g, *item.shape[3:]) for item in gmms]
    p = p.reshape(*p.shape[:2], -1)
    # z_gmm.shape = (1, 16, 16)
    z_gmm = torch.cat((mu, p, phi.unsqueeze(-1), eigen), dim=2).detach()
    # z_gmm = self.from_gmm(z_gmm) # z_gmm.shape = (1, 16, 512)
    return z_gmm


def make_representation_for_one(model_name, obj_file_path, output_name, num_epochs=EPOCHS):
    opt = options.Options(tag=model_name)
    if not os.path.exists(output_name):
        os.makedirs(output_name)
    if not os.path.exists(obj_file_path):
        raise FileNotFoundError(f'{obj_file_path} not found!')
    model = MeshProjectionMid(opt, obj_file_path, output_name)

    # predict with spaghetti
    for i in range(num_epochs // 2):
        if model.early_stop(model.train_epoch(i), i):
            break
    model.switch_embedding()
    for i in range(num_epochs):
        if model.early_stop(model.train_epoch(i), i):
            break

    z_d = model.mid_embeddings(
        torch.zeros(1, device=model.device, dtype=torch.int64)
    ).view(1, model.opt.num_gaussians, -1)
    zh_base, gmms = model.model.decomposition_control.forward_mid(z_d)
    z_gmm = merge_zh_step_a_v2(gmms)

    zh_base = zh_base.cpu().detach().numpy()
    zh_base = zh_base.reshape(16, 512)
    z_gmm = z_gmm.cpu().detach().numpy()
    z_gmm = z_gmm.reshape(16, 16)
    return zh_base, z_gmm


def make_representation(model_name, output_name, manifold_files_path):
    sub_folders = os.listdir(manifold_files_path)
    all_zh_base = []
    all_z_gmm = []
    for folder in tqdm(sub_folders):
        sub_path = os.path.join(manifold_files_path, folder)
        if os.path.isdir(sub_path):
            files = list(Path(sub_path).glob('*manifold.obj'))
            if not files:
                raise FileNotFoundError(f'manifold.obj is not found in {sub_path}!')
            manifold_file = str(files[0])
            logger.info(f'Processing {manifold_file} ...')
            zh_base, z_gmm = make_representation_for_one(model_name, manifold_file, output_name)
            all_zh_base.append(zh_base)
            all_z_gmm.append(z_gmm)
    all_zh_base = np.array(all_zh_base)
    all_z_gmm = np.array(all_z_gmm)
    # Calculate global men for gmms
    z_gmm_mean = np.mean(all_z_gmm, axis=(0, 1))
    # Calculate global standard deviation for gmms
    z_gmm_std = np.std(all_z_gmm, axis=(0, 1))
    # Save data
    logger.info('Save all data ...')
    save_file_name = 'spaghetti_chair_latents_10_samples_mean_std'
    data = {'s_j_affine': all_zh_base, 'g_js_affine': all_z_gmm, 'mean': z_gmm_mean, 'std': z_gmm_std}
    with File(f"{output_name}/{save_file_name}.hdf5", "w") as f:
        for k, v in data.items():
            f[k] = v


if __name__ == '__main__':
    model_name = 'chairs_large'
    BASE_DIR = '../../'

    obj_file_path = BASE_DIR + '/dataset/03001627_10/1a6f615e8b1b5ae4dbbc9440457e303e/model_manifold.obj'
    output_name = BASE_DIR + '/output/03001627_10'
    # make_representation_for_one(model_name, obj_file_path, output_name)

    manifold_shapeNet = BASE_DIR + '/dataset/03001627_10'
    make_representation(model_name, output_name, manifold_shapeNet)