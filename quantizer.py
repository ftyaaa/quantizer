import argparse
import os
from pathlib import Path

import open3d
import numpy as np
import torch
from tqdm import tqdm
import warnings


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--root_dir', type=str, help='root directory of dataset')
    parser.add_argument('--target_dir', type=str, help='target directory of quantized dataset')
    parser.add_argument('--sequences', type=str, default='', help='sequences to be quantized, separated by blankspace, leave blank to read all files')
    parser.add_argument('--original_precision', type=int, help='original precision of data')
    parser.add_argument('--target_precision', type=int, help='target precision of data')
    parser.add_argument('--original_data_format', type=str, default='ply', help='original data format, ply or npy')
    parser.add_argument('--target_data_format', type=str, default='ply', help='target data format, ply or npy')
    parser.add_argument('--write_ascii', type=bool, default=False, help='if target data format is ply, write_ascii or not')
    parser.add_argument('--cpu', type=bool, default=False)
    parser.add_argument('--gpu', type=str, default='0')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    scaling_factor = 2 ** (args.original_precision-args.target_precision)
    target_dir_ = Path(args.target_dir)
    target_dir_.mkdir(exist_ok=True)
    for sequence in (os.listdir(args.root_dir) if len(args.sequences)==0 else args.sequences.split(' ')):
        print(sequence)
        sequence_dir = os.path.join(args.root_dir, sequence)
        target_sequence_dir = os.path.join(args.target_dir, sequence)
        target_sequence_dir_ = Path(target_sequence_dir)
        target_sequence_dir_.mkdir(exist_ok=True)
        for file in tqdm(os.listdir(sequence_dir), smoothing=0.9):
            file_dir = os.path.join(sequence_dir, file)
            if args.original_data_format == 'ply':
                pc_ori = np.asarray(open3d.io.read_point_cloud(file_dir).points)
            elif args.original_data_format == 'npy':
                pc_ori = np.load(file_dir)
            else:
                raise RuntimeError('data format is limited to ply and npy')
            pc_ori_torch = torch.tensor(pc_ori)
            if pc_ori_torch.size(1) > 3:
                pc_ori_torch = pc_ori_torch[:, :3]
                warnings.warn('can only handle point clouds without attribute, assume that the first three are geometric coordinates and ignore the subsequent dimensions')
            if not args.cpu:
                pc_ori_torch = pc_ori_torch.cuda()
            pc_ori_torch_q = torch.unique(torch.floor(pc_ori_torch/scaling_factor), dim=0)
            pc_ori_q = pc_ori_torch_q.detach().cpu().numpy()
            target_file = file.replace('vox'+str(args.original_precision), 'vox'+str(args.target_precision))
            target_file = target_file.replace(args.original_data_format, args.target_data_format)
            target_file_dir = os.path.join(target_sequence_dir, target_file)
            if args.target_data_format == 'ply':
                pcd = open3d.geometry.PointCloud()  # 定义点云
                pcd.points = open3d.utility.Vector3dVector(pc_ori_q)
                open3d.io.write_point_cloud(target_file_dir, pcd, write_ascii=args.write_ascii)
            elif args.target_data_format == 'npy':
                np.save(target_file_dir, pc_ori_q)


