import cv2
import torch
import os
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import time
from Model.triplanelite import triplane_fea
from Dataloader.LLFF import LLFFDataset
from Teacher.dino import get_dino_model
from Processing.vis import load_settings
from Processing.trainer import *
from Processing.rendering import novel_views_LLFF, creat_video, render_img
from Processing.vis import load_settings, calc_query_emb, calc_feature_dist
import argparse

import sys

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def save_training_history(folderpath, History, method):
    plt.figure(figsize=(18,11))
    plt.subplot(2, 3, 1)
    plt.plot(History['train_loss'], label= method)
    plt.title('train loss')
    plt.subplot(2, 3, 2)
    plt.plot(History['train_img_loss'], label= method)
    plt.title('train img loss')
    plt.subplot(2, 3, 3)
    plt.plot(History['train_fea_loss'], label= method)
    if method == 'wo_distill_':
        plt.title('fea loss not use in first training step')
    else:
        plt.title('train fea loss')
    plt.subplot(2, 3, 4)
    plt.plot(History['train_PSNR'], label= method)
    # plt.ylim(19, 26)
    plt.title('train PSNR')
    plt.subplot(2, 3, 5)
    plt.plot(History['train_TV_loss'], label= method)
    plt.title('train TV loss')
    plt.subplot(2, 3, 6)
    plt.plot(History['train_L1_loss'], label= method)
    plt.title('train L1 loss')
    plt.legend()
    plt.savefig(os.path.join(folderpath, method + '_train_history.png'))
    plt.close()
    
    plt.figure()
    plt.plot(History['test_img_loss'], label= method)
    plt.savefig(os.path.join(folderpath, method + '_test_img_loss.png'))
    plt.close()

def train_scene(scene, Training_floder, datadir, fea_dir,
                max_steps, max_steps_distill, batchsize, 
                lambda_fea, hn, hf, nb_bins, downsample,
                lambda_TV, lambda_l1, gradient_scaling, 
                needParaprefix, show_training_figures_videos):
    # 0. creat folder    
    print('0. create folder')
    parameter_name = '_lambda_TV_' + str(lambda_TV) + '_lambda_l1_' + str(lambda_l1) + '_gradient_scaling_' + str(gradient_scaling)
    if needParaprefix:
        foldername = scene +  '_' + parameter_name + '_' + time.strftime("%Y%m%d-%H%M%S")
    else:
        foldername = time.strftime("%Y%m%d-%H%M%S")
    
    
    assert os.path.exists(Training_floder), 'Training_floder ${Training_floder} does not exist'
    folderpath = os.path.join(Training_floder, foldername)
    if not os.path.exists(folderpath):
        os.mkdir(folderpath)

    # 1. load data
    print('1. load data')
    
    LLFF_training = LLFFDataset(datadir, fea_dir, split='train',load_features=True, downsample = downsample)
    LLFF_test = LLFFDataset(datadir, fea_dir, split='test',load_features=True, downsample = downsample)
    trainingSampler = SimpleSampler(len(LLFF_training), batchsize)

    # 2. init model
    print('2. init model')
    aabb = torch.tensor([-1.7, 1.7])
    nerf_model = triplane_fea(aabb = aabb)
    model = nerf_model.to(device)
    model_optimizer = torch.optim.Adam([
        {'params': model.grids.parameters(), 'lr': 4e-3},
        {'params': model.sigma_net.parameters(), 'lr': 4e-4},
        {'params': model.fea_net.parameters(), 'lr': 4e-4},
        {'params': model.color_net.parameters(), 'lr': 4e-4},
        ])

    # 3. train setting
    print('3. train setting')
    scheduler = get_cosine_schedule_with_warmup(
                    model_optimizer, num_warmup_steps=512, num_training_steps=max_steps)

    # 4. train wo distill
    print('rgb_density training')
    name = 'rgb_density training'
    History = train(name, model, model_optimizer, scheduler, 
                    LLFF_training, testdataset = LLFF_test, trainingSampler = trainingSampler, folderpath = folderpath, 
                    iterations = max_steps, device=device, hn=hn, hf=hf, nb_bins = nb_bins, lambda_fea = lambda_fea,
                    lambda_TV = lambda_TV, lambda_l1 = lambda_l1, gradient_scaling=gradient_scaling)
    
    
    
    if show_training_figures_videos:
        # 4.2 rendering training results
        save_training_history(folderpath, History, name)
        print('4.2 rendering training results')
        novel_views_path = novel_views_LLFF(folderpath, name, nerf_model, device, LLFF_test, hn = 0, hf = 1, nb_bins = 96,
                            req_others = True)
        creat_video(novel_views_path, folderpath, name, req_others=True)

    # 5. train with distill
    print('distill features training')
    name = 'feature_train'
    model_optimizer = torch.optim.Adam(model.parameters(), lr = 5e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[50000], gamma=0.5)
    History_2 = train(name, model, model_optimizer, scheduler, 
                    LLFF_training, trainingSampler = trainingSampler,
                    testdataset = LLFF_test, folderpath = folderpath, 
                    iterations = max_steps_distill, device=device, hn=hn, hf=hf, nb_bins = nb_bins, lambda_fea = lambda_fea,
                    lambda_TV = lambda_TV, lambda_l1 = lambda_l1, distill_features=True, gradient_scaling=gradient_scaling)
    
    
    if show_training_figures_videos:
        save_training_history(folderpath, History_2, name)
        #5.2 rendering distill training results
        print('5.2 rendering distill training results')
        novel_views_path = novel_views_LLFF(folderpath, name, nerf_model, device, LLFF_test, hn = 0, hf = 1, nb_bins = 96,
                            req_others = True)
        creat_video(novel_views_path, folderpath, name, req_others=True)
    
        # 6. run selection
        print('6. run selection')
        settings = load_settings()[scene]
        factor = downsample
        r, c = settings['rc']
        extent = settings['sz']
        r = int(r * 8 / factor)
        c = int(c * 8 / factor)
        extent = int(extent * 8 / factor)
        img_w, img_h = LLFF_test.img_wh
        rgb_flower, emb_flower, depth_flower, mask_flower = render_img(nerf_model = nerf_model, device= device, Dataset = LLFF_test, img_index = 0, hn = 0, hf = 1, nb_bins = 96, req_others=True)
        embq, dir_q = calc_query_emb(emb_flower, r, c, extent, rgb=rgb_flower)
        dist = calc_feature_dist(embq, emb_flower)
        plt.figure(figsize=(4,3))
        plt.hist(dist.view(-1).cpu().numpy(), bins=20, density=True, alpha=0.5, label='Ditilled Triplanes')
        plt.savefig(os.path.join(folderpath, 'dist_hist.png'))
        plt.close()
        rgb_j_fg, emb_j_fg, depth_j_fg, mask_j_fg = render_img(nerf_model = nerf_model, device= device, \
                                Dataset = LLFF_test, img_index = 0, hn = 0, hf = 1, nb_bins = 96,\
                                req_others = True, embq=embq, dis_thr=settings['thr'] + settings['margin'], 
                                foreground=False, show_selection=True)
        
        plt.figure(figsize=(20, 8))
        plt.subplot(1, 3, 1)
        plt.imshow(rgb_j_fg)
        plt.subplot(1, 3, 2)
        plt.imshow(depth_j_fg, cmap = 'gray')
        plt.subplot(1, 3, 3)
        plt.imshow(mask_j_fg, cmap = 'gray')
        plt.savefig(os.path.join(folderpath, 'selection.png'))
        plt.close()
        
        name = 'w_selection'
        novel_views_path = novel_views_LLFF(folderpath, name, nerf_model, device, LLFF_test, hn = 0, hf = 1, nb_bins = 96,
                            req_others = True, dis_thr = settings['thr'] + settings['margin'], 
                            embq = embq, dist_less=False, show_selection = True)
        creat_video(novel_views_path, folderpath, name, req_others=True)
    
    
if __name__ == '__main__':
    # script input is scene name
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20231227)
    np.random.seed(20231227)
    
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('--lambdaTV', type=float, help='lambda_TV', default=1)
    parser.add_argument('--lambdal1', type=float, help='lambda_l1', default=1e-3)
    parser.add_argument('--maxsteps', type=int, help='max_steps', default=40000)
    parser.add_argument('--maxstepsdistill', type=int, help='max_steps_distill', default=7000)
    parser.add_argument('--lambdafea', type=float, help='lambda_fea', default=0.04)
    parser.add_argument('--hn', type=int, help='hn', default=0)
    parser.add_argument('--hf', type=int, help='hf', default=1)
    parser.add_argument('--nbbins', type=int, help='nb_bins', default=96)
    parser.add_argument('--scene', type=str, help='scene', default='flower')
    parser.add_argument('--gradientscaling', type=bool, help='gradient_scaling', default=False)
    parser.add_argument('--needParaprefix', type=bool, help='needParaprefix', default=False)
    parser.add_argument('--Trainingfloder', type=str, help='Training_floder path', default='./training_results/')
    parser.add_argument('--datadir', type=str, help='datadir path', default='./Dataset/nerf_llff_data/')
    parser.add_argument('--feadir', type=str, help='fea_dir path', default='./Dataset/nerf_llff_data/fea')
    parser.add_argument('--batchsize', type=int, help='batchsize', default=512)
    parser.add_argument('--downsample', type = int, help= 'downsampling of your LLFF dataset', default = 8)
    parser.add_argument('--show_results', type = bool, \
                        help= 'If you would like to see training figures, rendering results, make it to ture. (take more time)', default = False)

    args = parser.parse_args()

    Training_floder = args.Trainingfloder
    if not os.path.exists(Training_floder):
        os.mkdir(Training_floder)

    datadir = args.datadir + args.scene
    fea_dir = os.path.join(args.feadir, 'DINO_' + args.scene + '_64.pt')

    print(args)
    scene = args.scene
    max_steps = args.maxsteps
    max_steps_distill = args.maxstepsdistill
    lambda_fea = args.lambdafea
    hn = args.hn
    hf = args.hf
    nb_bins = args.nbbins
    lambda_TV = args.lambdaTV
    lambda_l1 = args.lambdal1
    gradient_scaling = args.gradientscaling
    
    train_scene(scene, Training_floder, datadir, fea_dir, 
                max_steps = max_steps, max_steps_distill = max_steps_distill, 
                lambda_fea = lambda_fea, downsample=args.downsample,
                hn = hn, hf = hf, nb_bins = nb_bins, lambda_TV = lambda_TV, 
                lambda_l1 = lambda_l1, gradient_scaling = gradient_scaling, 
                needParaprefix = args.needParaprefix, batchsize = args.batchsize,
                show_training_figures_videos = args.show_results)