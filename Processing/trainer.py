import torch
import numpy as np
import math
from tqdm import tqdm
from Processing.rendering import render_rays
import os
import copy

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(x))


def compute_plane_tv(t):
    # implement by k-planes
    # https://github.com/sarafridov/K-Planes
    batch_size, c, h, w = t.shape
    count_h = batch_size * c * (h - 1) * w
    count_w = batch_size * c * h * (w - 1)
    h_tv = torch.square(t[..., 1:, :] - t[..., :h-1, :]).sum()
    w_tv = torch.square(t[..., :, 1:] - t[..., :, :w-1]).sum()
    return 2 * (h_tv / count_h + w_tv / count_w) 

def compute_plane_l1(t):
    return torch.mean(torch.abs(t))


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    eta_min: float = 0.0,
    num_cycles: float = 0.999,
    last_epoch: int = -1,
):
    """
    https://github.com/huggingface/transformers/blob/bd469c40659ce76c81f69c7726759d249b4aef49/src/transformers/optimization.py#L129
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(eta_min, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

class SimpleSampler:
    # implement by CCNeRF
    # https://github.com/ashawkey/CCNeRF/blob/a7fe5ba720f40f21fe666de2786674c32b179e4f/train.py#L18
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]
    
    
def train(method, nerf_model, optimizer, scheduler, trainingdata, trainingSampler, 
          testdataset, folderpath, device='cuda:0', hn=0, hf=1, iterations=300000, 
          nb_bins=64, lambda_fea = 0.04, lambda_TV = 0.1, lambda_l1 = 1e-4,
          distill_features = False, blender = False, gradient_scaling = False):
    
    History = {'train_loss':[], 'test_loss':[], 'train_img_loss':[], 'test_img_loss':[], 
               'train_fea_loss':[], 'test_fea_loss':[], 'train_PSNR':[], 'test_PSNR':[],
               'train_TV_loss':[], 'test_TV_loss':[], 'train_L1_loss':[], 'test_L1_loss':[]}

    for _ in tqdm(range(iterations)):
        
        nerf_model.train()
        # get batch data
        bray_idx = trainingSampler.nextids()
        # Get data from data loader
        rays = trainingdata[bray_idx]['rays']
        ray_origins = rays[:, :3].to(device)
        ray_directions = rays[:, 3:6].to(device)
        gt_rgb = trainingdata[bray_idx]['rgbs'].to(device)
        
        # Rendering & run model
        if distill_features == True:            
            gt_fea = trainingdata[bray_idx]['features'].to(device)
            render_rgb, render_fea, render_depth, render_mask = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins, req_others = True, blender=blender,
                                                                            gradient_scaling = gradient_scaling)
        else:
            render_rgb = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins, blender=blender,
                                     gradient_scaling = gradient_scaling)    
        
        # Get loss
        img_loss = img2mse(render_rgb, gt_rgb)
            
        # print('training: rgb vs. ft', loss, loss_distillation, loss / loss_distillation)
        TV_loss = 0
        n_grid = len(nerf_model.grids[0])
        for grid in nerf_model.grids[0]:
            TV_loss = TV_loss + compute_plane_tv(grid)
        TV_loss = TV_loss / n_grid
        
        L1_loss = 0 
        for grid in nerf_model.grids[0]:
            L1_loss = L1_loss + compute_plane_l1(grid)
        L1_loss = L1_loss / n_grid
        
        loss = img_loss + lambda_TV * TV_loss + lambda_l1 * L1_loss
        if distill_features: 
            distances = ((render_fea - gt_fea) ** 2).sum(dim=1)
            loss_distillation = distances.mean()
            loss = loss + loss_distillation * lambda_fea        
            History['train_fea_loss'].append(loss_distillation.item())
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Record loss
        History['train_loss'].append(loss.item())
        History['train_img_loss'].append(img_loss.item())
        History['train_PSNR'].append(mse2psnr(img_loss).item())
        History['train_TV_loss'].append(TV_loss.item())
        History['train_L1_loss'].append(L1_loss.item())
            
        scheduler.step()
        
        # evaluate 2 times
        if (_ + 1) % (iterations // 2) == 0:
            nerf_model.eval()
            with torch.no_grad():
                H=testdataset.img_wh[1]
                W=testdataset.img_wh[0]
                chunk_size=20
                n_images = len(testdataset) // (testdataset.img_wh[0] * testdataset.img_wh[1])
                
                History_epoch = {'test_loss':[], 'test_img_loss':[], 'test_fea_loss':[], 'test_PSNR':[], 'test_TV_loss':[], 'test_L1_loss':[]}

                for img_index in tqdm(range(n_images)):
                    ray_origins = testdataset[img_index * H * W: (img_index + 1) * H * W]['rays'][:, :3]
                    ray_directions = testdataset[img_index * H * W: (img_index + 1) * H * W]['rays'][:, 3:6]
                    gt_rgb = testdataset[img_index * H * W: (img_index + 1) * H * W]['rgbs'][:, :3]
                    if distill_features == True:
                        gt_fea = testdataset[img_index * H * W: (img_index + 1) * H * W]['features']
                    
                    for i in range(int(np.ceil(H / chunk_size))):
                        ray_origins_chunk = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
                        ray_directions_chunk = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
                        gt_rgb_chunk = gt_rgb[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
                        
                        if distill_features == True:   
                            gt_fea_chunk = gt_fea[i * W * chunk_size: (i + 1) * W * chunk_size].to(device) 
                            render_rgb, render_fea, render_depth, render_mask = render_rays(nerf_model, ray_origins_chunk, ray_directions_chunk, hn=hn, hf=hf, nb_bins=nb_bins, req_others = True, blender=blender,
                                                                                            gradient_scaling = gradient_scaling)
                        else:
                            render_rgb = render_rays(nerf_model, ray_origins_chunk, ray_directions_chunk, hn=hn, hf=hf, nb_bins=nb_bins, blender=blender,
                                                     gradient_scaling = gradient_scaling)    
                        # get loss
                        img_loss = img2mse(render_rgb, gt_rgb_chunk)
                        loss = img_loss
                    
                        TV_loss = 0
                        n_grid = len(nerf_model.grids[0])
                        for grid in nerf_model.grids[0]:
                            TV_loss = TV_loss + compute_plane_tv(grid)
                        TV_loss = TV_loss / n_grid
                        L1_loss = 0 
                        for grid in nerf_model.grids[0]:
                            L1_loss = L1_loss + compute_plane_l1(grid)
                        L1_loss = L1_loss / n_grid
                        
                        # print('test: rgb vs. ft', loss, loss_distillation, loss / loss_distillation)
                        loss = img_loss + lambda_TV * TV_loss + lambda_l1 * L1_loss
                        if distill_features: 
                            distances = ((render_fea - gt_fea_chunk) ** 2).sum(dim=1)
                            loss_distillation = distances.mean() 
                            loss = loss + loss_distillation * lambda_fea                        
                            History_epoch['test_fea_loss'].append(loss_distillation.item())
                            
                        History_epoch['test_loss'].append(loss.item())
                        History_epoch['test_img_loss'].append(img_loss.item())
                        History_epoch['test_PSNR'].append(mse2psnr(img_loss).item())
                        History['train_TV_loss'].append(TV_loss.item())
                        History['train_L1_loss'].append(L1_loss.item())
                        
                History['test_loss'].append(np.mean(History_epoch['test_loss']))
                History['test_img_loss'].append(np.mean(History_epoch['test_img_loss']))
                if distill_features == True:  
                    History['test_fea_loss'].append(np.mean(History_epoch['test_fea_loss']))
                History['test_PSNR'].append(np.mean(History_epoch['test_PSNR']))
                History['test_TV_loss'].append(np.mean(History_epoch['test_TV_loss']))
                History['test_L1_loss'].append(np.mean(History_epoch['test_L1_loss']))
                
                
                # Save model if it is best in feature loss
                if distill_features == True and History['test_fea_loss'][-1] == min(History['test_fea_loss']):
                    torch.save(nerf_model.state_dict(), os.path.join(folderpath, 'bestmodel.pth'))
                    print('best feature loss:', History['test_fea_loss'][-1], 'saved')

    return History


def train_editnerf(edit_nerfmodel, model_optimizer, nb_epochs, edit4_flowers_dataset_nomask, 
                   trainingSampler, device, 
                   hn = 0, hf = 1, nb_bins = 96, lambda_TV = 1, lambda_l1 = 1e-3, lambda_depth = 1e-5,
                   mask_trick = False, original_nerf = None, embq = None, dis_thr = None, dist_less = None):
    if embq is not None:
        embq = embq.to(device)
    edit_nerfmodel.train()
    if mask_trick == True:
        original_nerf.eval()
        # gradient false
        for param in original_nerf.parameters():
            param.requires_grad = False

    iterations = nb_epochs * len(edit4_flowers_dataset_nomask) // trainingSampler.batch
    for _ in tqdm(range(iterations)):
        # get batch data
        bray_idx = trainingSampler.nextids()
        # Get data from data loader
        ray_origins = edit4_flowers_dataset_nomask[bray_idx]['rays'][:, :3].to(device)
        ray_directions = edit4_flowers_dataset_nomask[bray_idx]['rays'][:, 3:6].to(device)
        ground_truth_px_values = edit4_flowers_dataset_nomask[bray_idx]['rgbs'].to(device)
        gt_depth = edit4_flowers_dataset_nomask[bray_idx]['depths'][:, :1].to(device)

        if mask_trick == True:
            render_c, render_f, render_depth, render_mask, loss_masktrick = render_rays(edit_nerfmodel, ray_origins, ray_directions, 
                                                            hn=hn, hf=hf, nb_bins=nb_bins, req_others=True, original_nerf=original_nerf,
                                                            embq=embq, dis_thr=dis_thr, dist_less=dist_less)
        else:
            render_c, render_f, render_depth, render_mask = render_rays(edit_nerfmodel, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins, req_others=True)
        
        # Get loss
        img_loss = img2mse(render_c, ground_truth_px_values)
        depth_loss = img2mse(render_depth, gt_depth)
        
        TV_loss = 0
        n_grid = len(edit_nerfmodel.grids[0])
        for grid in edit_nerfmodel.grids[0]:
            TV_loss = TV_loss + compute_plane_tv(grid)
        TV_loss = TV_loss / n_grid
        
        L1_loss = 0 
        for grid in edit_nerfmodel.grids[0]:
            L1_loss = L1_loss + compute_plane_l1(grid)
        L1_loss = L1_loss / n_grid
        
        loss = img_loss + lambda_TV * TV_loss + lambda_l1 * L1_loss + lambda_depth * depth_loss 
        if mask_trick == True:
            loss = loss + loss_masktrick * 1e-2
        
        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()
            
    return edit_nerfmodel


# def train_editnerf(edit_nerfmodel, model_optimizer, nb_epochs, edit4_flowers_loader_nomask, device, 
#                    hn = 0, hf = 1, nb_bins = 96, lambda_TV = 1, lambda_l1 = 1e-3, lambda_depth = 1e-5,
#                    mask_trick = False, original_nerf = None, embq = None, dis_thr = None, dist_less = None):
#     if embq is not None:
#         embq = embq.to(device)
#     edit_nerfmodel.train()
#     if mask_trick == True:
#         original_nerf.eval()
#         # gradient false
#         for param in original_nerf.parameters():
#             param.requires_grad = False
#     for _ in (range(nb_epochs)):
#         print('epoch:', _, '/', nb_epochs)
#         for ep, batch in enumerate(tqdm(edit4_flowers_loader_nomask)):
#             # Get data from data loader
#             ray_origins = batch['rays'][:, :3].to(device)
#             ray_directions = batch['rays'][:, 3:6].to(device)
#             ground_truth_px_values = batch['rgbs'][:, :3].to(device)
#             gt_depth = batch['depths'][:, :1].to(device)
#             # Rendering & run model
#             if mask_trick == True:
#                 render_c, render_f, render_depth, render_mask, loss_masktrick = render_rays(edit_nerfmodel, ray_origins, ray_directions, 
#                                                                 hn=hn, hf=hf, nb_bins=nb_bins, req_others=True, original_nerf=original_nerf,
#                                                                 embq=embq, dis_thr=dis_thr, dist_less=dist_less)
#             else:
#                 render_c, render_f, render_depth, render_mask = render_rays(edit_nerfmodel, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins, req_others=True)
            
#             # Get loss
#             img_loss = img2mse(render_c, ground_truth_px_values)
#             depth_loss = img2mse(render_depth, gt_depth)
            
#             TV_loss = 0
#             n_grid = len(edit_nerfmodel.grids[0])
#             for grid in edit_nerfmodel.grids[0]:
#                 TV_loss = TV_loss + compute_plane_tv(grid)
#             TV_loss = TV_loss / n_grid
            
#             L1_loss = 0 
#             for grid in edit_nerfmodel.grids[0]:
#                 L1_loss = L1_loss + compute_plane_l1(grid)
#             L1_loss = L1_loss / n_grid
            
#             loss = img_loss + lambda_TV * TV_loss + lambda_l1 * L1_loss + lambda_depth * depth_loss 
#             if mask_trick == True:
#                 loss = loss + loss_masktrick * 1e-2
            
#             model_optimizer.zero_grad()
#             loss.backward()
#             model_optimizer.step()
            
#     return edit_nerfmodel


def train_cMLP(name, savepath_father, device, model, model_optimizer, max_steps, trainingSampler, trainingdata, dis_thr, embq, dist_less,
                hn = 0, hf = 1, nb_bins = 96, f2c_model = None, c2c_model = None):
    
    if embq is not None:
        embq

    if f2c_model is not None and c2c_model is not None:
        assert False, 'f2c_model and c2c_model cannot be both not None'
    # 0. initial model setting    
    nerf_model = copy.deepcopy(model)
    for param in nerf_model.parameters():
        param.requires_grad = False
    nerf_model.eval()
    
    if f2c_model is not None:
        f2c_model.train()
    else:
        c2c_model.train()

    History = {'train_img_loss':[], 'train_PSNR':[]}
    for _ in tqdm(range(max_steps)):
        
        bray_idx = trainingSampler.nextids()
        # Get data from data loader
        rays = trainingdata[bray_idx]['rays']
        ray_origins = rays[:, :3].to(device)
        ray_directions = rays[:, 3:6].to(device)
        gt_rgb = trainingdata[bray_idx]['rgbs'].to(device)
        
            
        if f2c_model is not None:
            render_rgb = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins,
                                    dis_thr = dis_thr, embq = embq, dist_less=dist_less, f2c_models = [f2c_model])
        else:
            render_rgb = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins,
                                    dis_thr = dis_thr, embq = embq, dist_less=dist_less, c2c_models = [c2c_model])
        
        img_loss = img2mse(render_rgb, gt_rgb)
        
        model_optimizer.zero_grad()
        img_loss.backward()
        model_optimizer.step()
        
        History['train_img_loss'].append(img_loss.item())
        History['train_PSNR'].append(mse2psnr(img_loss).item())
        
    if f2c_model is not None:
        torch.save(f2c_model.state_dict(), os.path.join(savepath_father, name + '_f2c_model.pth'))
        return History, f2c_model
    else:
        torch.save(c2c_model.state_dict(), os.path.join(savepath_father, name + '_c2c_model.pth'))   
        return History, c2c_model 
    