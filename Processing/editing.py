from Processing.rendering import render_img, render_rays, novel_views_LLFF, creat_video
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import cv2
from Dataloader.ray_untils import get_rays, ndc_rays_blender
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import pil_to_tensor
import copy
import time
from Processing.trainer import train_editnerf, SimpleSampler, train_cMLP
from Model.triplanelite import edit_c2cMLP






img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(x))
def get_comb_img(nerf_model, dataset, device, img_indexs = None, req_others = True, embq = None, dis_thr = None, dis_less = False, hn = 0, hf = 1, nb_bins=96, blender=False, show_selection = False):
    W, H = dataset.img_wh
    if img_indexs is None:
        n_imgs = len(dataset) // W // H
        img_indexs = [i for i in range(0, n_imgs, (n_imgs - 1) // 4 + 1)]
    rgblist, emblist, depthlist, masklist, gtlist, raylist = [], [], [], [], [], []
    
    for i in range(len(img_indexs)):
        if req_others == False:
            rgb = render_img(nerf_model = nerf_model, device= device, 
                        Dataset = dataset, img_index = img_indexs[i], hn = hn, hf = hf, nb_bins = nb_bins, blender=blender)
        else:
            rgb, emb, depth, mask = render_img(nerf_model = nerf_model, device= device, 
                        Dataset = dataset, img_index = img_indexs[i], hn = hn, hf = hf, nb_bins = nb_bins, req_others= req_others, 
                        embq=embq, dis_thr=dis_thr, foreground=dis_less, blender=blender, show_selection = show_selection)
        rgblist.append(rgb)
        if req_others == True:
            depthlist.append(depth)
            masklist.append(mask)
        gtlist.append(dataset[img_indexs[i] * H * W: (img_indexs[i] + 1) * H * W]['rgbs'][:, :3].reshape(H, W, 3))
        raylist.append(dataset[img_indexs[i] * H * W: (img_indexs[i] + 1) * H * W]['rays'].reshape(H, W, 6))
    ret_gt = torch.cat((torch.cat((gtlist[0], gtlist[1]), dim=1), torch.cat((gtlist[2], gtlist[3]), dim=1)), dim=0)
    ret_rgb = torch.cat((torch.cat((rgblist[0], rgblist[1]), dim=1), torch.cat((rgblist[2], rgblist[3]), dim=1)), dim=0)
    if req_others == True:
        ret_depth = torch.cat((torch.cat((depthlist[0], depthlist[1]), dim=1), torch.cat((depthlist[2], depthlist[3]), dim=1)), dim=0)
        ret_mask = torch.cat((torch.cat((masklist[0], masklist[1]), dim=1), torch.cat((masklist[2], masklist[3]), dim=1)), dim=0)
    ret_ray = torch.cat((torch.cat((raylist[0], raylist[1]), dim=1), torch.cat((raylist[2], raylist[3]), dim=1)), dim=0)
    if req_others == True:
        return ret_gt, ret_rgb, ret_depth, ret_mask, ret_ray
    else:
        return ret_gt, ret_rgb, ret_ray

def train_f2cMLP(nerf_model, f2c_model, model_optimizer, nb_epochs, edit4_flowers_loader, device, hn = 0, hf = 1, nb_bins = 96, embq = None, settings = None, selected = False):
    nerf_model.eval()
    f2c_model.train()
    history = []
    for _ in (range(nb_epochs)):
        print('epoch:', _, '/', nb_epochs)
        for ep, batch in enumerate(tqdm(edit4_flowers_loader)):
            # Get data from data loader
            ray_origins = batch['rays'][:, :3].to(device)
            ray_directions = batch['rays'][:, 3:6].to(device)
            ground_truth_px_values = batch['rgbs'][:, :3].to(device)
            # Rendering & run model
            if selected:
                regenerated_px_values = \
                    render_rays_residual_MLP(nerf_model, f2c_model, ray_origins, ray_directions, 
                    hn=hn, hf=hf, nb_bins=nb_bins, embq=embq, dis_thr=settings['thr'] + settings['margin'], 
                                    dist_less=True)
            else:
                regenerated_px_values = \
                    render_rays_residual_MLP(nerf_model, f2c_model, ray_origins, ray_directions, 
                        hn=hn, hf=hf, nb_bins=nb_bins)
            # Get loss
            img_loss = img2mse(regenerated_px_values, ground_truth_px_values)
            loss = img_loss
            
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()
            # Record loss
            history.append(loss.item())
            # print(loss.item())
    return history, f2c_model


def get_double_comb_img(nerf_model, dataset, device, img_indexs = None, embq = None, settings = None, hn = 0, hf = 1, nb_bins=96, blender=False):
    n_images = len(dataset) // (dataset.img_wh[0] * dataset.img_wh[1])
    gt_list, rgb_list, depth_list, mask_list, ray_list = [], [], [], [], []
    if img_indexs is None:
        img_indexs = []
        for j in range(4):
            img_indexs.append([i + j * (n_images // 4) for i in range(0, n_images // 4, (n_images // 4 - 1) // 4 + 1)])
    for i in tqdm(range(4)):
        gt, rgb, depth, mask, ray = get_comb_img(nerf_model, dataset, device, img_indexs = img_indexs[i], embq = embq, settings = settings, hn = hn, hf = hf, nb_bins=nb_bins, blender=blender)
        gt_list.append(gt)
        rgb_list.append(rgb)
        depth_list.append(depth)
        mask_list.append(mask)
        ray_list.append(ray)
        
    # concatenate 4 image to one image with top left, top right, bottom left, and bottom right
    gt = torch.cat([torch.cat(gt_list[:2], dim=1), torch.cat(gt_list[2:], dim=1)], dim=0)
    rgb = torch.cat([torch.cat(rgb_list[:2], dim=1), torch.cat(rgb_list[2:], dim=1)], dim=0)
    depth = torch.cat([torch.cat(depth_list[:2], dim=1), torch.cat(depth_list[2:], dim=1)], dim=0)
    mask = torch.cat([torch.cat(mask_list[:2], dim=1), torch.cat(mask_list[2:], dim=1)], dim=0)
    ray = torch.cat([torch.cat(ray_list[:2], dim=1), torch.cat(ray_list[2:], dim=1)], dim=0)
    return gt, rgb, depth, mask, ray

def with_mask(output, mask):
    mask_output = np.array(output) * np.array(mask)
    return mask_output

def split_combineimg(img):
    # 4 image
    if img.shape.__len__() == 3:
        H, W, C = img.shape
    else:
        H, W = img.shape
        C = 1
    img1 = img[:H//2, :W//2]
    img2 = img[:H//2, W//2:]
    img3 = img[H//2:, :W//2]
    img4 = img[H//2:, W//2:]
    # cat
    all_img = np.concatenate([img1, img2, img3, img4], axis=0)
    all_img = all_img.reshape(4, H // 2, W // 2, C)
    return all_img

def double_split_combineimg(img):
    all_img = []
    tmp = split_combineimg(img)
    for i in range(4):
        all_img.append(split_combineimg(tmp[i]))
    # cat
    all_img = np.concatenate(all_img, axis=0)
    return all_img

class edit_f2cMLP(torch.nn.Module):
    def __init__(self, in_dim, n_layers = 3):
        super().__init__()

        self.color_net = []
        for i in range(n_layers):
            if i != n_layers - 1:
                self.color_net.append(torch.nn.Linear(in_dim, in_dim))
                self.color_net.append(torch.nn.LeakyReLU(0.03))
            else:
                self.color_net.append(torch.nn.Linear(in_dim, 3))
                
        self.color_net = torch.nn.Sequential(*self.color_net)

    def forward(self, x):
        return self.color_net(x)

class editing_dataset(torch.utils.data.Dataset):
    def __init__(self, rays, gt, mask):
        # print('rays, gt, mask')
        # flatten numpy mask to (N, 1) array
        mask = mask.flatten()
        rays = rays.reshape(-1, 6)
        # print(rays.shape, gt.shape, mask.shape)
        # copy numpy to fit ray shape
        self.all_rays = rays[mask, :]
        # from numpy to tensor
        self.all_rays = torch.from_numpy(self.all_rays)
        self.define_transforms()

        self.all_rgbs = []
        for i in range(gt.shape[0]):
            img = gt[i]
            img = self.transform(img)
            img = img.view(3, -1).permute(1, 0)
            self.all_rgbs += [img]
        self.all_rgbs = torch.cat(self.all_rgbs, 0)
        self.all_rgbs = self.all_rgbs[mask, :]

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        sample = {'rays': self.all_rays[idx],
                  'rgbs': self.all_rgbs[idx]}

        return sample
    
    def define_transforms(self):
        self.transform = torchvision.transforms.ToTensor()

class editing_all_dataset(torch.utils.data.Dataset):
    def __init__(self, rays, gt, depth = None):
        self.all_rays = rays.reshape(-1, 6)
        # from numpy to tensor
        self.all_rays = torch.from_numpy(self.all_rays)
        self.define_transforms()

        self.all_rgbs = []
        
        for i in range(gt.shape[0]):
            img = gt[i]
            img = self.transform(img)
            img = img.view(3, -1).permute(1, 0)
            self.all_rgbs += [img]
            
        self.all_rgbs = torch.cat(self.all_rgbs, 0)
        self.all_depths = None
        if depth is not None:    
            self.all_depths = []
            for i in range(depth.shape[0]):
                img = depth[i]
                img = self.transform(img)
                img = img.view(1, -1).permute(1, 0)
                self.all_depths += [img]
            self.all_depths = torch.cat(self.all_depths, 0)
            # print(self.all_rays.shape, self.all_rgbs.shape, self.all_depths.shape)
    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        sample = {'rays': self.all_rays[idx],
                  'rgbs': self.all_rgbs[idx],}
        if self.all_depths is not None:
            sample['depths'] = self.all_depths[idx]

        return sample

    def define_transforms(self):
        self.transform = torchvision.transforms.ToTensor()


def creat_video_f(novel_views_folderpath, savePath, prtx):
    img_array = []
    # get the number of images
    filelist=os.listdir(novel_views_folderpath)
    for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
        if not(fichier.endswith(".png")):
            filelist.remove(fichier)
    n_images = len(filelist)
    for idx in range(n_images):
        img = cv2.imread(os.path.join(novel_views_folderpath,  'f_'+str(idx)+'.png'))
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    for idx in range(n_images):
        img = cv2.imread(os.path.join(novel_views_folderpath,  'f_'+str(idx)+'.png'))
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    freames_second = 30
    
    out = cv2.VideoWriter(f'{savePath}/{prtx}video.mp4',cv2.VideoWriter_fourcc(*'avc1'), freames_second, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    cv2.destroyAllWindows()
    out.release()

def cp(x):
    if x.dtype == torch.bool:
        x = x.to(float)
    return to_pil_image(x.permute(2, 0, 1))

def md(mask):

    mask = np.array(mask)
    mask[mask >= 1] = 1
    mask = torch.from_numpy(mask).bool()
    
    return mask

def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def rgb2canny(img):
    input_image = np.asarray(img)
    preprocessor = CannyDetector()
    low_threshold = 100
    high_threshold = 200
    detected_map = preprocessor(input_image, low_threshold, high_threshold)
    detected_map = HWC3(detected_map)
    return detected_map

def mkdir_ifnoexit(path):
    if not os.path.exists(path):
        os.mkdir(path)


class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)
    
    
def normalised_est_editdepth(org_dep, mask, edit_img, depth_estimator):
    H, W = org_dep.shape[0], org_dep.shape[1]
    edited_dep_grid = depth_estimator(edit_img)['depth']
    # edited_dep = split_combineimg(np.array(edited_dep_grid))
    # get max and min value in org_dep and edited_dep
    mask = mask.flatten()
    edited_dep_grid = np.array(edited_dep_grid) / 255.0
    edited_dep_grid = edited_dep_grid.flatten()
    org_dep = org_dep.flatten()
    # print(org_dep.shape, mask.shape, edited_dep_grid.size)
    org_dep_max = org_dep[mask].max()
    org_dep_min = org_dep[mask].min()
    edited_dep_max = np.max(edited_dep_grid[mask])
    edited_dep_min = np.min(edited_dep_grid[mask])
    # print('org_dep_max:', org_dep_max, 'org_dep_min:', org_dep_min)
    # print('edited', edited_dep_max, edited_dep_min)
    # normalise edited_dep_grid
    edited_dep_grid[mask] = (edited_dep_grid[mask] - edited_dep_min / (edited_dep_max - edited_dep_min)) / (org_dep_max - org_dep_min) + org_dep_min
    edited_dep_grid[~mask] = org_dep[~mask]
    # reshape
    edited_dep_grid = edited_dep_grid.reshape(H, W)
    return edited_dep_grid

def edit_color_list(edit_list_path, save_data_path, savepath_father, device, 
                    nerf_model, LLFF_test, settings, embq, isf2c_noc2c = True, 
                    edit_one_image = None):
    embq = embq.to(device)

    c_model_dict = {}
    for file_name in os.listdir(edit_list_path):
        if edit_one_image != None:
            if file_name != edit_one_image:
                continue
        file_path = os.path.join(edit_list_path, file_name)
        
        rgb_4 = split_combineimg(np.array(Image.open(file_path)))
        ray_4 = torch.load(os.path.join(save_data_path, 'ray_4.pt'))
        mask_4_s = Image.open(os.path.join(save_data_path, 'mask_4_s.png'))
        mask_4_s = np.array(mask_4_s)
        mask_4_s[mask_4_s >= 1] = 1
        mask_4_s = torch.from_numpy(mask_4_s).bool()
        edit_masks = split_combineimg(mask_4_s.unsqueeze(-1))
        edit_rays = split_combineimg(ray_4)
        edit4_flowers_dataset = editing_dataset(edit_rays, rgb_4, edit_masks)
        trainingSampler = SimpleSampler(len(edit4_flowers_dataset), 1024)
        max_steps = 1000
        
        if isf2c_noc2c == True:
            f2c_model = edit_f2cMLP(64)
            f2c_model.to(device)
            model_optimizer = torch.optim.Adam(f2c_model.parameters(), lr = 2e-4)
        else:
            c2c_model = edit_c2cMLP(3)
            c2c_model.to(device)
            model_optimizer = torch.optim.Adam(c2c_model.parameters(), lr = 1e-3)
            
        # without rgb prefix
        name = file_name.split('.')[0][6:]
        
        print('Start editing: ', name)
        if isf2c_noc2c == True:
            history, f2c_model = train_cMLP(name, savepath_father, device, nerf_model, model_optimizer, max_steps, trainingSampler, edit4_flowers_dataset,
                                            dis_thr = settings['thr'] + settings['margin'], embq = embq, dist_less=True, f2c_model = f2c_model,
                                             hn = 0, hf = 1, nb_bins = 96,)
            
        else:
            history, c2c_model = train_cMLP(name, savepath_father, device, nerf_model, model_optimizer, max_steps, trainingSampler, edit4_flowers_dataset,
                                            dis_thr = settings['thr'] + settings['margin'], embq = embq, dist_less=True, c2c_model = c2c_model,
                                             hn = 0, hf = 1, nb_bins = 96,)
        
        
        c_model_dict[name] = c2c_model if isf2c_noc2c == False else f2c_model
        print('Finish editing: ', name)

        print('Start rendering edited NeRF novel views:')
        if isf2c_noc2c == True:
            novel_views_path = novel_views_LLFF(savepath_father, name, nerf_model, device, LLFF_test, hn = 0, hf = 1, nb_bins = 96,
                            dis_thr = settings['thr'] + settings['margin'], embq = embq, dist_less=True,req_others = False, f2c_models=[f2c_model])
        else:
            novel_views_path = novel_views_LLFF(savepath_father, name, nerf_model, device, LLFF_test, hn = 0, hf = 1, nb_bins = 96,
                            dis_thr = settings['thr'] + settings['margin'], embq = embq, dist_less=True,req_others = False, c2c_models=[c2c_model])
        
        
        creat_video(novel_views_path, savepath_father, name, req_others=False)
    return c_model_dict
        


def context_iter(nerf_model, savepath_father, LLFF_training, indices, edited_0, canny_0, rgb_4_list, mask_4_s_list, 
                 depth_4_list, ray_4_list, pipe, prompt, generator, H, W, depth_estimator,
                 device = 'cuda:0', lr = 2e-4, nb_epochs = 1, lambda_L1 = 1e-3, num_iters = 3, strength = 0.5, 
                 record = True, lambda_depth = 1e-5, mask_trick = False, original_nerf = None, woedge = False,
                 embq = None, dis_thr = None, foreground = None):
    # 0. setting up
    edit_iter = 0
    edit_nerfmodel = copy.deepcopy(nerf_model)
    for param in edit_nerfmodel.parameters():
        param.requires_grad = True
    edit_nerfmodel.to(device)
    model_optimizer = torch.optim.Adam(edit_nerfmodel.parameters(), lr=lr)
    # convert to dataset
    rgb_0 = rgb_4_list[0]
    mask_0 = mask_4_s_list[0]
    depth_0 = depth_4_list[0]
    ray_0 = ray_4_list[0]
    edited_img = split_combineimg(np.array(edited_0))
    edited_dep_grid = normalised_est_editdepth(depth_0, mask_0, edited_0, depth_estimator=depth_estimator)
    edited_dep = split_combineimg(np.array(edited_dep_grid))
    edit_rays = split_combineimg(ray_0)
    edit4_flowers_dataset_nomask = editing_all_dataset(edit_rays, edited_img, edited_dep)
    # edit4_flowers_loader_nomask = torch.utils.data.DataLoader(edit4_flowers_dataset_nomask, batch_size=512, shuffle=True, num_workers=8, pin_memory=True)

    batchsize = 512
    trainingSampler = SimpleSampler(len(edit4_flowers_dataset_nomask), batchsize)

    # 1. edit nerf in first iteration
    edit_nerfmodel = train_editnerf(edit_nerfmodel, model_optimizer, nb_epochs, edit4_flowers_dataset_nomask, trainingSampler, 
                                    device, lambda_l1 = lambda_L1, lambda_depth = lambda_depth, 
                                    mask_trick = mask_trick, original_nerf = original_nerf,
                                    embq=embq, dis_thr=dis_thr, dist_less=foreground)
    # 1.2 record the first iteration
    if record:
        time_str = time.strftime("_%Y%m%d-%H%M%S_")
        editing_histories = {
            'photo': [],
            'rendering': [],
            'model_path': [],
            'depths': [],
            'masks': [],
            'edges': [],
            'img0': [],
            'img1':[], 
        }
        # save_path = os.path.join(savepath_father, 'edit_nerfmodel' + time_str + str(edit_iter) + '.pth')
        # torch.save(edit_nerfmodel.state_dict(), save_path)
        # editing_histories['model_path'].append(save_path)
        editing_histories['photo'].append(edited_0)
        rgb_record = get_comb_img(edit_nerfmodel, LLFF_training, device, indices[edit_iter * 4 : (edit_iter + 1) * 4], req_others=False)
        editing_histories['rendering'].append(rgb_record)
        editing_histories['depths'].append(depth_0)
        editing_histories['masks'].append(mask_0)
        editing_histories['edges'].append(canny_0)
        print('-------:Done iter:', edit_iter)

    for edit_iter in range(1, num_iters):
        # 2. edit two rendering frames and fix first two cells with the edited image in the first iteration
        
        # get the two rendering frames
        idx = indices[edit_iter * 4 + 2: (edit_iter + 1) * 4]
        rgb_1 = render_img(nerf_model = edit_nerfmodel, device= device, Dataset = LLFF_training, img_index = idx[0], hn = 0, hf = 1, nb_bins = 96, req_others=False)
        rgb_2 = render_img(nerf_model = edit_nerfmodel, device= device, Dataset = LLFF_training, img_index = idx[1], hn = 0, hf = 1, nb_bins = 96, req_others=False)
        # fix the first two cells with the edited image in the first iteration
        rgb_i = pil_to_tensor(edited_0).permute(1, 2, 0) / 255.0
        mask_i = mask_4_s_list[edit_iter]
        depth_i = depth_4_list[edit_iter]
        ray_i = ray_4_list[edit_iter]
        ray_i[:H, :] = ray_0[:H, :] 
        depth_i[:H, :] = depth_0[:H,:]
        mask_i[:H, :] = 0
        rgb_i[H:, :W] = rgb_1
        rgb_i[H:, W:] = rgb_2
        # edit the two rendering frames
        canny = Image.fromarray(np.uint8(rgb2canny(cp(rgb_i))))
        if woedge == False:
            control_image = [make_inpaint_condition(cp(rgb_i), cp(mask_i)), cp(depth_i), canny]
        else:
            control_image = [make_inpaint_condition(cp(rgb_i), cp(mask_i)), cp(depth_i)]
        edited_i = pipe(
            prompt,
            num_inference_steps=20,
            generator=generator,
            eta=1.0,
            strength=strength,
            image=cp(rgb_i),
            mask_image=cp(mask_i),
            control_image=control_image,
        ).images[0]
        # convert to torch dataset
        edited_i = edited_i.resize((rgb_0.shape[1],rgb_0.shape[0]))
        edited_img = split_combineimg(np.array(edited_i))
        edit_rays = split_combineimg(ray_i)
        edited_dep_grid = normalised_est_editdepth(depth_4_list[edit_iter], mask_4_s_list[edit_iter], edited_i, depth_estimator = depth_estimator)
        edited_dep = split_combineimg(np.array(edited_dep_grid))
        edit4_flowers_dataset_nomask = editing_all_dataset(edit_rays, edited_img, edited_dep)
        # edit4_flowers_loader_nomask = torch.utils.data.DataLoader(edit4_flowers_dataset_nomask, batch_size=512, shuffle=True, num_workers=8, pin_memory=True)
        
        batchsize = 512
        trainingSampler = SimpleSampler(len(edit4_flowers_dataset_nomask), batchsize)

        # 3. edit nerf in the current iteration
        edit_nerfmodel = train_editnerf(edit_nerfmodel, model_optimizer, nb_epochs,
                                        edit4_flowers_dataset_nomask, trainingSampler, 
                                        device, lambda_l1 = lambda_L1, lambda_depth = lambda_depth,
                                        mask_trick = mask_trick, original_nerf = original_nerf,
                                        embq=embq, dis_thr=dis_thr, dist_less=foreground)
        # 3.2 record the current iteration
        if record:
            # time_str = time.strftime("_%Y%m%d-%H%M%S_")
            # save_path = os.path.join(savepath_father, 'edit_nerfmodel' + time_str + str(edit_iter) + '.pth')
            # torch.save(edit_nerfmodel.state_dict(), save_path)
            # editing_histories['model_path'].append(save_path)
            editing_histories['photo'].append(edited_i)
            rgb_record = get_comb_img(edit_nerfmodel, LLFF_training, device, indices[edit_iter * 4 : (edit_iter + 1) * 4], req_others=False)
            editing_histories['rendering'].append(rgb_record)
            editing_histories['depths'].append(depth_i)
            editing_histories['masks'].append(mask_i)
            editing_histories['edges'].append(canny)
            print('-------:Done iter:', edit_iter)

    # 4. save the final model    
    time_str = time.strftime("_%Y%m%d-%H%M%S_")
    save_path = os.path.join(savepath_father, 'edit_nerfmodel' + time_str + str(edit_iter) + '.pth')
    torch.save(edit_nerfmodel.state_dict(), save_path)
    
    if record:
        return editing_histories, edit_nerfmodel
    else:
        return None, edit_nerfmodel
    
    
def save_history_fig(savepath_father, history, prompt, method, num_iters):
    # create subfolder for history
    foldername = prompt + method
    savepath = os.path.join(savepath_father, foldername)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # save history
    for i in range(num_iters):
        idx = i
        plt.figure()
        plt.imshow(history['photo'][i])
        plt.axis('off')
        plt.savefig(os.path.join(savepath, 'photo_'+str(idx)+'.png'), bbox_inches='tight', pad_inches = 0)
        plt.close()
        #
        plt.figure()
        plt.imshow(history['rendering'][i][1])
        plt.axis('off')
        plt.savefig(os.path.join(savepath, 'nerf_'+str(idx)+'.png'), bbox_inches='tight', pad_inches = 0)
        plt.close()
        # depths
        plt.figure()
        plt.imshow(history['depths'][i])
        plt.axis('off')
        plt.savefig(os.path.join(savepath, 'depth_'+str(idx)+'.png'), bbox_inches='tight', pad_inches = 0)
        plt.close()
        # masks
        plt.figure()
        plt.imshow(history['masks'][i])
        plt.axis('off')
        plt.savefig(os.path.join(savepath, 'mask_'+str(idx)+'.png'), bbox_inches='tight', pad_inches = 0)
        plt.close()
        # edges
        plt.figure()
        plt.imshow(np.array(history['edges'][i]))
        plt.axis('off')
        plt.savefig(os.path.join(savepath, 'edge_'+str(idx)+'.png'), bbox_inches='tight', pad_inches = 0)
        plt.close()
        
        
        
def save_img(path, img):
    if img.shape[2] == 1:
        img = img[:, :, 0]
    img = Image.fromarray(np.uint8(img * 255))
    img.save(path)