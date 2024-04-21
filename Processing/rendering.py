import torch
import numpy as np
from tqdm import tqdm
from Dataloader.LLFF import get_rays, ndc_rays_blender
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.decomposition import PCA


class GradientScaler(torch.autograd.Function):
# Modified from "Floaters no more" https://gradient-scaling.github.io/
  @staticmethod
  def forward(ctx, colors, sigmas, ray_dist):
    ctx.save_for_backward(ray_dist)
    return colors, sigmas, ray_dist
  @staticmethod
  def backward(ctx, grad_output_colors, grad_output_sigmas, grad_output_ray_dist):
    (ray_dist,) = ctx.saved_tensors
    scaling = torch.square(ray_dist).clamp(0, 1)

    return grad_output_colors * scaling, grad_output_sigmas * scaling, grad_output_ray_dist
    
def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)

def sampling_on_rays(ray_origins, ray_directions, hn=0, hf=1.1, nb_bins=64):
    device = ray_origins.device
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)  # [batch_size, nb_bins, 3]
    return x, delta, t    


def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=1, nb_bins=64, req_others = False, blender=False,
                         dis_thr = None, embq = None, dist_less=None, f2c_models = None, c2c_models = None, show_selection = False, 
                         gradient_scaling = False, original_nerf = None):
    # 1. sample rays
    x, delta, z_vals = sampling_on_rays(ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
    
    # 2. ouput 3D raw colors, densities and features
    outputs = nerf_model(x)
    colors, sigma = outputs['rgb'], outputs['density']
    
    if gradient_scaling:
        ray_dist = x - ray_origins.unsqueeze(1)
        ray_dist = torch.norm(ray_dist, dim=-1)
        colors = colors.reshape(ray_dist.shape[0], ray_dist.shape[1], -1)
        sigma = sigma.reshape(ray_dist.shape[0], ray_dist.shape[1], -1)
        ray_dist = ray_dist.unsqueeze(-1)
        
        # print('colots shape: ', colors.shape, 'sigma shape: ', sigma.shape, 'ray_dist shape: ', ray_dist.shape)
        colors, sigma, ray_dist = GradientScaler.apply(colors, sigma, ray_dist)
        
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    
    # 3. output pixel color if don't require others
    if req_others == False:
        if f2c_models is not None or c2c_models is not None:
            if f2c_models is not None and c2c_models is not None:
                assert False, "f2c_models and c2c_models can't be both not None"
            assert dis_thr is not None, "dis_thr is None"
            assert embq is not None, "embq is None"
            assert dist_less is not None, "dist_less is None"
            
            features = outputs['features']
            C = outputs['features'].shape[-1]
            features = features.reshape(x.shape[:-1] + (-1,))
            # To do: optimise (speed up) the following one by only compute in GPU
            with torch.no_grad():
                dist = (torch.nn.functional.normalize(features.view(-1, C)) - embq).norm(dim=1)
            if dist_less == True:
                mask = (dist < dis_thr).view(features.shape[:2])
            else:
                mask = (dist >= dis_thr).view(features.shape[:2])
            # if mask is zero
            if torch.sum(mask) != 0:
                if f2c_models is not None:
                    for f2c_model in f2c_models:
                        res_color = f2c_model(features[mask].reshape(-1, C)).reshape(-1, 3)
                        colors[mask] = colors[mask] + res_color.reshape(colors[mask].shape)
                else:
                    res_color = colors[mask].reshape(-1, 3)
                    for c2c_model in c2c_models:
                        res_color = c2c_model(res_color).reshape(-1, 3)
                    colors[mask] = res_color.reshape(colors[mask].shape)
        
        if blender:
            alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
            weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
            weight_sum = weights.sum(-1).sum(-1)
            c = (weights * colors).sum(dim=1) + 1 - weight_sum.unsqueeze(-1)
        else:
            alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
            weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
            c = (weights * colors).sum(dim=1)  # Pixel values
        return c
    # 4. get features
    features = outputs['features']
    C = outputs['features'].shape[-1]
    features = features.reshape(x.shape[:-1] + (-1,))
    
    # 5. get mask based on features
    if dis_thr is not None:
        if f2c_models is not None and c2c_models is not None:
            assert False, "f2c_models and c2c_models can't be both not None"
        assert embq is not None, "embq is None"
        assert dist_less is not None, "dist_less is None"
        with torch.no_grad():
            dist = (torch.nn.functional.normalize(features.view(-1, C)) - embq).norm(dim=1)
        if dist_less == True:
            mask = (dist < dis_thr).view(features.shape[:2])
        else:
            mask = (dist >= dis_thr).view(features.shape[:2])
            
        # 6. if add phi_{edit}
        if torch.sum(mask) != 0:
            if f2c_models is not None:
                for f2c_model in f2c_models:
                    res_color = f2c_model(features[mask].reshape(-1, C)).reshape(-1, 3)
                    colors[mask] = colors[mask] + res_color.reshape(colors[mask].shape)
            elif c2c_models is not None:
                res_color = colors[mask].reshape(-1, 3)
                for c2c_model in c2c_models:
                    res_color = c2c_model(res_color).reshape(-1, 3)
                colors[mask] = res_color.reshape(colors[mask].shape)
    
    # 6. if only show selection 
    if show_selection:
        sigma[mask] = 0
    
    # 7. output the final results
    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)

    f = (weights * features).sum(dim=1)
    f = torch.nn.functional.normalize(
        f,
        dim=1   
    )
    acc_map = torch.sum(weights.squeeze(-1), -1)
    depth_map = torch.sum(weights.squeeze(-1) * z_vals, -1)
    depth_map = depth_map + (1. - acc_map) * ray_directions[..., -1]
    aa, bb = nerf_model.aabb
    depth_map[depth_map < aa] = aa
    depth_map[depth_map > bb] = bb
    depth_map = (depth_map - aa) / (bb - aa)
    depth_map = 1.0 - depth_map
    mask = acc_map > 0.7

    if blender:
        weight_sum = weights.sum(-1).sum(-1)
        c = (weights * colors).sum(dim=1) + 1 - weight_sum.unsqueeze(-1)
    else:
        c = (weights * colors).sum(dim=1)  # Pixel values

    if original_nerf is not None:
        outputs_original = original_nerf(x)
        features_original, sigma_original,  = outputs_original['features'], outputs_original['density']
        features_original = features_original.reshape(x.shape[:-1] + (-1,))
        sigma_original = sigma_original.reshape(x.shape[:-1])
        with torch.no_grad():
            dist_original = (torch.nn.functional.normalize(features_original.view(-1, C)) - embq).norm(dim=1)

        if dist_less == True:
            mask_original = (dist_original < dis_thr).view(features_original.shape[:2])
        else:
            mask_original = (dist_original >= dis_thr).view(features_original.shape[:2])
        alpha_original = 1 - torch.exp(-sigma_original * delta)  # [batch_size, nb_bins]
        loss_masktrick = torch.sum((alpha_original[~mask_original] - alpha[~mask_original]) ** 2)
        return c, f, depth_map, mask, loss_masktrick
    
    return c, f, depth_map, mask
    
    

def render_img(nerf_model, device, Dataset, img_index, hn = 0, hf = 1.1, nb_bins = 64,\
    req_others = False, dis_thr = None, embq = None, foreground=None, blender=False, f2c_models = None, show_selection = False):
    
    if embq is not None:
        embq = embq.to(device)

    nerf_model.eval()

    
    with torch.no_grad():
        H=Dataset.img_wh[1]
        W=Dataset.img_wh[0]
        chunk_size = 5
        ray_origins = Dataset[img_index * H * W: (img_index + 1) * H * W]['rays'][:, :3]
        ray_directions = Dataset[img_index * H * W: (img_index + 1) * H * W]['rays'][:, 3:6]
        gt_rgb = Dataset[img_index * H * W: (img_index + 1) * H * W]['rgbs'][:, :3]

        if req_others == True:
            render_rgb, render_emb, render_depth, render_mask = torch.zeros_like(gt_rgb), torch.zeros(H*W, 64), torch.zeros_like(gt_rgb[..., 0]), torch.zeros_like(gt_rgb[..., 0]).bool()
        else:
            render_rgb = torch.zeros_like(gt_rgb)    
        for i in range(int(np.ceil(H / chunk_size))):
            ray_origins_chunk = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
            ray_directions_chunk = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
            
            if req_others == True:
                rgb_chunk, emb_chunk, depth_map_chunk, mask_chunk = render_rays(nerf_model, ray_origins_chunk, ray_directions_chunk, \
                        hn=hn, hf=hf, nb_bins=nb_bins, req_others = req_others, dis_thr = dis_thr, embq = embq, dist_less=foreground, blender=blender, show_selection=show_selection)
            else:
                rgb_chunk = render_rays(nerf_model, ray_origins_chunk, ray_directions_chunk, \
                        hn=hn, hf=hf, nb_bins=nb_bins, req_others = req_others, dis_thr = dis_thr, embq = embq, dist_less=foreground, blender=blender)
            
            render_rgb[i * W * chunk_size: (i + 1) * W * chunk_size] = rgb_chunk.cpu()
            if req_others == True:
                render_emb[i * W * chunk_size: (i + 1) * W * chunk_size] = emb_chunk.cpu()
                render_depth[i * W * chunk_size: (i + 1) * W * chunk_size] = depth_map_chunk.cpu()
                render_mask[i * W * chunk_size: (i + 1) * W * chunk_size] = mask_chunk.cpu()

        render_rgb = render_rgb.reshape(H, W, 3)
        if req_others == True:
            render_emb = render_emb.reshape(H, W, -1)
            render_depth = render_depth.reshape(H, W, -1)
            render_mask = render_mask.reshape(H, W, -1)
    if req_others == True:
        return render_rgb, render_emb, render_depth, render_mask
    else:
        return render_rgb
    
    
def get_all_samples(Dataset, device, hn = 0, hf = 1.1, nb_bins = 64, chunk_size = 5):
    H=Dataset.img_wh[1]
    W=Dataset.img_wh[0]
    chunk_size=5
    
    n_imgs = len(Dataset) // (H * W)
    all_samples = []
    for img_index in tqdm(range(n_imgs)):
        ray_origins = Dataset[img_index * H * W: (img_index + 1) * H * W]['rays'][:, :3]
        ray_directions = Dataset[img_index * H * W: (img_index + 1) * H * W]['rays'][:, 3:6]
        gt_rgb = Dataset[img_index * H * W: (img_index + 1) * H * W]['rgbs'][:, :3]
        gt_fea = Dataset[img_index * H * W: (img_index + 1) * H * W]['features']
        
        for i in range(int(np.ceil(H / chunk_size))):
            ray_origins_chunk = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
            ray_directions_chunk = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)

            sampling_points = sampling_on_rays(ray_origins_chunk, ray_directions_chunk, hn=hn, hf=hf, nb_bins=nb_bins)
            all_samples.append(sampling_points.cpu())

    all_samples = torch.cat(all_samples, dim=0)
    return all_samples
    
def plt_save_img(img, save_path, rgb = True):
    plt.figure()
    if rgb == True:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches = 0)
    plt.close()

def novel_views_LLFF(folderpath, name, edit_nerfmodel, device, dataset, hn = 0, hf = 1, nb_bins = 96,
                     req_others = False, blender=False, dis_thr = None, embq = None, 
                     dist_less=None, f2c_models = None, show_selection = False, c2c_models = None):
    if embq is not None:
        embq = embq.to(device)

    W, H = dataset.img_wh
    c2ws = dataset.render_path
    chunk_size = 40
    novel_views_folderpath = os.path.join(folderpath, name + '_novel_views')
    os.makedirs(novel_views_folderpath)
    
    render_fea_frames = torch.zeros(60, H, W, 64)
    with torch.no_grad():
        for idx, c2w in tqdm(enumerate(c2ws)):
            if idx >= 60:
                break
            c2w = torch.FloatTensor(c2w)
            rays_o, rays_d = get_rays(dataset.directions, c2w)  # both (h*w, 3)
            rays_o, rays_d = ndc_rays_blender(H, W, dataset.focal[0], 1.0, rays_o, rays_d)
            rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
            
            render_rgb = torch.zeros((H * W, 3))
            render_fea = torch.zeros((H * W, 64))
            render_depth = torch.zeros((H * W))
            render_mask = torch.zeros((H * W)).bool()
            
            for i in range(int(np.ceil(H / chunk_size))):
                ray_origins_chunk = rays_o[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
                ray_directions_chunk = rays_d[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
                
                if req_others == False:
                    rgb_chunk = render_rays(edit_nerfmodel, ray_origins_chunk, ray_directions_chunk, hn=hn, hf=hf, nb_bins=nb_bins,
                                        req_others = req_others, blender=blender, dis_thr = dis_thr, embq = embq, 
                                        dist_less=dist_less, f2c_models = f2c_models, show_selection = show_selection, c2c_models = c2c_models)
                else:
                    rgb_chunk, fea_chunk, depth_chunk, mask_chunk  = render_rays(edit_nerfmodel, ray_origins_chunk, ray_directions_chunk, hn=hn, hf=hf, nb_bins=nb_bins,
                                        req_others = req_others, blender=blender, dis_thr = dis_thr, embq = embq, 
                                        dist_less=dist_less, f2c_models = f2c_models, show_selection = show_selection, c2c_models = c2c_models)
                render_rgb[i * W * chunk_size: (i + 1) * W * chunk_size] = rgb_chunk.detach().cpu()

                if req_others == True:
                    render_fea[i * W * chunk_size: (i + 1) * W * chunk_size] = fea_chunk.detach().cpu()
                    render_depth[i * W * chunk_size: (i + 1) * W * chunk_size] = depth_chunk.detach().cpu()
                    render_mask[i * W * chunk_size: (i + 1) * W * chunk_size] = mask_chunk.detach().cpu()
            
                
            render_rgb = render_rgb.reshape(H, W, 3)   
            plt_save_img(render_rgb.cpu().numpy(), os.path.join(novel_views_folderpath, 'img_'+str(idx)+'.png'))     
            
            if req_others == True:
                render_fea_frames[idx] = render_fea.reshape(H, W, 64)
                render_depth = render_depth.reshape(H, W, 1)
                render_mask = render_mask.reshape(H, W, 1)
                plt_save_img(render_depth.cpu().numpy(), os.path.join(novel_views_folderpath, 'depth_'+str(idx)+'.png'), rgb = False)
                plt_save_img(render_mask.cpu().numpy(), os.path.join(novel_views_folderpath, 'mask_'+str(idx)+'.png'), rgb = False)
    
    if req_others == True:                
        # reduce by PCA for features
        render_fea_frames = render_fea_frames.flatten(0, -2).cpu().numpy()
        np.random.seed(6)
        pca = PCA(n_components=3)
        render_fea_frames[render_fea_frames == np.nan] = 0
        pca.fit(render_fea_frames)
        X_rgb = pca.transform(render_fea_frames).reshape(60, H, W, 3)
        for idx in range(60):
            plt_save_img(X_rgb[idx], os.path.join(novel_views_folderpath, 'fea_'+str(idx)+'.png'))
            
    return novel_views_folderpath

def creat_video(novel_views_folderpath, savePath, prtx, req_others = False):
    img_array = []
    # get the number of images
    filelist=os.listdir(novel_views_folderpath)
    for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
        if not(fichier.endswith(".png")):
            filelist.remove(fichier)
    # n_images = len(filelist)
    n_images = 60
    for idx in range(n_images):
        img = cv2.imread(os.path.join(novel_views_folderpath,  'img_'+str(idx)+'.png'))
        img_array.append(img)
    for idx in range(n_images):
        img = cv2.imread(os.path.join(novel_views_folderpath,  'img_'+str(idx)+'.png'))
        img_array.append(img)
    
    height, width, _ = img_array[0].shape
    size = (width,height)
    freames_second = 30
    out = cv2.VideoWriter(f'{savePath}/{prtx}rgb_video.mp4',cv2.VideoWriter_fourcc(*'mp4v'), freames_second, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    cv2.destroyAllWindows()
    out.release()
    
    if req_others == True:
        # features
        img_array = []
        for idx in range(n_images):
            img = cv2.imread(os.path.join(novel_views_folderpath,  'fea_'+str(idx)+'.png'))
            img_array.append(img)
        for idx in range(n_images):
            img = cv2.imread(os.path.join(novel_views_folderpath,  'fea_'+str(idx)+'.png'))
            img_array.append(img)
        out = cv2.VideoWriter(f'{savePath}/{prtx}fea_video.mp4',cv2.VideoWriter_fourcc(*'mp4v'), freames_second, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        cv2.destroyAllWindows()
        out.release()
        # depth
        img_array = []
        for idx in range(n_images):
            img = cv2.imread(os.path.join(novel_views_folderpath,  'depth_'+str(idx)+'.png'))
            img_array.append(img)
        for idx in range(n_images):
            img = cv2.imread(os.path.join(novel_views_folderpath,  'depth_'+str(idx)+'.png'))
            img_array.append(img)
        out = cv2.VideoWriter(f'{savePath}/{prtx}depth_video.mp4',cv2.VideoWriter_fourcc(*'mp4v'), freames_second, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        cv2.destroyAllWindows()
        out.release()
        # mask
        img_array = []
        for idx in range(n_images):
            img = cv2.imread(os.path.join(novel_views_folderpath,  'mask_'+str(idx)+'.png'))
            img_array.append(img)
        for idx in range(n_images):
            img = cv2.imread(os.path.join(novel_views_folderpath,  'mask_'+str(idx)+'.png'))
            img_array.append(img)
        out = cv2.VideoWriter(f'{savePath}/{prtx}mask_video.mp4',cv2.VideoWriter_fourcc(*'mp4v'), freames_second, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        cv2.destroyAllWindows()
        out.release()