
from Teacher.dino import get_dino_model
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import torch
from PIL import Image
import torchvision
import argparse

PIL2tensor = torchvision.transforms.ToTensor()
device = 'cuda:0'


def main(scene, dinopath, prefixpath, savepath):
    
    # 0. set up
    dino_teacher = get_dino_model(model_name = 'dino', model_path = dinopath, device = device)
    dir = os.path.join(prefixpath, scene, 'images_8')

    images_path = []
    print(dir)
    if not os.path.exists(dir):
        raise IOError(f"{dir} is not exist.")

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            images_path.append(path)

    Dataset = []
    for img_path in tqdm(images_path):
        img = Image.open(img_path).convert('RGB')
        emb = dino_teacher.extract_features(img, upsample = True, reduce_dim = 64)
        emb = emb.cpu().detach()
        Dataset.append(emb)
    
    # convert dataset to tensor
    Dataset = torch.stack(Dataset, dim=0)
    # save dataset
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savepath = os.path.join(savepath, 'DINO_'+scene+'_64.pt')
    torch.save(Dataset, savepath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get features from DINO')
    parser.add_argument('--scene', type=str, required=True, help='scene name')
    parser.add_argument('--dinopath', type=str, default= './pre_trained_models/dino_vitbase8_pretrain.pth', help='path to DINO model')
    parser.add_argument('--prefixpath', type=str, default= './Dataset/nerf_llff_data', help='path to dataset')
    parser.add_argument('--savepath', type=str, default= './Dataset/nerf_llff_data/fea', help='path to save features')
    args = parser.parse_args()

    main(args.scene, args.dinopath, args.prefixpath, args.savepath)