# modify from N3F official code, https://github.com/dichotomies/N3F
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import cv2


def load_settings():
    return {
        "flower": {
            "rc": [200, 300],
            "sz": 50,
            "thr": 1.05,
            "margin": 0.05,
            "view_a": 0,
            "view_b": 4,
        },
        "fortress": {
            "rc": [200, 300],
            "sz": 50,
            "thr": 0.3,
            "margin": 0.05,
            "view_a": 0,
            "view_b": 4,
        },
        "leaves": {
            "rc": [250, 330],
            "sz": 50,
            "thr": 0.6,
            "margin": 0.05,
            "view_a": 0,
            "view_b": 4,
        },
        "orchids": {
            "rc": [170, 240],
            "sz": 40,
            "thr": 0.6,
            "margin": 0.05,
            "view_a": 0,
            "view_b": 4,
        },
        "room": {
            "rc": [150, 420],
            "sz": 50,
            "thr": 1.1,
            "margin": 0.05,
            "view_a": 0,
            "view_b": 4,
        },
        "trex": {
            "rc": [185, 310],
            "sz": 40,
            "thr": 0.6,
            "margin": 0.05,
            "view_a": 0,
            "view_b": 4,
        },
        "horns": {
            "rc": [100, 200],
            "sz": 50,
            "thr": 1.05,
            "margin": 0.15,
            "view_a": 0,
            "view_b": 16
        },
        "fern": {
            "rc": [125, 200],
            "sz": 50,
            "thr": 0.5 + 0.3,
            "margin": 0.15,
            "view_a": 0,
            "view_b": 16
        },
        "lego": {
            "rc": [150, 250],
            "sz": 50,
            "thr": 0.23,
            "margin": 0.01,
            "view_a": 0,
            "view_b": 16
        }
    }


def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi,ma]

def calc_pca(emb, vis=False, method = ''):
    X = emb.flatten(0, -2).cpu().numpy()
    np.random.seed(6)
    pca = PCA(n_components=3)
    X[X == np.nan] = 0
    pca.fit(X)
    X_rgb = pca.transform(X).reshape(*emb.shape[:2], 3)
    if vis:
        plt.imshow(X_rgb)
        plt.axis("off")
        plt.title("PCA of the feature embedding")
        plt.close()
    return X_rgb


def calc_feature_dist(embq, emb):
    dist = torch.norm(embq - emb, dim=-1)
    return dist.cpu()


def calc_query_emb(emb, r, c, extent, rgb=None, dir = None, method = '', vis = False):
    if rgb is not None:

        rgb_cut = rgb.clone().cpu()
        rgb_cut[r : r + extent, c : c + extent] = 0
        rgb_patch = rgb[r : r + extent, c : c + extent].cpu()

        if vis:

            f, ax = plt.subplots(1, 2, figsize=(5, 2))
            ax[0].imshow(rgb_cut)
            ax[0].axis("off")
            ax[1].imshow(rgb_patch)
            ax[1].axis("off")
            plt.suptitle("Rendered image without patch vs. patch")
            plt.show()
            plt.close()
    
    dir_q = None
    if dir is not None:
        dir_q = dir[r : r + extent, c : c + extent, :].reshape(-1,9).mean(dim = 0)


    emb_patch = emb[r : r + extent, c : c + extent]
    embq = torch.nn.functional.normalize(emb_patch.flatten(0, -2).mean(dim=0), dim=0)

    return embq, dir_q

