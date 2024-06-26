#  modified from N3F https://github.com/dichotomies/N3F/tree/master/feature_extractor/lib   
import torch
import torchvision
from PIL import Image
from .vision_transformer import VisionTransformer
from sklearn.decomposition import PCA

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def pca(X, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return pca.transform(X)

class DINO:
    def __init__(self, patch_size, device="cuda:0"):
        self.device = device
        self.model = VisionTransformer(patch_size=patch_size, qkv_bias=True).to(
            self.device
        )
        self.model.eval()

    def load_checkpoint(self, ckpt_file, checkpoint_key="model"):
        state_dict = torch.load(ckpt_file, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = self.model.load_state_dict(state_dict, strict=False)
        print("Pretrained weights loaded with msg: {}".format(msg))

    def extract_features(
        self, img: Image or torch.Tensor, 
        transform=True, upsample=None, 
        reduce_dim = None, normalised = True
    ):
        if transform:
            img = self.transform(img, 256).unsqueeze(0)  # Nx3xHxW
        with torch.no_grad():
            out = self.model.get_intermediate_layers(img.to(self.device), n=1)[0]
            out = out[:, 1:, :]  # we discard the [CLS] token
            h, w = int(img.shape[2] / self.model.patch_embed.patch_size), int(
                img.shape[3] / self.model.patch_embed.patch_size
            )

            dim = out.shape[-1]
            
            if reduce_dim is not None:
                out = out.reshape(-1, dim).cpu().numpy()
                pca = PCA(n_components=reduce_dim)
                out = pca.fit_transform(out)
                out = torch.tensor(out).to(self.device)
                dim = reduce_dim
            
            out = out.reshape(-1, h, w, dim).permute(0, 3, 1, 2)
                
            if normalised == True:
                for i in range(out.shape[0]):
                    out[i] = torch.nn.functional.normalize(out[i], dim = 0)

            if upsample is not None:
                out = torch.nn.functional.interpolate(out, size = (img.shape[2], img.shape[3]), mode="nearest").squeeze(0)
        return out

    @staticmethod
    def transform(img, image_size):
        transforms = torchvision.transforms.Compose([
            # torchvision.transforms.Resize(image_size), 
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize(MEAN, STD)
        ])
        img = transforms(img)
        return img
    
def get_dino_model(model_name, model_path, device):
    if model_name == "dino":
        model = DINO(patch_size=8, device=device)
        # downloaded from links provided in https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        model.load_checkpoint(model_path)
    elif model_name == "dino16":
        model = DINO(patch_size=16, device=device)
        model.load_checkpoint(model_path)
    else:
        raise NotImplementedError
    return model
