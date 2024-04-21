import torch
from .density_activation import *
from .utils import *

class f2cMLP(torch.nn.Module):
    def __init__(self, in_dim, n_layers = 3):
        super().__init__()

        self.color_net = []
        for i in range(n_layers):
            if i != n_layers - 1:
                self.color_net.append(torch.nn.Linear(in_dim, in_dim))
                self.color_net.append(torch.nn.ReLU())
            else:
                self.color_net.append(torch.nn.Linear(in_dim, 3))
                self.color_net.append(torch.nn.Sigmoid())
        self.color_net = torch.nn.Sequential(*self.color_net)

    def forward(self, x):
        return self.color_net(x)


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
    
    
class edit_c2cMLP(torch.nn.Module):
    def __init__(self, in_dim, n_layers = 3):
        super().__init__()

        self.color_net = []
        for i in range(n_layers):
            if i != n_layers - 1:
                self.color_net.append(torch.nn.Linear(in_dim, 16))
                self.color_net.append(torch.nn.LeakyReLU(0.03))
                in_dim = 16
            else:
                self.color_net.append(torch.nn.Linear(in_dim, 3))
                
        self.color_net = torch.nn.Sequential(*self.color_net)

    def forward(self, x):
        return self.color_net(x)
    
class edit_f2cMLP_tinycuda(torch.nn.Module):
    def __init__(self, in_dim, n_layers = 3):
        
        super().__init__()
        import tinycudann
        self.color_net = tinycudann.Network(
            n_input_dims=in_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "LeakyReLU",
                "output_activation": "None",
                "n_neurons": in_dim,
                "n_hidden_layers": n_layers - 1,
            },
        )

    def forward(self, x):
        return self.color_net(x)


class triplane_fea(torch.nn.Module):
    def __init__(self, aabb, plane_resolution = 256, plane_channels = 128, distill_dim = 64, 
                 density_activation = 'trunc_exp', tinycuda = False,):
        super().__init__()
        if tinycuda:
            import tinycudann
            
        self.aabb = torch.nn.Parameter(aabb, requires_grad=False)
        self.plane_resolution = plane_resolution
        self.plane_channels = plane_channels
        self.density_activation = init_density_activation(density_activation)
        self.distill_dim = distill_dim
        
        
        # 1. Init planes
        self.grids = torch.nn.ModuleList()
        gp = torch.nn.Parameter(torch.empty([1, plane_channels, self.plane_resolution, self.plane_resolution]))
        torch.nn.init.uniform_(gp, a=0.1, b=0.5)
        self.feature_dim = self.plane_channels
        grid_coefs = torch.nn.ParameterList([gp, gp, gp])
        self.grids.append(grid_coefs)

        # 2. Init density network
        self.geo_feat_dim = self.distill_dim + 32
        layer1 = torch.nn.Linear(self.feature_dim, self.geo_feat_dim + 1)
        layer2 = torch.nn.Linear(self.geo_feat_dim + 1, self.geo_feat_dim + 1)
        self.sigma_net = torch.nn.Sequential(layer1, torch.nn.ReLU(), layer2) 

        # 3. Feature network
        self.in_dim_emb = self.geo_feat_dim
        layer3 = torch.nn.Linear(self.in_dim_emb, self.distill_dim)
        self.fea_net = torch.nn.Sequential(torch.nn.ReLU(), layer3) 
        
        # 4. Color network
        self.in_dim_color = self.distill_dim
        if self.in_dim_color == 64 and tinycuda:
            self.color_net = tinycudann.Network(
                n_input_dims=self.in_dim_color,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )
        else:
            self.color_net = f2cMLP(self.in_dim_color)

    def get_density(self, pts: torch.Tensor):
        n_rays, n_samples = pts.shape[:2]
        pts = pts.reshape(-1, pts.shape[-1])
        features = interpolate_ms_features(
            pts, ms_grids=self.grids,  # noqa
            grid_dimensions= 2,
            concat_features=False, num_levels=None)
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)
        features = self.sigma_net(features)
        
        features, density_before_activation = torch.split(
            features, [self.geo_feat_dim, 1], dim=-1)
        
        density = self.density_activation(
            density_before_activation.to(pts)
        ).view(n_rays, n_samples, 1)
        
        return density, features

    def forward(self, pts: torch.Tensor):
        pts = normalize_aabb(pts, self.aabb)
        # density
        density, geo_features = self.get_density(pts.view(1, -1, 3))
        # features
        semantic_features = self.fea_net(geo_features.view(-1, self.geo_feat_dim))
        # colors
        rgb = self.color_net(semantic_features.view(-1, self.distill_dim)).float()
        return {"rgb": rgb, "density": density, "features": semantic_features, "geo_features": geo_features}
