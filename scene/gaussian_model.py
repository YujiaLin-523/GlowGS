#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, contract_to_unisphere, mortonEncode
from utils.quantization_utils import half_ste, quantize, dequantize
from utils.gpcc_utils import encode_xyz, decode_xyz
from encoders.geometry_appearance_encoder import GeometryAppearanceEncoder
import tinycudann as tcnn

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_base_activation = torch.exp
        self.scaling_base_inverse_activation = torch.log

        self.scaling_activation = torch.sigmoid
        self.scaling_inverse_activation = inverse_sigmoid

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.xyz_activation = half_ste

        self.mask_activation = torch.sigmoid
        self.sh_mask_activation = torch.sigmoid

    def setup_configs(self, hash_size, width, depth):
        self.encoding_config = {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": hash_size,
            "base_resolution": 16,
            "per_level_scale": 1.447269237440378,
        }
        self.network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": width,
            "n_hidden_layers": depth,
        }
    
    def setup_params(self):
        # Create base hash encoder
        hash_encoder = tcnn.Encoding(n_input_dims=3, encoding_config=self.encoding_config)
        
        # GeometryAppearanceEncoder with feature role split support
        # Uses stored GeoEncoder parameters for flexible size/quality tradeoff
        self._grid = GeometryAppearanceEncoder(
            hash_encoder=hash_encoder,
            out_dim=hash_encoder.n_output_dims,
            geo_channels=self._geo_channels,
            geo_resolution=self._geo_resolution,
            geo_rank=self._geo_rank,
            feature_role_split=self._feature_role_split
        )
        self._grid = self._grid.cuda()
        self._grid.geo_encoder = self._grid.geo_encoder.cuda()
        self._grid.appearance_projection = self._grid.appearance_projection.cuda()
        self._grid.geometry_projection = self._grid.geometry_projection.cuda()
        self._grid.fusion_layer = self._grid.fusion_layer.cuda()
        
        # MLP heads for Gaussian attributes
        self._features_rest_head = tcnn.Network(n_input_dims=self._grid.n_output_dims, n_output_dims=(self.max_sh_degree + 1) ** 2 * 3 - 3, network_config=self.network_config)
        self._scaling_head = tcnn.Network(n_input_dims=self._grid.n_output_dims, n_output_dims=3, network_config=self.network_config)
        self._rotation_head = tcnn.Network(n_input_dims=self._grid.n_output_dims, n_output_dims=4, network_config=self.network_config)
        self._opacity_head = tcnn.Network(n_input_dims=self._grid.n_output_dims, n_output_dims=1, network_config=self.network_config)
        
        self._aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=torch.float, device='cuda')
        self._rotation_init = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float, device='cuda')
        self._opacity_init = torch.tensor([[np.log(0.1 / (1 - 0.1))]], dtype=torch.float, device='cuda')

    def __init__(self, sh_degree: int, hash_size=19, width=64, depth=2, feature_role_split=True,
                 geo_resolution=48, geo_rank=6, geo_channels=8):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._feature_role_split = feature_role_split  # Geometry/appearance disentanglement flag
        # Store GeoEncoder parameters for serialization/deserialization
        self._geo_resolution = geo_resolution
        self._geo_rank = geo_rank
        self._geo_channels = geo_channels
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling_base = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._mask = torch.empty(0)
        self._sh_mask = torch.empty(0)
        self._grid = torch.empty(0)
        self._features_rest_head = torch.empty(0)
        self._scaling_head = torch.empty(0)
        self._rotation_head = torch.empty(0)
        self._opacity_head = torch.empty(0)
        self._aabb = torch.empty(0)
        self._rotation_init = torch.empty(0)
        self._opacity_init = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.optimizer_i = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.setup_configs(hash_size, width, depth)
        self.setup_params()


    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling_base,
            self._mask,
            self._sh_mask,
            self._grid,
            self._features_rest_head,
            self._scaling_head,
            self._rotation_head,
            self._opacity_head,
            self._aabb,
            self._rotation_init,
            self._opacity_init,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.optimizer_i.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc,
        self._features_rest,
        self._scaling_base,
        self._mask,
        self._sh_mask,
        self._grid,
        self._features_rest_head,
        self._scaling_head,
        self._rotation_head,
        self._opacity_head,
        self._aabb,
        self._rotation_init,
        self._opacity_init,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        opt_i_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.optimizer_i.load_state_dict(opt_i_dict)

    def update_attributes(self):
        """
        Update implicit Gaussian attributes from encoder output.
        
        Feature role split routing:
            - geometry_latent → scale, rotation, opacity heads
            - appearance_latent → SH/color head (features_rest)
        """
        contracted_xyz = self.get_contracted_xyz
        encoder_output = self._grid(contracted_xyz)
        
        if self._feature_role_split and isinstance(encoder_output, tuple):
            # Geometry/appearance disentanglement enabled
            geometry_latent, appearance_latent = encoder_output
            
            # Stability check
            geometry_latent = self._stabilize_features(geometry_latent)
            appearance_latent = self._stabilize_features(appearance_latent)
            
            # Store for debug logging
            self._last_geometry_latent = geometry_latent
            self._last_appearance_latent = appearance_latent
            
            # Geometric attributes from geometry branch (structure-focused)
            self._opacity = self._opacity_head(geometry_latent).float() + self._opacity_init
            self._scaling = self._scaling_head(geometry_latent).float()
            self._rotation = self._rotation_head(geometry_latent).float() + self._rotation_init
            
            # Appearance attributes from appearance branch (texture-focused)
            if self.max_sh_degree > 0:
                self._features_rest = self._features_rest_head(appearance_latent).view(-1, (self.max_sh_degree + 1) ** 2 - 1, 3).float()
        else:
            # Fallback: single fused latent for all heads
            feats = encoder_output if not isinstance(encoder_output, tuple) else encoder_output[0]
            feats = self._stabilize_features(feats)
            
            self._opacity = self._opacity_head(feats).float() + self._opacity_init
            self._scaling = self._scaling_head(feats).float()
            self._rotation = self._rotation_head(feats).float() + self._rotation_init
            if self.max_sh_degree > 0:
                self._features_rest = self._features_rest_head(feats).view(-1, (self.max_sh_degree + 1) ** 2 - 1, 3).float()
        
        # NOTE: Removed torch.cuda.empty_cache() here for real-time performance
        # empty_cache() is expensive (~1-2ms) and should only be called when necessary
    
    def _stabilize_features(self, feats: torch.Tensor) -> torch.Tensor:
        """Numerical stability for feature tensors."""
        if torch.isnan(feats).any() or torch.isinf(feats).any():
            feats = torch.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0)
        return torch.clamp(feats, -10.0, 10.0)

    @property
    def get_scaling_base(self):
        # Clamp before exp to prevent overflow (exp(10) ≈ 22026, exp(20) ≈ 4.8e8)
        clamped_scaling = torch.clamp(self._scaling_base, -15.0, 10.0)
        return self.scaling_base_activation(clamped_scaling)

    @property
    def get_scaling(self):
        return self.get_scaling_base * self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        # Add epsilon to prevent NaN from normalize when rotation is all zeros
        # This can happen during initialization or if gradients zero out the rotation
        rotation = self._rotation
        eps = 1e-6
        norm = torch.norm(rotation, dim=-1, keepdim=True)
        # If norm is too small, use a default rotation (1, 0, 0, 0)
        rotation = torch.where(
            norm > eps,
            rotation,
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=rotation.device, dtype=rotation.dtype).expand_as(rotation)
        )
        return self.rotation_activation(rotation)
    
    @property
    def get_xyz(self):
        return self.xyz_activation(self._xyz)

    @property
    def get_contracted_xyz(self):
        contracted = contract_to_unisphere(self.get_xyz.detach(), self._aabb)
        # Extra safety: clamp to [-1, 1] and check for NaN/Inf
        # This ensures encoder always receives valid input
        contracted = torch.clamp(contracted, -1.0, 1.0)
        if torch.isnan(contracted).any() or torch.isinf(contracted).any():
            contracted = torch.nan_to_num(contracted, nan=0.0, posinf=1.0, neginf=-1.0)
        return contracted
        
    @property
    def get_features(self):
        features_dc = self._features_dc
        if self.max_sh_degree > 0:
            features_rest = self._features_rest
            return torch.cat((features_dc, features_rest), dim=1)
        else:
            return features_dc
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @property
    def get_mask(self):
        return self.mask_activation(self._mask)

    @property
    def get_sh_mask(self):
        return self.sh_mask_activation(self._sh_mask)
    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None]

        # explicit attributes
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling_base = nn.Parameter(scales.requires_grad_(True))
        self._mask = nn.Parameter(torch.ones((fused_point_cloud.shape[0], 1), device="cuda").requires_grad_(True))
        self._sh_mask = nn.Parameter(torch.ones((fused_point_cloud.shape[0], self.max_sh_degree), device="cuda").requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        param_groups = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_dc_lr, "name": "f_dc"},
            {'params': [self._scaling_base], 'lr': training_args.scaling_base_lr, "name": "scaling_base"},
            {'params': [self._mask], 'lr': training_args.mask_lr, "name": "mask"},
            {'params': [self._sh_mask], 'lr': training_args.sh_mask_lr, "name": "sh_mask"}
        ]
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

        param_groups_i = [
            {'params': self._grid.parameters(), 'lr': 0.0, "name": "grid"},
            {'params': list(self._opacity_head.parameters()), 'lr': 0.0, "name": "opacity"},
            {'params': self._features_rest_head.parameters(), 'lr': 0.0, "name": "f_rest"},
            {'params': self._scaling_head.parameters(), 'lr': 0.0, "name": "scaling"},
            {'params': list(self._rotation_head.parameters()), 'lr': 0.0, "name": "rotation"},
        ]
        self.optimizer_i = torch.optim.Adam(param_groups_i, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init,
                                                     lr_final=training_args.grid_lr_final,
                                                     lr_delay_steps=training_args.grid_lr_delay_steps,
                                                     lr_delay_mult=training_args.grid_lr_delay_mult,
                                                     max_steps=training_args.grid_lr_max_steps)

        self.opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.opacity_lr_init,
                                                        lr_final=training_args.opacity_lr_final,
                                                        lr_delay_steps=training_args.opacity_lr_delay_steps,
                                                        lr_delay_mult=training_args.opacity_lr_delay_mult,
                                                        max_steps=training_args.opacity_lr_max_steps)

        self.feature_scheduler_args = get_expon_lr_func(lr_init=training_args.feature_rest_lr_init,
                                                        lr_final=training_args.feature_rest_lr_final,
                                                        lr_delay_steps=training_args.feature_rest_lr_delay_steps,
                                                        lr_delay_mult=training_args.feature_rest_lr_delay_mult,
                                                        max_steps=training_args.feature_rest_lr_max_steps)

        self.scaling_scheduler_args = get_expon_lr_func(lr_init=training_args.scaling_lr_init,
                                                        lr_final=training_args.scaling_lr_final,
                                                        lr_delay_steps=training_args.scaling_lr_delay_steps,
                                                        lr_delay_mult=training_args.scaling_lr_delay_mult,
                                                        max_steps=training_args.scaling_lr_max_steps)

        self.rotation_scheduler_args = get_expon_lr_func(lr_init=training_args.rotation_lr_init,
                                                         lr_final=training_args.rotation_lr_final,
                                                         lr_delay_steps=training_args.rotation_lr_delay_steps,
                                                         lr_delay_mult=training_args.rotation_lr_delay_mult,
                                                         max_steps=training_args.rotation_lr_max_steps)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
        
        for param_group in self.optimizer_i.param_groups:
            if param_group["name"] == "grid":
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "opacity":
                lr = self.opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "f_dc":
                lr = self.feature_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "f_rest":
                lr = self.feature_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "scaling":
                lr = self.scaling_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "rotation":
                lr = self.rotation_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = torch.clamp_min(self.scaling_base_inverse_activation(self.get_scaling), min=-20).detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # def load_ply(self, path):
    #     plydata = PlyData.read(path)

    #     xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
    #                     np.asarray(plydata.elements[0]["y"]),
    #                     np.asarray(plydata.elements[0]["z"])),  axis=1)
    #     opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    #     features_dc = np.zeros((xyz.shape[0], 3, 1))
    #     features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    #     features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    #     features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    #     extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    #     extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    #     assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
    #     features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    #     for idx, attr_name in enumerate(extra_f_names):
    #         features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    #     # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    #     features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

    #     scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    #     scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    #     scales = np.zeros((xyz.shape[0], len(scale_names)))
    #     for idx, attr_name in enumerate(scale_names):
    #         scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    #     rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    #     rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    #     rots = np.zeros((xyz.shape[0], len(rot_names)))
    #     for idx, attr_name in enumerate(rot_names):
    #         rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    #     sh_mask = np.stack([np.asarray(plydata.elements[0][f"sh_mask_{i}"]) for i in range(3)],  axis=1)

    #     self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    #     self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    #     self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    #     self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    #     self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    #     self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
    #     self._sh_mask = nn.Parameter(torch.tensor(sh_mask, dtype=torch.float, device="cuda").requires_grad_(True))

    #     self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._scaling_base = optimizable_tensors["scaling_base"]
        self._mask = optimizable_tensors["mask"]
        self._sh_mask = optimizable_tensors["sh_mask"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_scaling_base, new_mask, new_sh_mask):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "scaling_base": new_scaling_base,
            "mask": new_mask,
            "sh_mask": new_sh_mask,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._scaling_base = optimizable_tensors["scaling_base"]
        self._mask = optimizable_tensors["mask"]
        self._sh_mask = optimizable_tensors["sh_mask"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_scaling_base = self.scaling_base_inverse_activation(self.get_scaling_base[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_mask = self._mask[selected_pts_mask].repeat(N,1)
        new_sh_mask = self._sh_mask[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_scaling_base, new_mask, new_sh_mask)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_scaling_base = self._scaling_base[selected_pts_mask]
        new_mask = self._mask[selected_pts_mask]
        new_sh_mask = self._sh_mask[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_scaling_base, new_mask, new_sh_mask)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.update_attributes()
        self.densify_and_clone(grads, max_grad, extent)
        self.update_attributes()
        self.densify_and_split(grads, max_grad, extent)

        self.update_attributes()
        prune_mask = torch.logical_or((self.get_mask <= 0.01).squeeze(), (self.get_opacity < min_opacity).squeeze())
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def mask_prune(self):
        prune_mask = (self.get_mask <= 0.01).squeeze()
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def sort_attributes(self):
        xyz = self.get_xyz.float().detach().cpu().numpy()
        xyz_q = (xyz - xyz.min(0)) / (xyz.max(0) - xyz.min(0)) * (2 ** 21 - 1)
        order = mortonEncode(torch.tensor(xyz_q).long()).sort().indices

        self._xyz = self._xyz[order]
        self._scaling_base = self._scaling_base[order]
        self._features_dc = self._features_dc[order]
        self._sh_mask = self._sh_mask[order]

    def compress_attributes(self, path):
        os.makedirs(path, exist_ok=True)

        self.mask_prune()
        self.sort_attributes()

        xyz = self.get_xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().cpu().numpy()
        scale_base = self._scaling_base.detach().cpu().numpy()
        sh_mask = self.get_sh_mask.detach().cpu().numpy()
        
        # Save model configuration for correct loading
        config = {
            'feature_role_split': self._feature_role_split,
            'geo_resolution': self._geo_resolution,
            'geo_rank': self._geo_rank,
            'geo_channels': self._geo_channels,
            'max_sh_degree': self.max_sh_degree,
        }
        np.savez(os.path.join(path, "config.npz"), **config)
        
        # Save the complete GeometryAppearanceEncoder state_dict
        # This includes hash_encoder, geo_encoder, and fusion_layer parameters
        # All files use underscore naming format for consistency
        
        grid_state_dict = self._grid.state_dict()
        # Pack all grid components into a single compressed archive to reduce header
        # overhead and improve compression across parameters.
        grid_save_dict = {}
        for key, value in grid_state_dict.items():
            normalized_key = key.replace('.', '_')  # e.g., "hash_encoder.params" -> "hash_encoder_params"
            param_data = value.detach().cpu().numpy()
            # Quantize grid parameters (log quantization) to uint8
            param_quant, param_scale, param_min = quantize(param_data, bit=6, log=True)
            grid_save_dict[f"{normalized_key}_data"] = param_quant.astype(np.uint8)
            grid_save_dict[f"{normalized_key}_scale"] = param_scale
            grid_save_dict[f"{normalized_key}_min"] = param_min
            grid_save_dict[f"{normalized_key}_orig"] = np.array(key)

        # Save single combined file
        np.savez_compressed(os.path.join(path, "grid_all.npz"), **grid_save_dict)

        # G-PCC encoding
        encode_xyz(xyz, path)

        # uniform quantization
        f_dc_quant, f_dc_scale, f_dc_min = quantize(f_dc, bit=8)
        scaling_base_quant, scaling_base_scale, scaling_base_min = quantize(scale_base, bit=8)

        np.savez_compressed(os.path.join(path, "f_dc.npz"), data=f_dc_quant.astype(np.uint8), scale=f_dc_scale, min=f_dc_min)
        np.savez_compressed(os.path.join(path, "scale_base.npz"), data=scaling_base_quant.astype(np.uint8), scale=scaling_base_scale, min=scaling_base_min)
        
        # sh mask
        sh_mask = (sh_mask > 0.01).cumprod(axis=-1).astype(np.uint8)
        np.savez_compressed(os.path.join(path, "sh_mask.npz"), data=sh_mask)

        # mlps: quantize MLP parameter tensors to 8-bit (log) and pack into one file
        mlp_save = {}
        opacity_params = self._opacity_head.state_dict()['params'].cpu().detach().numpy()
        f_rest_params = self._features_rest_head.state_dict()['params'].cpu().detach().numpy()
        scale_params = self._scaling_head.state_dict()['params'].cpu().detach().numpy()
        rotation_params = self._rotation_head.state_dict()['params'].cpu().detach().numpy()

        o_q, o_s, o_m = quantize(opacity_params, bit=8, log=True)
        fr_q, fr_s, fr_m = quantize(f_rest_params, bit=8, log=True)
        sc_q, sc_s, sc_m = quantize(scale_params, bit=8, log=True)
        rt_q, rt_s, rt_m = quantize(rotation_params, bit=8, log=True)

        mlp_save['opacity_data'] = o_q.astype(np.uint8)
        mlp_save['opacity_scale'] = o_s
        mlp_save['opacity_min'] = o_m
        mlp_save['f_rest_data'] = fr_q.astype(np.uint8)
        mlp_save['f_rest_scale'] = fr_s
        mlp_save['f_rest_min'] = fr_m
        mlp_save['scale_data'] = sc_q.astype(np.uint8)
        mlp_save['scale_scale'] = sc_s
        mlp_save['scale_min'] = sc_m
        mlp_save['rotation_data'] = rt_q.astype(np.uint8)
        mlp_save['rotation_scale'] = rt_s
        mlp_save['rotation_min'] = rt_m

        np.savez_compressed(os.path.join(path, "mlps.npz"), **mlp_save)

        # Calculate and print total size
        size = 0
        size += os.path.getsize(os.path.join(path, "xyz.bin"))
        print("xyz.bin", os.path.getsize(os.path.join(path, "xyz.bin")) / 1e6)
        size += os.path.getsize(os.path.join(path, "f_dc.npz"))
        print("f_dc.npz", os.path.getsize(os.path.join(path, "f_dc.npz")) / 1e6)
        size += os.path.getsize(os.path.join(path, "scale_base.npz"))
        print("scale_base.npz", os.path.getsize(os.path.join(path, "scale_base.npz")) / 1e6)
        
        # Calculate total size of grid archive (prefer combined file)
        grid_total_size = 0
        grid_all_path = os.path.join(path, "grid_all.npz")
        if os.path.exists(grid_all_path):
            grid_total_size = os.path.getsize(grid_all_path)
            print("grid_all.npz", grid_total_size / 1e6)
        else:
            for filename in os.listdir(path):
                if filename.startswith("grid_") and filename.endswith(".npz"):
                    file_path = os.path.join(path, filename)
                    file_size = os.path.getsize(file_path)
                    grid_total_size += file_size
                    print(f"{filename}", file_size / 1e6)
        size += grid_total_size
        print(f"grid_total (all components)", grid_total_size / 1e6)
        
        size += os.path.getsize(os.path.join(path, "sh_mask.npz"))
        print("sh_mask.npz", os.path.getsize(os.path.join(path, "sh_mask.npz")) / 1e6)
        # mlps are saved in a single file now
        if os.path.exists(os.path.join(path, "mlps.npz")):
            size += os.path.getsize(os.path.join(path, "mlps.npz"))
            print("mlps.npz", os.path.getsize(os.path.join(path, "mlps.npz")) / 1e6)
        else:
            # backward-compatible individual mlp files
            for name in ["opacity.npz", "f_rest.npz", "scale.npz", "rotation.npz"]:
                if os.path.exists(os.path.join(path, name)):
                    size += os.path.getsize(os.path.join(path, name))
                    print(name, os.path.getsize(os.path.join(path, name)) / 1e6)

        print(f"Total size: {(size / 1e6)} MB")

    def decompress_attributes(self, path):
        # Load and verify configuration
        config_path = os.path.join(path, 'config.npz')
        if os.path.exists(config_path):
            config = np.load(config_path)
            # Verify config matches current model (print warnings if mismatch)
            saved_feature_role_split = bool(config['feature_role_split'])
            if saved_feature_role_split != self._feature_role_split:
                print(f"[WARNING] Model feature_role_split mismatch: saved={saved_feature_role_split}, current={self._feature_role_split}")
            # Support both old 'glf_*' and new 'geo_*' config key names
            if 'geo_resolution' in config:
                saved_geo_resolution = int(config['geo_resolution'])
            elif 'glf_resolution' in config:
                saved_geo_resolution = int(config['glf_resolution'])
            else:
                saved_geo_resolution = self._geo_resolution
            if saved_geo_resolution != self._geo_resolution:
                print(f"[WARNING] GeoEncoder resolution mismatch: saved={saved_geo_resolution}, current={self._geo_resolution}")
        
        # G-PCC decoding
        xyz = decode_xyz(path)
        self._xyz = torch.from_numpy(np.asarray(xyz)).cuda()

        # uniform dequantization
        f_dc_dict = np.load(os.path.join(path, 'f_dc.npz'))
        scale_base_dict = np.load(os.path.join(path, 'scale_base.npz'))

        f_dc = dequantize(f_dc_dict['data'], f_dc_dict['scale'], f_dc_dict['min'])
        scaling_base = dequantize(scale_base_dict['data'], scale_base_dict['scale'], scale_base_dict['min'])

        self._features_dc = torch.from_numpy(np.asarray(f_dc)).cuda()
        self._scaling_base = torch.from_numpy(np.asarray(scaling_base)).cuda()
        
        grid_state_dict = {}
        # Prefer loading combined grid archive if present
        grid_all_path = os.path.join(path, "grid_all.npz")
        if os.path.exists(grid_all_path):
            grid_npz = np.load(grid_all_path)
            # find unique normalized keys by scanning files ending with _data
            for name in grid_npz.files:
                if not name.endswith("_data"):
                    continue
                base = name[:-5]  # remove '_data'
                data = grid_npz[f"{base}_data"]
                scale = grid_npz[f"{base}_scale"]
                mn = grid_npz[f"{base}_min"]
                grid_params = dequantize(data, scale, mn, log=True)
                orig_arr = grid_npz[f"{base}_orig"]
                orig_key = orig_arr.item() if np.asarray(orig_arr).shape == () else str(orig_arr)
                grid_state_dict[orig_key] = torch.from_numpy(np.asarray(grid_params)).half().cuda()
        else:
            for filename in os.listdir(path):
                if filename.startswith("grid_") and filename.endswith(".npz"):
                    grid_dict = np.load(os.path.join(path, filename))
                    grid_params = dequantize(grid_dict['data'], grid_dict['scale'], grid_dict['min'], log=True)

                    orig_key = grid_dict['orig_key'].item() if grid_dict['orig_key'].shape == () else str(grid_dict['orig_key'])
                    grid_state_dict[orig_key] = torch.from_numpy(np.asarray(grid_params)).half().cuda()

        if grid_state_dict:
            self._grid.load_state_dict(grid_state_dict, strict=True)

        # sh_mask
        sh_mask = np.load(os.path.join(path, 'sh_mask.npz'))['data'].astype(np.float32)
        self._sh_mask = torch.from_numpy(np.asarray(sh_mask)).cuda()

        # mlps: prefer combined mlps.npz (quantized) else fallback to older float16 files
        mlps_path = os.path.join(path, 'mlps.npz')
        if os.path.exists(mlps_path):
            mlp_npz = np.load(mlps_path)
            o = dequantize(mlp_npz['opacity_data'], mlp_npz['opacity_scale'], mlp_npz['opacity_min'], log=True)
            fr = dequantize(mlp_npz['f_rest_data'], mlp_npz['f_rest_scale'], mlp_npz['f_rest_min'], log=True)
            sc = dequantize(mlp_npz['scale_data'], mlp_npz['scale_scale'], mlp_npz['scale_min'], log=True)
            rt = dequantize(mlp_npz['rotation_data'], mlp_npz['rotation_scale'], mlp_npz['rotation_min'], log=True)

            self._opacity_head.params = nn.Parameter(torch.from_numpy(np.asarray(o)).half().cuda())
            self._features_rest_head.params = nn.Parameter(torch.from_numpy(np.asarray(fr)).half().cuda())
            self._scaling_head.params = nn.Parameter(torch.from_numpy(np.asarray(sc)).half().cuda())
            self._rotation_head.params = nn.Parameter(torch.from_numpy(np.asarray(rt)).half().cuda())
        else:
            # backward compatible loading
            opacity_params = np.load(os.path.join(path, 'opacity.npz'))['data']
            f_rest_params = np.load(os.path.join(path, 'f_rest.npz'))['data']
            scale_params = np.load(os.path.join(path, 'scale.npz'))['data']
            rotation_params = np.load(os.path.join(path, 'rotation.npz'))['data']

            self._opacity_head.params = nn.Parameter(torch.from_numpy(np.asarray(opacity_params)).half().cuda())
            self._features_rest_head.params = nn.Parameter(torch.from_numpy(np.asarray(f_rest_params)).half().cuda())
            self._scaling_head.params = nn.Parameter(torch.from_numpy(np.asarray(scale_params)).half().cuda())
            self._rotation_head.params = nn.Parameter(torch.from_numpy(np.asarray(rotation_params)).half().cuda())

        self.active_sh_degree = self.max_sh_degree

    def load_attributes(self, path):
        self.decompress_attributes(path)
        self.update_attributes()
        sh_mask = self._sh_mask
        sh_mask_degree = torch.ones_like(self._features_rest) # (N, 15, 3)
        for deg in range(1, self.active_sh_degree + 1):
            sh_mask_degree[:, deg**2 - 1:, :] *= sh_mask[:, deg - 1:deg].unsqueeze(1)
        
        self._features_rest *= sh_mask_degree
        self.sh_levels = sh_mask.sum(1).int()