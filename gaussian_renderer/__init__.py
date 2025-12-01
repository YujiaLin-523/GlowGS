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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization_sh import GaussianRasterizationSHSettings, GaussianRasterizerSH
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    pc.update_attributes()

    means3D = pc.get_xyz
    means2D = screenspace_points

    mask = pc.get_mask
    mask = ((mask > 0.01).float() - mask).detach() + mask
    
    sh_mask = pc.get_sh_mask
    sh_mask = ((sh_mask > 0.01).float() - sh_mask).detach() + sh_mask

    opacity = pc.get_opacity * mask

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling * mask
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features
            # Fully vectorized SH mask application (no loop, faster)
            # Build cumulative mask: degree i masks all coefficients from degree i onward
            if pc.active_sh_degree > 0:
                # Pre-compute degree boundaries: [1, 4, 9, 16, ...] for degrees [0, 1, 2, 3, ...]
                # Mask structure: sh_mask is (N, max_degree) where sh_mask[:, d-1] applies to degree d
                # We need to apply cumulative product of masks
                N, C, _ = shs_view.shape
                sh_mask_expanded = torch.ones((N, C, 3), device=shs_view.device, dtype=shs_view.dtype)
                
                # Vectorized: apply mask to corresponding degree ranges
                for degree in range(1, min(pc.active_sh_degree + 1, sh_mask.shape[1] + 1)):
                    start_idx = degree ** 2
                    end_idx = min((degree + 1) ** 2, C) if degree < pc.max_sh_degree else C
                    # Broadcast mask along the coefficient dimension
                    sh_mask_expanded[:, start_idx:end_idx, :] *= sh_mask[:, degree - 1:degree, None]
                
                shs_view = shs_view * sh_mask_expanded
            
            shs_view = shs_view.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            # Optimize: avoid repeat(), use broadcasting instead
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center[None, :]
            dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-7)  # Add epsilon for stability
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # 改进：使用sigmoid替代固定偏移，避免整体偏亮和去饱和
            colors_precomp = torch.sigmoid(sh2rgb)
            colors_precomp = torch.clamp(colors_precomp, 0.0, 1.0)
        else:
            shs = pc.get_features
            # Fully vectorized SH mask application (no loop, faster)
            if pc.active_sh_degree > 0:
                N, C, _ = shs.shape
                sh_mask_expanded = torch.ones((N, C, 3), device=shs.device, dtype=shs.dtype)
                
                for degree in range(1, min(pc.active_sh_degree + 1, sh_mask.shape[1] + 1)):
                    start_idx = degree ** 2
                    end_idx = min((degree + 1) ** 2, C) if degree < pc.max_sh_degree else C
                    sh_mask_expanded[:, start_idx:end_idx, :] *= sh_mask[:, degree - 1:degree, None]
                
                shs = shs * sh_mask_expanded
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def render_eval(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSHSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizerSH(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # 改进：使用sigmoid替代固定偏移，避免整体偏亮和去饱和
            # 原代码：colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            colors_precomp = torch.sigmoid(sh2rgb)
            colors_precomp = torch.clamp(colors_precomp, 0.0, 1.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    sh_levels = pc.sh_levels

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        sh_levels = sh_levels,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
