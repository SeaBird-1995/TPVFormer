'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-08-11 11:44:58
Email: haimingzhang@link.cuhk.edu.cn
Description: TPVFormer with the lidar point cloud fusion.
'''
import torch
from torch.nn import functional as F
from mmcv.runner import force_fp32, auto_fp16, BaseModule
from mmseg.models import SEGMENTORS
from mmdet3d.models import builder
from mmcv.ops import Voxelization
from dataloader.grid_mask import GridMask
from .tpvformer import TPVFormer


@SEGMENTORS.register_module()
class TPVFusionFormer(TPVFormer):

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_backbone=None,
                 pts_neck=None,
                 **kwargs,
                 ):

        super().__init__(**kwargs)
        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        if pts_voxel_encoder:
            self.pts_voxel_encoder = builder.build_voxel_encoder(
                pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = builder.build_middle_encoder(
                pts_middle_encoder)
        if pts_backbone:
            self.pts_backbone = builder.build_backbone(pts_backbone)
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)

    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in 3D detector branch."""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None
    
    @auto_fp16(apply_to=('img', 'points', 'point_cloud'))
    def forward(self,
                points=None,
                point_cloud=None,
                img_metas=None,
                img=None,
                use_grid_mask=None,
        ):
        """Forward training function.
        """
        img_feats = self.extract_img_feat(img=img, use_grid_mask=use_grid_mask)
        outs = self.tpv_head(img_feats, img_metas)
        
        # [(b,384,200,200)]
        pts_feats = self.extract_pts_feat(point_cloud)

        outs = self.tpv_aggregator(outs, points, pts_feats)
        return outs
    
    def extract_pts_feat(self, pts, img_feats=None, img_metas=None):
        """Extract features of points."""
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(
            voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x
    
    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch