from copy import deepcopy
import numpy as np
import torch
from torch import nn

from pointnet2.pointnet2_modules import PointnetSAModuleMSG
from pointnet2.pointnet2_utils import furthest_point_sample, gather_operation

from .layers import BEVFeatureGatherer, VoxelFeatureExtractor
from .roi_grid_pool import RoiGridPool
from .sparse_cnn import CNN_FACTORY
from .proposal import ProposalLayer
from .refinement import RefinementLayer


class PV_RCNN(nn.Module):
    """
    TODO: Improve docstrings.
    TODO: Some docstrings may claim incorrect dimensions.
    TODO: Figure out clean way to handle proposals_only forward.
    """

    def __init__(self, cfg):
        super(PV_RCNN, self).__init__()
        self.pnets = self.build_pointnets(cfg)
        self.roi_grid_pool = RoiGridPool(cfg)
        self.vfe = VoxelFeatureExtractor()
        self.cnn = CNN_FACTORY[cfg.CNN](cfg)
        self.bev = BEVFeatureGatherer(
            cfg, self.cnn.voxel_offset, self.cnn.base_voxel_size)
        self.proposal_layer = ProposalLayer(cfg)
        self.refinement_layer = RefinementLayer(cfg)
        self.cfg = cfg

    def build_pointnets(self, cfg):
        """Copy list because PointNet modifies it in-place."""
        pnets = []
        for i, mlps in enumerate(cfg.PSA.MLPS):
            pnets += [PointnetSAModuleMSG(
                npoint=-1, radii=cfg.PSA.RADII[i],
                nsamples=cfg.SAMPLES_PN,
                mlps=deepcopy(mlps), use_xyz=True,
            )]
        return nn.Sequential(*pnets)

    def sample_keypoints(self, points):
        """
        fps expects points shape (B, N, 3)
        fps returns indices shape (B, K)
        gather expects features shape (B, C, N)
        """
        points = points[..., :3].contiguous()
        indices = furthest_point_sample(points, self.cfg.NUM_KEYPOINTS)
        keypoints = gather_operation(points.transpose(1, 2).contiguous(), indices)
        keypoints = keypoints.transpose(1, 2).contiguous()
        return keypoints

    def _pointnets(self, cnn_out, keypoint_xyz):
        """xyz (B, N, 3) | features (B, N, C) | new_xyz (B, M, C) | return (B, M, Co)"""
        pnet_out = []
        for (voxel_xyz, voxel_features), pnet in zip(cnn_out, self.pnets):
            voxel_xyz = voxel_xyz.contiguous()
            voxel_features = voxel_features.transpose(1, 2).contiguous()
            out = pnet(voxel_xyz, voxel_features, keypoint_xyz)[1]
            pnet_out += [out]
        return pnet_out

    def point_feature_extract(self, item, cnn_features, bev_map):
        points_split = torch.split(item['points'], [3, 1], dim=-1)
        cnn_features = [points_split] + cnn_features
        point_features = self._pointnets(cnn_features, item['keypoints'])
        bev_features = self.bev(bev_map, item['keypoints'])
        point_features = torch.cat(point_features + [bev_features], dim=1)
        return point_features

    def proposal(self, item):
        item['keypoints'] = self.sample_keypoints(item['points'])
        features = self.vfe(item['features'], item['occupancy'])
        cnn_features, bev_map = self.cnn(features, item['coordinates'], item['batch_size'])
        scores, boxes = self.proposal_layer(bev_map)
        item.update(dict(P_cls=scores, P_reg=boxes))
        return item
        """
        input item:['idx', 'boxes', 'class_idx', 'points', 'G_cls', 'G_reg', 
                    'M_cls', 'M_reg', 'features', 'coordinates', 'occupancy', 'batch_size']

        output item: dict()
            idx: ['1543998917815', '1542877835047', '1544430249777', ...], len=8
            boxes: [torch.Size([33, 7]), torch.Size([39, 7]), torch.Size([41, 7]), ...], len=8
            class_idx: [torch.Size([33]), torch.Size([39]), torch.Size([41]), ...]], len=8
            points: torch.Size([8, 145374, 4])
            G_cls: torch.Size([8, 4, 2, 200, 176])
            G_reg: torch.Size([8, 3, 2, 200, 176, 7])
            M_cls: torch.Size([8, 1, 2, 200, 176])
            M_reg: torch.Size([8, 3, 2, 200, 176, 1])
            features: torch.Size([157606, 5, 4])
            coordinates: torch.Size([157606, 4])
            occupancy: torch.Size([157606])
            batch_size: 8
            keypoints: torch.Size([8, 2048, 3])       ==> new added 
            P_cls: torch.Size([8, 4, 2, 200, 176])    ==> new added 
            P_reg: torch.Size([8, 3, 2, 200, 176, 7]) ==> new added 

        features: torch.Size([157606, 4])
        bev_map: torch.Size([8, 128, 200, 176])
        cnn_features:[tuple(a,b)*4]
        [ (torch.Size([8, 20000, 3]), torch.Size([8, 20000, 4]))
          (torch.Size([8, 46917, 3]), torch.Size([8, 46917, 32]))
          (torch.Size([8, 37164, 3]), torch.Size([8, 37164, 64]))
          (torch.Size([8, 16283, 3]), torch.Size([8, 16283, 64])) ]
        scores: torch.Size([8, 4, 2, 200, 176])
        boxes: torch.Size([8, 3, 2, 200, 176, 7])
        """

    def forward(self, item):
        raise self.proposal(proposal)
