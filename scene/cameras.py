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
from torch import nn
import numpy as np
from pathlib import Path
from PIL import Image
import os 
import cv2
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,image_path=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        tan_fovx = np.tan(self.FoVx / 2.0)
        tan_fovy = np.tan(self.FoVy / 2.0)
        self.focal_y = self.image_height / (2.0 * tan_fovy)
        self.focal_x = self.image_width / (2.0 * tan_fovx)
        # for DTU_robust
        path = Path(image_path)
        new_parent = path.parent.parent / 'feature_mask_sam'
        image_name=os.path.splitext(os.path.basename(image_path))[0]
        mask_path_sam = str(new_parent / image_name) + '.png'
        print("mask_path_sam",mask_path_sam)
        if os.path.exists(mask_path_sam):
            # print(" SAM mask loaded!")
            self.feature_mask_sam = torch.tensor(cv2.imread(mask_path_sam, cv2.IMREAD_GRAYSCALE)).to(self.data_device).squeeze()/255
            self.feature_mask_sam = torch.nn.functional.interpolate(self.feature_mask_sam[None,None], size=(self.image_height,self.image_width), mode='bilinear', align_corners=False).squeeze()
        else:
            # print(" SAM mask not loaded!")
            self.feature_mask_sam = None
         
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

