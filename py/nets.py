from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import BendingEnergyLoss
from monai.networks.nets import Regressor, VoxelMorph
from monai.networks.nets.regunet import AffineHead
from monai.networks.blocks import Warp
import numpy as np
from constants import (
    RIGID_INSHAPE,
    CHANNELS,
    STRIDES,
    DDF_REGULARIZATION,
    ELASTIC_INSHAPE,
)


class RegressorModel(nn.Module):
    def __init__(self, inshape=RIGID_INSHAPE, channels=CHANNELS, strides=STRIDES):
        super().__init__()
        outshape = (1024,)
        self.conv = Regressor(
            inshape, outshape, channels, strides, norm="batch", act="gelu"
        )  # TODO: consider changing norm to group / instance
        self.act_center = nn.GELU()
        self.act_orient = nn.GELU()
        self.center = nn.Linear(1024, 3)
        self.orient = nn.Linear(1024, 6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        pred_AC = self.sigmoid(self.center(self.act_center(x)))
        directions = self.orient(self.act_orient(x)).reshape(-1, 2, 3)
        pred_PA = directions[:, 0]
        pred_IS = directions[:, 1]
        pred_PA = pred_PA / (pred_PA.norm(dim=1)).reshape(-1, 1)
        pred_IS = pred_IS / (pred_IS.norm(dim=1)).reshape(-1, 1)
        return pred_AC, pred_PA, pred_IS

    def loss(self, pred, target):
        pred_AC, pred_PA, pred_IS = pred
        gt_AC, gt_PA, gt_IS = target
        pos_loss = F.smooth_l1_loss(pred_AC, gt_AC, reduce="mean")
        orient_loss = (1 - F.cosine_similarity(pred_PA, gt_PA)).mean()
        orient_loss2 = (1 - F.cosine_similarity(pred_IS, gt_IS)).mean()
        return {
            "AC Loss": pos_loss,
            "PA Loss": orient_loss,
            "IS Loss": orient_loss2,
            "Total": pos_loss + orient_loss + orient_loss2,
        }

    def export(self, output_path):
        self.eval()
        self.to("cpu")
        dummy_input = torch.randn(1, *RIGID_INSHAPE)
        with torch.no_grad():
            pred = self(dummy_input)
            print(pred)
            torch.onnx.export(
                self,
                dummy_input,
                output_path,
                input_names=["input"],
                output_names=["ac", "nose", "top"],
            )


class AffineVoxelMorph(nn.Module):
    def __init__(self, mask=None) -> None:
        super().__init__()
        inshape = (1, *ELASTIC_INSHAPE)
        outshape = (3, 4)
        channels = CHANNELS
        strides = STRIDES

        spatial_dims = 3
        image_size = inshape[1:]
        decode_size = (
            np.array(image_size) // reduce(lambda x, y: x * y, strides)
        ).tolist()
        in_channels = channels[-1]

        self.mask = mask if mask is not None else 1

        self.conv = Regressor(
            inshape, outshape, channels, strides, act="gelu", norm="batch"
        ).net  # TODO: consider changing norm to group / instance
        self.affine_head = AffineHead(
            spatial_dims, image_size, decode_size, in_channels
        )
        self.vm = VoxelMorph()
        self.warp = Warp(mode="bilinear", padding_mode="zeros")

    def forward(self, moving_image, fixed_image):
        features = self.conv(moving_image)
        affine = self.affine_head([features], None)
        affine_warped = self.warp(moving_image, affine)
        elastic_warped, ddf_image = self.vm(affine_warped.detach(), fixed_image)
        return affine_warped, affine, elastic_warped, ddf_image

    def loss(self, pred, target):
        affine_warped, _, elastic_warped, ddf_image = pred
        mse_loss = nn.MSELoss()
        regularization = BendingEnergyLoss()

        affine_sim = mse_loss(affine_warped * self.mask, target * self.mask)
        elastic_sim = mse_loss(elastic_warped * self.mask, target * self.mask)
        reg = regularization(ddf_image)
        return {
            "Affine Similarity": affine_sim,
            "Elastic Similarity": elastic_sim,
            "Regularization": reg,
            "Total": affine_sim + elastic_sim + reg * DDF_REGULARIZATION,
        }


class AffineVoxelMorphONNX(AffineVoxelMorph):
    def forward(self, moving_image, fixed_image, original_image):
        features = self.conv(moving_image)
        affine = self.affine_head([features], None)
        affine_warped = self.warp(moving_image, affine)
        _, ddf_image = self.vm(affine_warped, fixed_image)
        original_warped = self.warp(self.warp(original_image, affine), ddf_image)
        return original_warped

    def export(self, output_path):
        self.eval()
        self.to("cpu")
        dummy_input = (
            torch.randn(1, 1, *ELASTIC_INSHAPE),
            torch.randn(1, 1, *ELASTIC_INSHAPE),
            torch.randn(1, 1, *ELASTIC_INSHAPE),
        )
        with torch.no_grad():
            ddf = self(*dummy_input)
            print(ddf.shape)
            torch.onnx.export(
                self,
                dummy_input,
                output_path,
                verbose=False,
                opset_version=20,
                input_names=["input", "template", "input_raw"],
                output_names=["warped"],
            )
