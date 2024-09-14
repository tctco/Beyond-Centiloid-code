from pathlib import Path

from tqdm import tqdm
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.transforms import (
    MapTransform,
    Compose,
    Resized,
    LoadImaged,
    RandAffined,
    ClipIntensityPercentilesd,
    ScaleIntensityd,
    Spacingd,
    GaussianSmoothd,
    RandCoarseShuffled,
    CropForegroundd,
    RandAxisFlipd,
    ApplyPendingd,
    RandGaussianNoised,
)
from monai.networks.nets import Regressor
from monai.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INSHAPE = (1, 64, 64, 64)
CHANNELS = (64, 64, 64, 128, 256)
STRIDES = (2, 2, 2, 2, 2)
transform = Compose(
    [
        LoadImaged(keys=["image"], ensure_channel_first=True),
        ClipIntensityPercentilesd(keys=["image"], lower=1, upper=99),
        ScaleIntensityd(keys=["image"]),
        GaussianSmoothd(keys=["image"], sigma=1),
        Spacingd(keys=["image"], pixdim=(3, 3, 3), mode=("bilinear"), lazy=True),
        CropForegroundd(
            keys=["image"], source_key="image", select_fn=lambda x: x > 0.35, lazy=True
        ),
        Resized(keys=["image"], spatial_size=INSHAPE[1:], lazy=True),
        ApplyPendingd(keys=["image"]),
    ]
)


class RegressorModel(nn.Module):
    def __init__(self, inshape=INSHAPE, channels=CHANNELS, strides=STRIDES):
        super().__init__()
        outshape = (1024,)
        self.conv = Regressor(
            inshape, outshape, channels, strides, norm="batch", act="gelu"
        )
        self.act_center = nn.GELU()
        self.act_orient = nn.GELU()
        self.center = nn.Linear(1024, 3)
        self.orient = nn.Linear(1024, 6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        pred_vox_pos_vec = self.sigmoid(self.center(self.act_center(x)))
        vecs = self.orient(self.act_orient(x)).reshape(-1, 2, 3)
        pred_vox_orient_vec = vecs[:, 0]
        pred_vox_orient_vec2 = vecs[:, 1]
        pred_vox_orient_vec = pred_vox_orient_vec / (
            0.1 * pred_vox_orient_vec.norm(dim=1)
        ).reshape(-1, 1)
        pred_vox_orient_vec2 = pred_vox_orient_vec2 / (
            0.1 * pred_vox_orient_vec2.norm(dim=1)
        ).reshape(-1, 1)
        return pred_vox_pos_vec, pred_vox_orient_vec, pred_vox_orient_vec2

    def loss(self, pred, target, eval=False):
        pred_vox_pos_vec, pred_vox_orient_vec, pred_vox_orient_vec2 = pred
        target_vox_pos_vec, target_vox_orient_vec, target_vox_orient_vec2 = target
        pos_loss = F.l1_loss(pred_vox_pos_vec, target_vox_pos_vec).mean()
        orient_loss = (
            1 - F.cosine_similarity(pred_vox_orient_vec, target_vox_orient_vec)
        ).mean()
        orient_loss2 = (
            1 - F.cosine_similarity(pred_vox_orient_vec2, target_vox_orient_vec2)
        ).mean()
        if not eval:
            return pos_loss + orient_loss + orient_loss2
        else:
            return pos_loss, orient_loss, orient_loss2


class ExtractTarget(MapTransform):
    """
    Extract Voxel AC position, Posterior-Anterior direction,
    Inferior-Superior direction based on affine matrix.
    """

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            vox_pos_vec, vox_orient_vec, vox_orient_vec2 = self._extract_target(
                d[key].affine, d[key].shape
            )
            d[key + "_pos"] = torch.tensor(vox_pos_vec, dtype=torch.float32)
            d[key + "_orient"] = torch.tensor(vox_orient_vec, dtype=torch.float32)
            d[key + "_orient2"] = torch.tensor(vox_orient_vec2, dtype=torch.float32)
        return d

    def _extract_target(self, affine, inshape):
        phy_pos_vec = np.array([0, 0, 0, 1])
        phy_orient_vec = np.array([0, 99999, 0, 1])
        phy_orient_vec2 = np.array([0, 0, 99999, 1])
        vox_pos_vec = np.dot(np.linalg.inv(affine), phy_pos_vec)[:3]
        vox_pos_vec = vox_pos_vec / inshape[1:4]
        vox_orient_vec = np.dot(np.linalg.inv(affine), phy_orient_vec)[:3]
        vox_orient_vec = vox_orient_vec / np.linalg.norm(vox_orient_vec)
        vox_orient_vec2 = np.dot(np.linalg.inv(affine), phy_orient_vec2)[:3]
        vox_orient_vec2 = vox_orient_vec2 / np.linalg.norm(vox_orient_vec2)
        return vox_pos_vec, vox_orient_vec, vox_orient_vec2


def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def dl_rigid_transform_onnx(path, session):
    dataset = {"image": path}
    data = transform(dataset)
    image = data["image"]
    pred_vox_pos_vec, pred_vox_orient_vec, pred_vox_orient_vec2 = session.run(
        None, {"input": image.unsqueeze(0).numpy()}
    )
    pred_vox_pos_vec = pred_vox_pos_vec.squeeze()
    pred_vox_orient_vec = pred_vox_orient_vec.squeeze() * 99999
    pred_vox_orient_vec2 = pred_vox_orient_vec2.squeeze() * 99999
    pred_vox_pos_vec = pred_vox_pos_vec * 64
    pred_vox_pos_vec = np.append(pred_vox_pos_vec, 1).reshape(-1, 1)
    pred_vox_orient_vec = np.append(pred_vox_orient_vec, 1).reshape(-1, 1)
    pred_vox_orient_vec2 = np.append(pred_vox_orient_vec2, 1).reshape(-1, 1)
    affine = image.affine.cpu().numpy()
    pred_phy_pos_vec = affine @ pred_vox_pos_vec
    pred_phy_orient_vec = affine @ pred_vox_orient_vec - pred_phy_pos_vec
    pred_phy_orient_vec2 = affine @ pred_vox_orient_vec2 - pred_phy_pos_vec

    pred_phy_pos_vec = pred_phy_pos_vec.flatten()[0:3]
    pred_phy_orient_vec = pred_phy_orient_vec.flatten()[0:3]
    pred_phy_orient_vec2 = pred_phy_orient_vec2.flatten()[0:3]

    pred_phy_orient_vec2 = pred_phy_orient_vec2 / np.linalg.norm(pred_phy_orient_vec2)

    pred_phy_orient_vec = (
        pred_phy_orient_vec
        - pred_phy_orient_vec
        @ pred_phy_orient_vec2.reshape(-1, 1)
        * pred_phy_orient_vec2
    )
    pred_phy_orient_vec = pred_phy_orient_vec / np.linalg.norm(pred_phy_orient_vec)

    vy = pred_phy_orient_vec
    vz = pred_phy_orient_vec2
    vx = np.cross(vy, vz)

    img = nib.load(path)
    rotation_mat = np.eye(4)
    rotation_mat[:3, :3] = np.column_stack([vx, vy, vz]).T
    translate_mat = np.eye(4)
    translate_mat[:3, 3] = -pred_phy_pos_vec
    rigid_affine = rotation_mat @ translate_mat
    new_affine = rigid_affine @ img.affine
    new_nifti_image = nib.Nifti1Image(img.get_fdata(), affine=new_affine)
    return new_nifti_image


def dl_rigid_transform(path, model):
    dataset = {"image": path}
    data = transform(dataset)
    image = data["image"]
    image = image.unsqueeze(0)
    with torch.no_grad():
        pred_vox_pos_vec, pred_vox_orient_vec, pred_vox_orient_vec2 = model(
            image.to(DEVICE)
        )
    pred_vox_pos_vec = pred_vox_pos_vec.cpu().detach().numpy().squeeze()
    pred_vox_orient_vec = pred_vox_orient_vec.cpu().detach().numpy().squeeze() * 99999
    pred_vox_orient_vec2 = pred_vox_orient_vec2.cpu().detach().numpy().squeeze() * 99999
    pred_vox_pos_vec = pred_vox_pos_vec * 64
    pred_vox_pos_vec = np.append(pred_vox_pos_vec, 1).reshape(-1, 1)
    pred_vox_orient_vec = np.append(pred_vox_orient_vec, 1).reshape(-1, 1)
    pred_vox_orient_vec2 = np.append(pred_vox_orient_vec2, 1).reshape(-1, 1)
    affine = image.affine.cpu().numpy()
    pred_phy_pos_vec = affine @ pred_vox_pos_vec
    pred_phy_orient_vec = affine @ pred_vox_orient_vec - pred_phy_pos_vec
    pred_phy_orient_vec2 = affine @ pred_vox_orient_vec2 - pred_phy_pos_vec

    pred_phy_pos_vec = pred_phy_pos_vec.flatten()[0:3]
    pred_phy_orient_vec = pred_phy_orient_vec.flatten()[0:3]
    pred_phy_orient_vec2 = pred_phy_orient_vec2.flatten()[0:3]

    pred_phy_orient_vec2 = pred_phy_orient_vec2 / np.linalg.norm(pred_phy_orient_vec2)

    pred_phy_orient_vec = (
        pred_phy_orient_vec
        - pred_phy_orient_vec
        @ pred_phy_orient_vec2.reshape(-1, 1)
        * pred_phy_orient_vec2
    )
    pred_phy_orient_vec = pred_phy_orient_vec / np.linalg.norm(pred_phy_orient_vec)

    vy = pred_phy_orient_vec
    vz = pred_phy_orient_vec2
    vx = np.cross(vy, vz)

    img = nib.load(path)
    rotation_mat = np.eye(4)
    rotation_mat[:3, :3] = np.column_stack([vx, vy, vz]).T
    translate_mat = np.eye(4)
    translate_mat[:3, 3] = -pred_phy_pos_vec
    rigid_affine = rotation_mat @ translate_mat
    new_affine = rigid_affine @ img.affine
    new_nifti_image = nib.Nifti1Image(img.get_fdata(), affine=new_affine)

    return (
        new_nifti_image,
        (pred_phy_pos_vec**2).sum() ** 0.5,
        abs(angle_between_vectors(pred_phy_orient_vec, np.array([0, 1, 0]))),
        abs(angle_between_vectors(pred_phy_orient_vec2, np.array([0, 0, 1]))),
    )


def export_onnx(checkpoint, inshape=INSHAPE, channels=CHANNELS, strides=STRIDES):
    """Export model to onnx format

    Args:
        checkpoint (str): path to checkpoint file
        inshape (tuple[int], optional): model input shape. Defaults to INSHAPE.
        channels (tuple[int], optional): monai Regressor channels. Defaults to CHANNELS.
        strides (tuple[int], optional): monai Regressor strides. Defaults to STRIDES.
    """
    model = RegressorModel(inshape, channels, strides)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    model.to("cpu")
    with torch.no_grad():
        dummy_input = torch.randn(1, *inshape).to("cpu")
        torch.onnx.export(
            model,
            dummy_input,
            f"{Path(checkpoint).stem}.onnx",
            verbose=True,
            input_names=["input"],
            output_names=["ac", "nose", "top"],
        )

def build_data_loader(train_images:list[Path], val_images:list[Path], batch_size):
    train = [{"image": str(x.resolve())} for x in train_images]
    val = [{"image": str(x.resolve())} for x in val_images]
    train_transform = Compose([
        LoadImaged(keys=["image"], ensure_channel_first=True),
        ClipIntensityPercentilesd(keys=["image"], lower=1, upper=99),
        ScaleIntensityd(keys=["image"]),
        RandGaussianNoised(keys=["image"], prob=0.25, mean=0.35),
        RandCoarseShuffled(
            keys=["image"],
            prob=0.25,
            holes=1,
            max_holes=30,
            spatial_size=1,
            max_spatial_size=20,
        ),
        GaussianSmoothd(keys=["image"], sigma=1),
        Spacingd(keys=["image"], pixdim=(3, 3, 3), mode=("bilinear"), lazy=True),
        RandAffined(
            keys=["image"],
            prob=1,
            rotate_range=(-np.pi, np.pi),
            translate_range=10,
            scale_range=0.1,
            padding_mode="zeros",
            lazy=True,
        ),
        CropForegroundd(
            keys=["image"], source_key="image", select_fn=lambda x: x > 0.35, lazy=False
        ),
        RandAxisFlipd(keys=["image"], prob=0.5, lazy=True),
        Resized(keys=["image"], spatial_size=INSHAPE[1:], lazy=True),
        ExtractTarget(keys=["image"]),
        ApplyPendingd(keys=["image"]),
    ])
    val_transform = Compose([
        LoadImaged(keys=["image"], ensure_channel_first=True),
        ClipIntensityPercentilesd(keys=["image"], lower=1, upper=99),
        ScaleIntensityd(keys=["image"]),
        GaussianSmoothd(keys=["image"], sigma=1),
        Spacingd(keys=["image"], pixdim=(3, 3, 3), mode=("bilinear")),
        CropForegroundd(
            keys=["image"], source_key="image", select_fn=lambda x: x > 0.35
        ),
        Resized(keys=["image"], spatial_size=(64, 64, 64)),
        ExtractTarget(keys=["image"]),
    ])
    train_dataset = Dataset(data=train, transform=train_transform)
    val_dataset = Dataset(data=val, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def train(train_loader, val_loader):
    model = RegressorModel().to(DEVICE)
    max_epochs = 100
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs
    )

    min_loss = 1000
    for epoch in range(max_epochs):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            pred = model(batch["image"].to(DEVICE))
            loss = model.loss(
                pred,
                (
                    batch["image_pos"].to(DEVICE),
                    batch["image_orient"].to(DEVICE),
                    batch["image_orient2"].to(DEVICE),
                ),
            )
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()
            lr_scheduler.step()
        print(f"Epoch {epoch}, Train Loss {np.mean(train_losses)}")

        model.eval()
        losses, angle1, angle2 = [], [], []
        with torch.no_grad():
            for _, batch in enumerate(tqdm(val_loader)):
                pred = model(batch["image"].to(DEVICE))
                l1, l2, l3 = model.loss(
                    pred,
                    (
                        batch["image_pos"].to(DEVICE),
                        batch["image_orient"].to(DEVICE),
                        batch["image_orient2"].to(DEVICE),
                    ),
                    True,
                )
                losses.append(l1.item())
                angle1.append(l2.item())
                angle2.append(l3.item())
            total_loss = (
                np.mean(losses) + np.mean(angle1) + np.mean(angle2)
            )
            if total_loss < min_loss:
                min_loss = total_loss
                torch.save(model.state_dict(), f"./rigid_best.pth")
                print(f"Saved model with loss {min_loss}")
        print(
            f"Epoch {epoch}, Loss {np.mean(losses)}, {np.mean(angle1)}, {np.mean(angle2)}"
        )
        torch.save(model.state_dict(), f"./rigid_last.pth")


if __name__ == "__main__":
    export_onnx("./2head-pib-noise-gelu-64channel.pth")
