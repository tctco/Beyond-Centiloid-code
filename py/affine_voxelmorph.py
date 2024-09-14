from functools import reduce
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm

import torch
from torch.nn import MSELoss
import torch.nn as nn

from monai.networks.nets import VoxelMorph, Regressor
from monai.networks.nets.regunet import AffineHead
from monai.data import Dataset, DataLoader
from monai.losses import BendingEnergyLoss
from monai.metrics import MSEMetric
from monai.networks.blocks import Warp
from monai.transforms import (
    LoadImaged,
    ResampleToMatchd,
    Compose,
    RandAffined,
    ScaleIntensityd,
    ClipIntensityPercentilesd,
    SpatialPadd,
    RandGaussianSmoothd,
    MapTransform,
    SpatialPad,
    LoadImage,
    ResampleToMatch,
)
import nibabel as nib
from nilearn.image import new_img_like

INPUT_SHAPE = (79 + 17, 95 + 33, 79 + 17)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 100
LAM_SIM = 1
LAM_SMOOTH = 0.1
abeta_template = Path("./template_pet.nii")
brain_mask = Path("./mask_resampled.nii")


class ReplaceNand(MapTransform):
    """
    Transform to replace NaN values with a specified value, default is 0.
    """

    def __init__(self, keys, value=0):
        super().__init__(keys)
        self.value = value

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = np.nan_to_num(d[key], nan=self.value)
        return d


class AffineVoxelMorph(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        inshape = (1, *INPUT_SHAPE)
        outshape = (3, 4)
        channels = (16, 32, 64, 128, 256)
        strides = (2, 2, 2, 2, 2)

        spatial_dims = 3
        image_size = inshape[1:]
        decode_size = (
            np.array(image_size) // reduce(lambda x, y: x * y, strides)
        ).tolist()
        in_channels = channels[-1]

        self.conv = Regressor(inshape, outshape, channels, strides).net
        self.affine_head = AffineHead(
            spatial_dims, image_size, decode_size, in_channels
        )
        self.vm = VoxelMorph()
        self.warp = Warp(mode="bilinear", padding_mode="zeros")

    def forward(self, moving_image, fixed_image):
        features = self.conv(moving_image)
        affine = self.affine_head([features], None)
        affine_transformed = self.warp(moving_image, affine)
        warped_image, ddf_image = self.vm(affine_transformed, fixed_image)
        return affine_transformed, warped_image, ddf_image, affine


def build_dataloader(
    train_original_images: list[Path],
    train_spm_images: list[Path],
    val_original_images: list[Path],
    val_spm_images: list[Path],
    template_path: Path,
    mask_path: Path,
    batch_size: int,
):
    train = [
        {
            "moving": str(m.resolve()),
            "fixed": str(f.resolve()),
            "template": str(template_path),
            "mask": str(mask_path),
        }
        for m, f in zip(train_original_images, train_spm_images)
    ]
    val = [
        {
            "moving": str(m.resolve()),
            "fixed": str(f.resolve()),
            "template": str(template_path),
            "mask": str(mask_path),
        }
        for m, f in zip(val_original_images, val_spm_images)
    ]

    val_pipeline = [
        LoadImaged(
            keys=["moving", "fixed", "template", "mask"], ensure_channel_first=True
        ),
        SpatialPadd(keys=["fixed", "template", "mask"], spatial_size=INPUT_SHAPE),
        ResampleToMatchd(keys=["moving"], key_dst="fixed", padding_mode="zeros"),
        ReplaceNand(keys=["moving", "fixed"]),
        ClipIntensityPercentilesd(keys=["moving", "fixed"], lower=1, upper=99),
        ScaleIntensityd(keys=["moving", "fixed", "template"]),
    ]

    train_pipeline = [
        LoadImaged(
            keys=["moving", "fixed", "template", "mask"], ensure_channel_first=True
        ),
        SpatialPadd(keys=["fixed", "template", "mask"], spatial_size=INPUT_SHAPE),
        RandAffined(
            keys=["moving"],
            prob=0.5,
            translate_range=10,
            rotate_range=np.pi / 6,
            scale_range=0.1,
            padding_mode="zeros",
        ),
        ResampleToMatchd(keys=["moving"], key_dst="fixed", padding_mode="zeros"),
        ReplaceNand(keys=["moving", "fixed"]),
        ClipIntensityPercentilesd(keys=["moving", "fixed"], lower=1, upper=99),
        ScaleIntensityd(keys=["moving", "fixed", "template"]),
        RandGaussianSmoothd(
            keys=["moving"],
            prob=0.25,
        ),
    ]

    transform_train = Compose(train_pipeline)
    transform_val = Compose(val_pipeline)
    train_dataset = Dataset(data=train, transform=transform_train)
    val_dataset = Dataset(data=val, transform=transform_val)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    return train_loader, val_loader


def loss_fun(
    fixed_image,
    pred_image,
    affined_image,
    ddf_image,
    lam_sim,
    lam_smooth,
    mask,
):
    mse_loss = MSELoss()
    regularization = BendingEnergyLoss()
    sim2 = mse_loss(affined_image * mask, fixed_image * mask)
    sim = mse_loss(pred_image * mask, fixed_image * mask)
    smooth = regularization(ddf_image) if lam_smooth > 0 else 0

    return lam_sim * sim + lam_smooth * smooth + lam_sim * sim2


def train_affine_voxel_morph(
    train_original_images,
    train_spm_images,
    val_original_images,
    val_spm_images,
    template_path,
    mask_path,
    batch_size,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    max_epochs=MAX_EPOCHS,
    lam_sim=LAM_SIM,
    lam_smooth=LAM_SMOOTH,
    device=DEVICE,
):
    train_loader, val_loader = build_dataloader(
        train_original_images,
        train_spm_images,
        val_original_images,
        val_spm_images,
        template_path,
        mask_path,
        batch_size,
    )

    model = AffineVoxelMorph().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs
    )

    mse_metric_before = MSEMetric()
    mse_metric_after = MSEMetric()
    scaler = torch.cuda.amp.GradScaler()

    best_eval_mse = float("inf")

    for epoch in range(max_epochs):
        model.train()
        epoch_loss, n_steps = 0, 0
        epoch_start = time.time()
        for batch_data in tqdm(train_loader):
            n_steps += 1
            fixed_image = batch_data["fixed"].to(device)
            moving_image = batch_data["moving"].to(device)
            template_image = batch_data["template"].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                affined_image, pred_image, ddf_image, _ = model.forward(
                    template_image, moving_image
                )
                mask = batch_data["mask"].to(device)
                mask = mask + 0.25
                loss = loss_fun(
                    fixed_image,
                    pred_image,
                    affined_image,
                    ddf_image,
                    lam_sim,
                    lam_smooth,
                    mask,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        lr_scheduler.step()
        epoch_loss /= n_steps
        print(
            f"{epoch + 1} | loss = {epoch_loss:.6f} "
            f"elapsed time: {time.time()-epoch_start:.2f} sec. "
            f"lr: {lr_scheduler.get_last_lr()[0]:.6f}"
        )

        # EVALUATION
        model.eval()
        with torch.no_grad():
            for batch_data in tqdm(val_loader):
                fixed_image = batch_data["fixed"].to(device)
                moving_image = batch_data["moving"].to(device)
                template_image = batch_data["template"].to(device)
                with torch.cuda.amp.autocast(enabled=True):
                    affined_image, pred_image, ddf_image, _ = model.forward(
                        template_image, moving_image
                    )

                mask = batch_data["mask"].to(device)
                mse_metric_before(y_pred=moving_image * mask, y=fixed_image * mask)
                mse_metric_after(y_pred=pred_image * mask, y=fixed_image * mask)

        mse_before = mse_metric_before.aggregate().item()
        mse_metric_before.reset()
        mse_after = mse_metric_after.aggregate().item()
        mse_metric_after.reset()
        print(
            f"Val {epoch + 1} | mse_before = {mse_before:.5f}, mse_after = {mse_after:.5f}"
        )

        if mse_after < best_eval_mse:
            best_eval_mse = mse_after
            torch.save(model.state_dict(), "./best_affine_voxelmorph.pth")
            print(f"{epoch + 1} | Saving best mse model: {best_eval_mse}")

        torch.save(model.state_dict(), "./last_affine_voxelmorph.pth")


def calc_ddf(
    moving_image: str,
    template: str,
    model: AffineVoxelMorph,
    device=DEVICE,
):
    transform = Compose(
        [
            LoadImaged(keys=["moving", "fixed"], ensure_channel_first=True),
            SpatialPadd(keys=["fixed"], spatial_size=INPUT_SHAPE),
            ResampleToMatchd(keys=["moving"], key_dst="fixed", padding_mode="zeros"),
            ReplaceNand(keys=["moving", "fixed"]),
            ClipIntensityPercentilesd(keys=["moving", "fixed"], lower=1, upper=99),
            ScaleIntensityd(keys=["moving", "fixed"]),
        ]
    )
    tmp_dataset = Dataset(
        [{"moving": moving_image, "fixed": template}], transform=transform
    )
    with torch.no_grad():
        affined_img, pred_img, ddf, affine_ddf = model(
            fixed_image=tmp_dataset[0]["fixed"][None, ...].to(device),
            moving_image=tmp_dataset[0]["moving"][None, ...].to(device),
        )
    return ddf, affine_ddf


def warp(moving_image: str, template: str, model):
    ddf, affine_ddf = calc_ddf(moving_image, template, model)
    ddf = ddf.cpu()[0]
    affine_ddf = affine_ddf.cpu()[0]
    img = LoadImage(ensure_channel_first=True)(moving_image)
    template_img = LoadImage(ensure_channel_first=True)(template)
    padder = SpatialPad(spatial_size=INPUT_SHAPE)
    template_img = padder(template_img)
    resampler = ResampleToMatch(padding_mode="zeros")
    img = resampler(img, template_img)
    img = Warp()(img[None, ...], affine_ddf[None, ...])
    img = Warp()(img, ddf[None, ...]).squeeze()
    img = img[8:-9, 16:-17, 8:-9]
    template_nii = nib.load(template)
    warped_image = new_img_like(template_nii, img.numpy())
    return warped_image


def export_onnx(checkpoint_path, output_path):
    class AffineVoxelMorph_ONNX(AffineVoxelMorph):
        def forward(self, moving_image, fixed_image, original_image):
            features = self.conv(moving_image)
            affine = self.affine_head([features], None)
            affine_transformed = self.warp(moving_image, affine)
            _, ddf_image = self.vm(affine_transformed, fixed_image)
            warped = self.warp(self.warp(original_image, affine), ddf_image)
            return warped

    model = AffineVoxelMorph_ONNX()
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    dummy_input = (
        torch.randn(1, 1, *INPUT_SHAPE),
        torch.randn(1, 1, *INPUT_SHAPE),
        torch.randn(1, 1, *INPUT_SHAPE),
    )
    with torch.no_grad():
        ddf = model(*dummy_input)
        print(ddf.shape)
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            verbose=False,
            opset_version=20,
            input_names=["input", "template", "input_raw"],
            output_names=["warped"],
        )
