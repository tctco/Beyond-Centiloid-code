from pathlib import Path
import torch
from monai.data import MetaTensor
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
    RandGaussianNoised,
    Rand3DElasticd,
    ApplyTransformToPointsd,
    EnsureChannelFirstd,
    SqueezeDimd,
    ResampleToMatch,
    LoadImage,
    SpatialPad,
    RandGaussianSmoothd,
    SignalFillEmptyd,
    RandAdjustContrastd,
    Pad,
    ScaleIntensity,
    RandFlipd,
    ClipIntensityPercentiles,
)
import numpy as np
import nibabel as nib
from constants import RIGID_INSHAPE, ELASTIC_INSHAPE

try:
    padder = SpatialPad(ELASTIC_INSHAPE)
    loader = LoadImage(ensure_channel_first=True)
    intensity_scalar = ScaleIntensity()
    resampler = ResampleToMatch()
    template = loader("./data/template_pet.nii")
    template_mr = loader("./data/MNI152_T1_1mm.nii.gz")
    template_mr = resampler(template_mr, template)
    template_mr = intensity_scalar(template_mr)
    template_mr = padder(template_mr)
    nib.save(
        nib.Nifti1Image(template_mr[0].numpy(), template_mr.affine),
        "./data/padded_template.nii",
    )
    mask = loader("./data/mask_resampled.nii")
    mask = np.ones(mask.shape)
    mask = padder(mask)
    nib.save(nib.Nifti1Image(mask[0].numpy(), mask.affine), "./data/padded_mask.nii")
except:
    print("failed to create padded_mask.nii")

padded_template_path = Path(__file__).parent / "data" / "padded_template.nii"
padded_template_path = str(padded_template_path)


class ClipIntensityPercentilesToTarget(MapTransform):
    # used to replace ClipIntensityPercentilesd(keys=["image", "target"], lower=1, upper=99)
    def __init__(self, keys, target, lower, upper):
        super().__init__(keys)
        self.lower = lower
        self.upper = upper
        self.target = target
        self.clipper = ClipIntensityPercentiles(
            lower=lower, upper=upper, return_clipping_values=True
        )

    def __call__(self, data):
        d = dict(data)
        d[self.target] = self.clipper(d[self.target])
        vmin, vmax = d[self.target].meta["clipping_values"][0]
        for k in self.keys:
            if k == self.target:
                continue
            d[k][d[k] < vmin] = vmin
            d[k][d[k] > vmax] = vmax
        return d


class ScaleZeroOneToTargetd(MapTransform):
    # used to replace ScaleIntensityd(keys=["image", "target"])
    def __init__(self, keys, target):
        super().__init__(keys)
        self.target = target

    def __call__(self, data):
        d = dict(data)
        vmin = d[self.target].min()
        vmax = d[self.target].max()
        d[self.target] = (d[self.target] - vmin) / (vmax - vmin)
        for k in self.keys:
            if k == self.target:
                continue
            d[k] = torch.clip((d[k] - vmin) / (vmax - vmin), 0, 1)
        return d


class ResampleToMatchTemplated(MapTransform):
    def __init__(self, keys, template_path):
        super().__init__(keys)
        loader = LoadImage(ensure_channel_first=True)
        self.template = loader(template_path)
        self.resampler = ResampleToMatch(padding_mode="zeros")

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            d[k] = self.resampler(d[k], self.template)
        d["template"] = self.template
        return d


class ExtractKeypointsd(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            affine = d[k].affine
            d[f"{k}_AC"] = MetaTensor(
                ExtractKeypointsd._physical2voxel([0, 0, 0], affine), affine=affine
            )
            d[f"{k}_PA"] = MetaTensor(
                ExtractKeypointsd._physical2voxel([0, 999999, 0], affine),
                affine=affine,
            )
            d[f"{k}_IS"] = MetaTensor(
                ExtractKeypointsd._physical2voxel([0, 0, 999999], affine),
                affine=affine,
            )
        return d

    @staticmethod
    def _physical2voxel(vec, affine):
        vec = torch.tensor(vec, dtype=torch.float64)
        vec = torch.concatenate([vec, torch.tensor([1.0])]).reshape(4, 1)
        return (torch.linalg.inv(affine) @ vec).reshape(1, 4)[..., :3]


class NormalizeKeypointsd(MapTransform):
    def __init__(self, keys, input_shape):
        super().__init__(keys)
        self.input_shape = input_shape

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            d[k] = d[k] / torch.tensor(self.input_shape, dtype=d[k].dtype)
        return d


def select_fn(x):
    return x > 0.35


train_rigid_transform = Compose(
    [
        LoadImaged(keys=["image"], ensure_channel_first=True),
        SignalFillEmptyd(keys=["image"]),
        ExtractKeypointsd(keys=["image"]),
        EnsureChannelFirstd(
            keys=["image_AC", "image_PA", "image_IS"], channel_dim="no_channel"
        ),
        RandAdjustContrastd(keys=["image"]),
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
        Spacingd(
            keys=["image"],
            pixdim=(3, 3, 3),
            mode=("bilinear"),
        ),
        RandAffined(
            keys=["image"],
            prob=0.5,
            rotate_range=(-np.pi, np.pi),
            translate_range=10,
            scale_range=0.1,
            padding_mode="border",
        ),
        RandAxisFlipd(
            keys=["image"],
            prob=0.5,
        ),
        CropForegroundd(
            keys=["image"],
            source_key="image",
            select_fn=select_fn,
        ),
        Resized(keys=["image"], spatial_size=RIGID_INSHAPE[1:]),
        ApplyTransformToPointsd(
            keys=["image_AC", "image_PA", "image_IS"],
            refer_keys=["image", "image", "image"],
            dtype=torch.float32,
        ),
        SqueezeDimd(keys=["image_AC", "image_PA", "image_IS"]),
        SqueezeDimd(keys=["image_AC", "image_PA", "image_IS"]),
        NormalizeKeypointsd(keys=["image_AC"], input_shape=RIGID_INSHAPE[1:]),
    ]
)

val_rigid_transform = Compose(
    [
        LoadImaged(keys=["image"], ensure_channel_first=True),
        SignalFillEmptyd(keys=["image"]),
        ExtractKeypointsd(keys=["image"]),
        EnsureChannelFirstd(
            keys=["image_AC", "image_PA", "image_IS"], channel_dim="no_channel"
        ),
        ClipIntensityPercentilesd(keys=["image"], lower=1, upper=99),
        ScaleIntensityd(keys=["image"]),
        GaussianSmoothd(keys=["image"], sigma=1),
        Spacingd(keys=["image"], pixdim=(3, 3, 3), mode=("bilinear")),
        CropForegroundd(keys=["image"], source_key="image", select_fn=select_fn),
        Resized(keys=["image"], spatial_size=RIGID_INSHAPE[1:]),
        ApplyTransformToPointsd(
            keys=["image_AC", "image_PA", "image_IS"],
            refer_keys=["image", "image", "image"],
            dtype=torch.float32,
        ),
        SqueezeDimd(keys=["image_AC", "image_PA", "image_IS"]),
        SqueezeDimd(keys=["image_AC", "image_PA", "image_IS"]),
        NormalizeKeypointsd(keys=["image_AC"], input_shape=RIGID_INSHAPE[1:]),
    ]
)

iterated_rigid_transform = Compose(
    [
        LoadImaged(keys=["image"], ensure_channel_first=True),
        SignalFillEmptyd(keys=["image"]),
        ResampleToMatchTemplated(keys=["image"], template_path=padded_template_path),
        ClipIntensityPercentilesd(keys=["image"], lower=1, upper=99),
        ScaleIntensityd(keys=["image"]),
        GaussianSmoothd(keys=["image"], sigma=1),
        Spacingd(keys=["image"], pixdim=(3, 3, 3), mode=("bilinear")),
        CropForegroundd(
            keys=["image"], source_key="image", select_fn=lambda x: x > 0.35
        ),
        Resized(keys=["image"], spatial_size=RIGID_INSHAPE[1:]),
    ]
)


train_elastic_transform = Compose(
    [
        LoadImaged(keys=["image", "target"], ensure_channel_first=True),
        SignalFillEmptyd(keys=["image", "target"]),
        Rand3DElasticd(
            keys=["image"],
            sigma_range=(3, 8),
            magnitude_range=(30, 60),
            prob=0.5,
            rotate_range=(-np.pi / 6, np.pi / 6),
            scale_range=(-0.1, 0.1),
            translate_range=(-10, 10),
            padding_mode="zeros",
        ),
        ResampleToMatchTemplated(
            keys=["image", "target"], template_path=padded_template_path
        ),
        RandFlipd(keys=["image", "target"], prob=0.5, spatial_axis=0),
        ClipIntensityPercentilesToTarget(
            keys=["image"], target="target", lower=1, upper=99
        ),
        ScaleZeroOneToTargetd(keys=["image"], target="target"),
        RandGaussianSmoothd(
            keys=["image"],
            prob=0.1,
        ),
    ]
)

val_elastic_transform = Compose(
    [
        LoadImaged(keys=["image", "target"], ensure_channel_first=True),
        SignalFillEmptyd(keys=["image", "target"]),
        ResampleToMatchTemplated(
            keys=["image", "target"], template_path=padded_template_path
        ),
        ClipIntensityPercentilesToTarget(
            keys=["image"], target="target", lower=1, upper=99
        ),
        ScaleZeroOneToTargetd(keys=["image"], target="target"),
    ]
)

test_elastic_transform = Compose(
    [
        LoadImaged(keys=["image", "raw_image"], ensure_channel_first=True),
        SignalFillEmptyd(keys=["image", "raw_image"]),
        ResampleToMatchTemplated(
            keys=["image", "raw_image"], template_path=padded_template_path
        ),
        ClipIntensityPercentilesd(keys=["image"], lower=1, upper=99),
        ScaleIntensityd(keys=["image"]),
    ]
)
