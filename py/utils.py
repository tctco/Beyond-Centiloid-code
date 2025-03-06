from typing import Union
import os
from pathlib import Path
from typing import Optional
import torch
from tqdm import tqdm
from nilearn.image import resample_to_img, reorder_img
import nibabel as nib
import numpy as np
from monai.data import Dataset, DataLoader
from nets import AffineVoxelMorphONNX, AffineVoxelMorph
from transforms import (
    val_rigid_transform,
    test_elastic_transform,
    iterated_rigid_transform,
)
from constants import DEVICE, RIGID_INSHAPE

PACKAGE_DIR = Path(__file__).parent


def get_tracer_name(img_path):
    img_path = str(img_path)
    if "av45" in img_path.lower():
        return "AV45"
    elif "pib" in img_path.lower():
        return "PIB"
    elif "av1451" in img_path.lower():
        return "AV1451"
    elif "ftp" in img_path.lower():
        return "AV1451"
    elif "fbb" in img_path.lower():
        return "FBB"
    elif "apn" in img_path.lower():
        return "APN"
    elif "nav" in img_path.lower():
        return "NAV4694"
    elif "fbb" in img_path.lower():
        return "FBB"
    elif "fmm" in img_path.lower():
        return "FMM"
    elif "fmt" in img_path.lower():
        return "FMT"
    else:
        raise ValueError(f"Unknown tracer for {img_path}")


def get_modality(img_path):
    tracer = get_tracer_name(img_path)
    if tracer in ["AV45", "PIB", "FBB", "FMM", "NAV4694"]:
        return "abeta"
    elif tracer in ["AV1451", "FTP", "APN"]:
        return "tau"
    elif tracer in ["FMT"]:
        return "cancer"
    else:
        raise ValueError(f"Unknown modality for {img_path}")


def world2voxel(point, affine):
    return np.linalg.inv(affine).dot([*point, 1])[:3]


def voxel2world(point, affine):
    return affine.dot([*point, 1])[:3]


def voxelvec2worldvec(vec, affine):
    return voxel2world(vec, affine) - voxel2world([0, 0, 0], affine)


def worldvec2voxelvec(vec, affine):
    return world2voxel(vec, affine) - world2voxel([0, 0, 0], affine)


def normalize(point):
    norm = np.linalg.norm(point)
    if norm == 0:
        return point
    return point / norm


def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def build_affine_matrix(AC, PA, IS, affine):
    # all in world coordinates
    AC = np.array(AC)
    PA = np.array(PA)
    IS = np.array(IS)

    PA = normalize(PA)
    IS = IS - np.dot(IS, PA) * PA
    IS = normalize(IS)

    LR = np.cross(PA, IS)
    R = np.eye(4)
    R[:3, :3] = np.column_stack((LR, PA, IS)).T
    T = np.eye(4)
    T[:3, 3] = -AC
    A = R @ T @ affine
    return A


def angle_between(v1, v2):
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def dl_rigid_transform(
    paths, model, batch_size, device=DEVICE, transform=val_rigid_transform
) -> list[nib.Nifti1Image]:
    model.eval()
    model = model.to(device)
    affine_matrics: list[np.ndarray] = [
        nib.load(path, mmap=False).affine for path in paths
    ]
    transformed_affine_matrics: list[np.ndarray] = []
    dataset = [{"image": str(path)} for path in paths]
    dataset = Dataset(dataset, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    pred_voxel_AC, pred_voxel_PA, pred_voxel_IS = [], [], []
    for batch in loader:
        images = batch["image"]
        transformed_affine_matrics.extend(
            [img.affine.numpy() for img in batch["image"]]
        )
        images = images.to(device)
        with torch.no_grad():
            pred_AC, pred_PA, pred_IS = model(images)
        pred_AC = pred_AC * RIGID_INSHAPE[1]  # TODO: should be points * INSHAPE
        pred_PA = pred_PA * 99999
        pred_IS = pred_IS * 99999
        pred_AC = pred_AC.cpu().numpy().tolist()
        pred_PA = pred_PA.cpu().numpy().tolist()
        pred_IS = pred_IS.cpu().numpy().tolist()
        pred_voxel_AC.extend(pred_AC)
        pred_voxel_PA.extend(pred_PA)
        pred_voxel_IS.extend(pred_IS)
    pred_world_AC = [
        voxel2world(point, affine)
        for point, affine in zip(pred_voxel_AC, transformed_affine_matrics)
    ]
    pred_world_PA = [
        voxelvec2worldvec(point, affine)
        for point, affine in zip(pred_voxel_PA, transformed_affine_matrics)
    ]
    pred_world_IS = [
        voxelvec2worldvec(point, affine)
        for point, affine in zip(pred_voxel_IS, transformed_affine_matrics)
    ]

    corrected_affines = [
        build_affine_matrix(AC, PA, IS, affine)
        for AC, PA, IS, affine in zip(
            pred_world_AC, pred_world_PA, pred_world_IS, affine_matrics
        )
    ]
    results = []
    for path, affine in zip(paths, corrected_affines):
        nii = nib.load(path, mmap=False)
        new_nii = nib.Nifti1Image(nii.get_fdata(), affine, nii.header)
        results.append(new_nii)
    return results


def iteratively_rigid_normalize(
    path: Path,
    save_path: Path,
    model,
    iter=5,
    ac_diff_threshold_mm=5,
    device=DEVICE,
    verbose=False,
):
    result = dl_rigid_transform([path], model, 1, device)[0]
    name = save_path.name
    nib.save(result, save_path)
    last_ac_world = result.affine[:3, 3]
    if verbose:
        iter = tqdm(range(iter))
    else:
        iter = range(iter)
    for i in iter:
        result = dl_rigid_transform(
            [save_path], model, 1, device, iterated_rigid_transform
        )[0]
        ac_world = result.affine[:3, 3]
        dist = euclidean_distance(last_ac_world, ac_world)
        if verbose:
            print(f"AC dist diff in last 2 iteration: {dist}")
        if dist < ac_diff_threshold_mm:
            break
        last_ac_world = ac_world
        if verbose:
            save_path = save_path.with_name(f"iter_{i+1}_{name}")
            nib.save(result, save_path)
        else:
            nib.save(result, save_path)
    return result


def dl_elastic_transform(
    paths,
    model: Union[AffineVoxelMorphONNX, AffineVoxelMorph],
    batch_size,
    device=DEVICE,
) -> list[nib.Nifti1Image]:
    model.eval()
    model = model.to(device)
    dataset = [{"image": str(path), "raw_image": str(path)} for path in paths]
    dataset = Dataset(dataset, test_elastic_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    template_affine = nib.load(PACKAGE_DIR / "data" / "padded_template.nii").affine
    warped_target = nib.load(PACKAGE_DIR / "data" / "wMAC.nii")
    affine_matrix = warped_target.affine
    warped_images: list[np.ndarray] = []
    for batch in loader:
        moving_images = batch["image"]
        raw_images = batch["raw_image"]
        template_images = batch["template"]
        moving_images = moving_images.to(device)
        raw_images = raw_images.to(device)
        template_images = template_images.to(device)
        with torch.no_grad():
            if isinstance(model, AffineVoxelMorphONNX):
                warped = model(moving_images, template_images, raw_images)
            elif isinstance(model, AffineVoxelMorph):
                _, _, warped, _ = model(moving_images, template_images)
            else:
                raise ValueError(f"Unsupported model: {type(model)}")
        elastic_transformed = warped.cpu().numpy()
        warped_images.extend(
            # [x.squeeze()[8.5:-8.5, 17:-16, 9:-8] for x in elastic_transformed]
            [x.squeeze() for x in elastic_transformed]
        )
    results = []
    translate_mat = np.array(
        [[1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, -1], [0, 0, 0, 1]]
    )  # 向右 向后 向下移动1个单位
    for img in warped_images:
        # nii = nib.nifti1.Nifti1Image(img, affine_matrix)
        nii = nib.nifti1.Nifti1Image(img, translate_mat @ template_affine)
        nii = reorder_img(nii)
        nii = resample_to_img(nii, warped_target)
        results.append(nii)
    return results


class DeepCascadeSpatialNormalizer:
    def __init__(self, rigid_model, elastic_model, batch_size, device=DEVICE):
        self.rigid_model = rigid_model
        self.elastic_model = elastic_model
        self.batch_size = batch_size
        self.device = device

    def normalize(
        self,
        input_paths: list[Path],
        output_paths: list[Path],
        rigid_paths: Optional[list[Path]] = None,
        enable_iter=False,
        verbose=False,
    ):
        if rigid_paths is None:
            rigid_paths = output_paths

        if self.batch_size == 1:
            if verbose:
                zipped = tqdm(zip(input_paths, rigid_paths, output_paths))
            else:
                zipped = zip(input_paths, rigid_paths, output_paths)
            for input_path, rigid_path, output_path in zipped:
                if enable_iter:
                    result = iteratively_rigid_normalize(
                        input_path,
                        rigid_path,
                        self.rigid_model,
                        device=self.device,
                        verbose=False,
                    )
                else:
                    result = dl_rigid_transform(
                        [input_path], self.rigid_model, 1, self.device
                    )[0]
                    nib.save(result, rigid_path)
                result = dl_elastic_transform(
                    [rigid_path], self.elastic_model, 1, self.device
                )[0]
                nib.save(result, output_path)
        else:
            if enable_iter:
                for input_path, output_path in zip(input_paths, rigid_paths):
                    _ = iteratively_rigid_normalize(
                        input_path,
                        output_path,
                        self.rigid_model,
                        device=self.device,
                        verbose=False,
                    )
            else:
                rigid_results = dl_rigid_transform(
                    input_paths, self.rigid_model, self.batch_size, self.device
                )
                for path, result in zip(rigid_paths, rigid_results):
                    nib.save(result, path)
            elastic_results = dl_elastic_transform(
                output_paths, self.elastic_model, self.batch_size, self.device
            )
            for path, result in zip(output_paths, elastic_results):
                nib.save(result, path)
