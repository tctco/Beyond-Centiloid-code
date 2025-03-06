from pathlib import Path
from itertools import product
from functools import partial
import os
from collections import defaultdict
from typing import Union
import pandas as pd
import nibabel as nib
from nilearn.datasets import fetch_atlas_aal
from nilearn.maskers import NiftiLabelsMasker
from nilearn.masking import apply_mask
from nilearn.image import resample_to_img
from utils import angle_between, voxel2world, voxelvec2worldvec, euclidean_distance
import numpy as np

NiftyImage = Union[nib.Nifti1Image, nib.filebasedimages.FileBasedImage]


class RigidEvaluator:
    def __init__(self) -> None:
        self.diff_AC = []
        self.diff_PA = []
        self.diff_IS = []

    def collect(self, pred_AC, pred_PA, pred_IS, gt_AC, gt_PA, gt_IS, affine):
        pred_AC = pred_AC.cpu().numpy()
        pred_PA = pred_PA.cpu().numpy()
        pred_IS = pred_IS.cpu().numpy()
        gt_AC = gt_AC.cpu().numpy()
        gt_PA = gt_PA.cpu().numpy()
        gt_IS = gt_IS.cpu().numpy()

        diff_AC = [
            euclidean_distance(voxel2world(pred, affine), voxel2world(gt, affine))
            for pred, gt in zip(pred_AC, gt_AC)
        ]
        diff_PA = [
            angle_between(
                voxelvec2worldvec(pred, affine), voxelvec2worldvec(gt, affine)
            )
            for pred, gt in zip(pred_PA, gt_PA)
        ]
        diff_IS = [
            angle_between(
                voxelvec2worldvec(pred, affine), voxelvec2worldvec(gt, affine)
            )
            for pred, gt in zip(pred_IS, gt_IS)
        ]

        self.diff_AC.extend(diff_AC)
        self.diff_PA.extend(diff_PA)
        self.diff_IS.extend(diff_IS)

    def get_results(self, printout=True) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "AC": self.diff_AC,
                "PA": self.diff_PA,
                "IS": self.diff_IS,
            }
        )
        if printout:
            print(df.describe())
        return df

    def clean(self):
        self.diff_AC = []
        self.diff_PA = []
        self.diff_IS = []


def validate_input_img(img):
    if isinstance(img, Path) or isinstance(img, str):
        img = nib.load(img)
    elif isinstance(img, NiftyImage):
        pass
    else:
        raise ValueError(f"img must be a path or a NiftyImage, got {type(img)}")
    return img


class CenTauRCalculator:
    def __init__(self):
        centaur_masks = (
            "universal,mesial temporal,meta temporal,temporo parietal,frontal".split(
                ","
            )
        )
        centaur_tracers = "ro948,ftp,mk6240,gtp1,pm-pbb3,pi2620".split(",")
        tracer_synonyms = {
            "ftp": ["flortaucipir", "av1451"],
            "pm-pbb3": ["florzolotau", "apn"],
        }
        coefs = [
            (13.05, -15.57),
            (13.63, -15.85),
            (10.08, -10.06),
            (10.67, -11.92),
            (16.73, -15.34),
            (8.45, -9.61),
            (11.76, -13.08),
            (10.42, -12.11),
            (7.28, -7.01),
            (7.88, -8.75),
            (7.97, -7.83),
            (6.03, -6.83),
            (13.16, -16.19),
            (12.95, -15.37),
            (9.36, -10.6),
            (9.60, -11.10),
            (11.78, -11.21),
            (7.78, -9.33),
            (13.05, -15.62),
            (13.75, -15.92),
            (9.98, -10.15),
            (10.84, -12.27),
            (16.16, -14.68),
            (8.21, -9.52),
            (12.61, -13.45),
            (11.61, -13.01),
            (10.05, -8.91),
            (9.41, -9.71),
            (15.7, -13.18),
            (9.07, -9.01),
        ]
        self.suvr2centaur = {}
        for i, (mask, tracer) in enumerate(product(centaur_masks, centaur_tracers)):
            self.suvr2centaur[(mask, tracer)] = partial(
                lambda suvr, i=i: coefs[i][0] * suvr + coefs[i][1]
            )
            synonyms = tracer_synonyms.get(tracer, [])
            for syn in synonyms:
                self.suvr2centaur[(mask, syn)] = self.suvr2centaur[(mask, tracer)]

        self.masks = {}
        self.masks["universal"] = nib.load("./templates/centaur/CenTauR.nii")
        self.masks["mesial temporal"] = nib.load(
            "./templates/centaur/Mesial_CenTauR.nii"
        )
        self.masks["meta temporal"] = nib.load("./templates/centaur/Meta_CenTauR.nii")
        self.masks["temporo parietal"] = nib.load("./templates/centaur/TP_CenTauR.nii")
        self.masks["frontal"] = nib.load("./templates/centaur/Frontal_CenTauR.nii")
        self.ref_mask = nib.load("./templates/centaur/voi_CerebGry_tau_2mm.nii")

    def print_available(self):
        print("Available mask-tracer pairs: ", self.suvr2centaur.keys())

    def calculate(
        self, img: Union[os.PathLike, NiftyImage], tracer: str, mask: str = "universal"
    ):
        mask = mask.lower()
        tracer = tracer.lower()
        if (mask, tracer) not in self.suvr2centaur:
            raise ValueError(f"Mask {mask} and tracer {tracer} not found")
        suvr = self.SUVr(img, mask)
        return self.suvr2centaur[(mask, tracer)](suvr)

    def SUVr(self, img: NiftyImage, mask="universal"):
        img = validate_input_img(img)
        img = resample_to_img(img, self.ref_mask)
        mask = mask.lower()
        if mask not in self.masks:
            raise ValueError(f"Mask {mask} not found")
        return (
            apply_mask(img, self.masks[mask]).mean()
            / apply_mask(img, self.ref_mask).mean()
        )

    def ref_SUV(self, img: NiftyImage):
        img = validate_input_img(img)
        return apply_mask(img, self.ref_mask).mean()


class CentiloidCalculator:
    def __init__(self):
        self.suvr2centiloid = {
            **dict.fromkeys(["fmm", "flutemetamol"], lambda suvr: 121.4 * suvr - 121.2),
            **dict.fromkeys(
                ["fbp", "florbetapir", "av45"],
                lambda suvr: 175.4 * suvr - 182.3,
            ),
            **dict.fromkeys(["fbb", "florbetaben"], lambda suvr: 153.4 * suvr - 154.9),
            **dict.fromkeys(["pib"], lambda suvr: 93.7 * suvr - 94.6),
            **dict.fromkeys(["nav4694"], lambda suvr: 100 * (suvr - 1.028) / 1.174),
        }

        self.voi_mask = nib.load("./templates/centiloid/voi_ctx_2mm.nii")
        self.ref_masks = {
            "whole cerebellum": nib.load("./templates/centiloid/voi_WhlCbl_2mm.nii"),
            "cerebellar gray": nib.load("./templates/centiloid/voi_CerebGry_2mm.nii"),
            "pons": nib.load("./templates/centiloid/voi_Pons_2mm.nii"),
            "cerebellum brainstem": nib.load(
                "./templates/centiloid/voi_WhlCblBrnStm_2mm.nii"
            ),
        }

    def print_available(self):
        print("Available tracers: ", self.suvr2centiloid.keys())
        print("Available reference masks: ", self.ref_masks.keys())

    def calculate(self, img: NiftyImage, tracer: str):
        tracer = tracer.lower()
        if tracer not in self.suvr2centiloid:
            raise ValueError(f"Tracer {tracer} not found")
        suvr = self.SUVr(img)
        return self.suvr2centiloid[tracer](suvr)

    def SUVr(self, img: NiftyImage, ref: str = "whole cerebellum"):
        img = validate_input_img(img)
        img = resample_to_img(img, self.voi_mask)
        ref = ref.lower()
        if ref not in self.ref_masks:
            raise ValueError(f"Reference mask {ref} not found")
        return (
            apply_mask(img, self.voi_mask).mean()
            / apply_mask(img, self.ref_masks[ref]).mean()
        ).astype(float)

    def ref_SUV(self, img: NiftyImage, mask: str = "whole cerebellum"):
        img = validate_input_img(img)
        img = resample_to_img(img, self.ref_masks[mask])
        mask = mask.lower()
        if mask not in self.ref_masks:
            raise ValueError(f"Reference mask {mask} not found")
        return apply_mask(img, self.ref_masks[mask]).mean()


class BrainRegionSUVCalculator:
    def __init__(self):
        aal = fetch_atlas_aal()
        self.aal = aal
        self.masker = NiftiLabelsMasker(
            labels_img=aal.maps, labels=aal.labels, verbose=0
        )

    def calculate(self, imgs: list[Union[os.PathLike, NiftyImage]]) -> pd.DataFrame:
        imgs = [validate_input_img(x) for x in imgs]
        suv = self.masker.fit_transform(imgs)
        columns = self.masker.labels
        result = pd.DataFrame(suv, columns=columns)
        return result

    def calculate_compound_region(self, imgs, brain_regions):
        imgs = [validate_input_img(x) for x in imgs]
        mask = np.zeros_liks(self.aal.template.get_fdata())
        aal_template = nib.load(self.aal.maps)
        for region in brain_regions:
            index = int(self.aal.indices[self.aal.labels.index(region)])
            mask[aal_template.get_fdata() == index] = 1
        mask = nib.Nifti1Image(mask, aal_template.affine)
        suvs = apply_mask(imgs, mask).mean()
        return suvs


class ElasticEvaluator:
    def __init__(self) -> None:
        aal = fetch_atlas_aal()
        self.masker = NiftiLabelsMasker(
            labels_img=aal.maps, labels=aal.labels, verbose=0
        )
        self.brain_regions = defaultdict(list)

    def collect(self, preds, gts):
        preds = preds.cpu()
        gts = gts.cpu()
        for pred, gt in zip(preds, gts):
            affine = gt.affine.numpy()
            pred = pred.numpy()
            gt = gt.numpy()
            pred = nib.Nifti1Image(pred[0], affine)  # channel first problem
            gt = nib.Nifti1Image(gt[0], affine)
            pred = self.masker.fit_transform(pred).flatten()
            gt = self.masker.fit_transform(gt).flatten()
            for pred_suv, gt_suv, label in zip(pred, gt, self.masker.labels):
                self.brain_regions[f"{label}_pred"].append(pred_suv)
                self.brain_regions[f"{label}_gt"].append(gt_suv)
                self.brain_regions[f"{label}_diff"].append(
                    abs(pred_suv - gt_suv) / gt_suv
                )

    def get_results(self, printout=True) -> pd.DataFrame:
        df = pd.DataFrame(self.brain_regions)
        if printout:
            print(df.describe())
        return df

    def clean(self):
        self.brain_regions = defaultdict(list)
