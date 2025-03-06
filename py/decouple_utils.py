from pathlib import Path
import os
import warnings
from nilearn.image import resample_to_img, reorder_img
from nilearn.plotting import plot_stat_map
import nibabel as nib
from captum.attr import GuidedGradCam
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from utils import DeepCascadeSpatialNormalizer
from nets import AffineVoxelMorphONNX, RegressorModel
from evaluator import CentiloidCalculator, CenTauRCalculator
from utils import DeepCascadeSpatialNormalizer, get_tracer_name
from plot import register_alpha_colormap
from constants import *

mask = nib.load("./data/padded_mask.nii")
mask = torch.from_numpy(mask.get_fdata()).reshape((1, 1, *mask.shape))
rigid_model = RegressorModel(RIGID_INSHAPE)
rigid_model.load_state_dict(torch.load("./models/best_RegressorModel.pth"))
rigid_model.eval()
affine_model = AffineVoxelMorphONNX(mask)
affine_model.load_state_dict(torch.load("./models/best_AffineVoxelMorph.pth"))
affine_model.eval()

CMAP_NAME = register_alpha_colormap("turbo")
VOXEL_VOLUME = (1.5 / 10) ** 3  # cm^3
TARGET_CLASS = 0
normalizer = DeepCascadeSpatialNormalizer(rigid_model, affine_model, 1)
cl_calc = CentiloidCalculator()
ctr_calc = CenTauRCalculator()


class ADOnlyDiscriminator(torch.nn.Module):
    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, x):
        output = self.discriminator.model.net(x)
        output = self.discriminator.model.final[:2](output)

        return output


class DecoupledImage:
    def __init__(
        self,
        original_img,
        stripped_img,
        prob_map,
        ad_component,
        path,
        label=None,
        pred_AD_prob=None,
        pred_fake_prob=None,
        pred_striped_AD_prob=None,
        pred_striped_fake_prob=None,
        grad_cam=None,
        cmap_name=CMAP_NAME,
    ):
        self.original_img = original_img
        self.stripped_img = stripped_img
        self.prob_map = prob_map
        self.ad_component = ad_component
        self.label: int = label  # 0 for CN, 1 for AD
        self.pred_AD_prob = pred_AD_prob
        self.pred_fake_prob = pred_fake_prob
        self.path = Path(path) if path else None
        self.pred_striped_AD_prob = pred_striped_AD_prob
        self.pred_striped_fake_prob = pred_striped_fake_prob
        self.grad_cam = grad_cam
        self.vmax = original_img.get_fdata().max()
        self.cmap_name = cmap_name

    def summary(self, join_char=" ", prefix=None, suffix=None):
        title = []
        if prefix:
            title.append(prefix)
        if self.path:
            title.append(f"Path: {self.path.stem}")
        if self.label:
            title.append(f"Diagnosis: {self.label}")
        if self.pred_AD_prob:
            title.append(f"AD prob: {self.pred_AD_prob:.2f}")
        if self.pred_fake_prob:
            title.append(f"Fake prob: {self.pred_fake_prob:.2f}")
        if self.pred_striped_AD_prob:
            title.append(f"Stripped AD prob: {self.pred_striped_AD_prob:.2f}")
        if self.pred_striped_fake_prob:
            title.append(f"Stripped Fake prob: {self.pred_striped_fake_prob:.2f}")
        title.append(f"ADAD Score: {self.sum_AD_component():.2f}")
        if suffix:
            title.append(suffix)
        title = join_char.join(title)
        # auto wrap based on word count
        if len(title) > 30:
            title = title.split()
            title = (
                " ".join(title[: len(title) // 2])
                + "\n"
                + " ".join(title[len(title) // 2 :])
            )
        return title

    def sum_AD_component(self):
        return self.ad_component.get_fdata().sum() * VOXEL_VOLUME

    def _save_display(self, display, out):
        if out:
            display.savefig(out, dpi=500)

    def plot_prob_map(self, annotate=False, display_title=True, out=None, **kwargs):
        title = self.summary(prefix="Prob Map") if display_title else None
        display = plot_stat_map(
            self.prob_map,
            bg_img=self.original_img,
            cmap=self.cmap_name,
            vmin=0,
            vmax=1,
            annotate=annotate,
            title=title,
            draw_cross=False,
            **kwargs,
        )
        self._save_display(display, out)

    def plot_ad_component(self, annotate=False, display_title=True, out=None, **kwargs):
        title = self.summary(prefix="AD Component") if display_title else None
        display = plot_stat_map(
            self.ad_component,
            bg_img=self.original_img,
            cmap=self.cmap_name,
            vmin=0,
            annotate=annotate,
            title=title,
            vmax=self.vmax,
            draw_cross=False,
            **kwargs,
        )
        self._save_display(display, out)

    def plot_stripped_img(self, annotate=False, display_title=True, out=None, **kwargs):
        title = self.summary(prefix="Stripped") if display_title else None
        display = plot_stat_map(
            self.stripped_img,
            bg_img=self.original_img,
            cmap=self.cmap_name,
            annotate=annotate,
            title=title,
            vmax=self.vmax,
            vmin=0,
            draw_cross=False,
            **kwargs,
        )
        self._save_display(display, out)

    def plot_original_img(self, annotate=False, display_title=True, out=None, **kwargs):
        title = self.summary(prefix="Original") if display_title else None
        display = plot_stat_map(
            self.original_img,
            bg_img=self.original_img,
            cmap=self.cmap_name,
            annotate=annotate,
            title=title,
            vmin=0,
            vmax=self.vmax,
            draw_cross=False,
            **kwargs,
        )
        self._save_display(display, out)
        # plot_epi(self.original_img, title=title, annotate=annotate, cmap='gray', output_file=out, vmin=0, vmax=self.vmax)

    def plot_grad_cam(self, annotate=False, display_title=True, out=None, **kwargs):
        title = self.summary(prefix="Guided Grad-CAM") if display_title else None
        tmp = self.grad_cam.get_fdata()
        tmp = abs(tmp) ** 0.5
        tmp = nib.Nifti1Image(tmp, self.grad_cam.affine)
        display = plot_stat_map(
            tmp,
            bg_img=self.original_img,
            cmap=self.cmap_name,
            annotate=annotate,
            title=title,
            draw_cross=False,
            colorbar=False,
            **kwargs,
        )
        self._save_display(display, out)

    def save_prob_map(self, out_path):
        nib.save(self.prob_map, out_path)

    def save_ad_component(self, out_path):
        nib.save(self.ad_component, out_path)

    def save_stripped_img(self, out_path):
        nib.save(self.stripped_img, out_path)

    def save_original_img(self, out_path):
        nib.save(self.original_img, out_path)

    def save_grad_cam(self, out_path):
        nib.save(self.grad_cam, out_path)

    @property
    def name(self):
        return self.path.stem


def collect_GAN_results(
    gan, dataset, result, dataset_name, need_grad_cam=True, pathology="abeta"
):
    discriminator = ADOnlyDiscriminator(gan.discriminator)
    target_layer = discriminator.discriminator.model.net.layer_3
    guided_grad_cam = GuidedGradCam(discriminator, target_layer)
    for item in tqdm(dataset):
        img, label = item["image"], item["label"]
        img_path = img.meta.get("filename_or_obj")
        fname = Path(img_path).stem
        ptid = "_".join(fname.split("_")[:3])
        img_id = fname.split("_")[-2]
        normalizer.normalize([img_path], ["./temp/tmp.nii"])
        if pathology == "abeta":
            calc = cl_calc
        elif pathology == "tau":
            calc = ctr_calc
        else:
            raise ValueError(
                f"pathology should be either 'abeta' or 'tau', got {pathology}"
            )
        tracer_name = get_tracer_name(img_path)
        metric = calc.calculate("./temp/tmp.nii", tracer_name)
        img = img.unsqueeze(0).to(gan.device)
        with torch.no_grad():
            processed_img, prob_map, ad_component = gan(img)
            pred_AD_prob, pred_true_fake = (
                gan.discriminator(img).squeeze().cpu().numpy()
            )
            pred_stripped_AD_prob, pred_stripped_true_fake = (
                gan.discriminator(processed_img).squeeze().cpu().numpy()
            )
        if need_grad_cam:
            ad_attribution = guided_grad_cam.attribute(
                img, TARGET_CLASS, interpolate_mode="trilinear"
            )
            result["AD attribution"].append((ad_attribution).sum().item())
        result["fpath"].append(img_path)
        result["ptid"].append(ptid)
        result["img_id"].append(img_id)
        result["tracer"].append(tracer_name)
        result["Diagnosis"].append("AD" if label else "CN")
        result["AD Component"].append(torch.sum(ad_component).item() * VOXEL_VOLUME)
        result["Changed Portion"].append(
            torch.sum(ad_component).item() / torch.sum(img).item()
        )
        result["Dataset"].append(dataset_name)
        result["Pred AD Prob"].append(pred_AD_prob)
        result["Pred True/Fake Prob"].append(pred_true_fake)
        result["Pred Stripped AD Prob"].append(pred_stripped_AD_prob)
        result["Pred Stripped True/Fake Prob"].append(pred_stripped_true_fake)
        result["Metric"].append(metric)
    return result


def load_cerebellar_gray_intensity_normalized(img_path, clean_up=True):
    normalizer.normalize([img_path], ["./tmp.nii"], ["./rigid.nii"])
    cerebellar_gray_suv = cl_calc.ref_SUV("./tmp.nii", "cerebellar gray")
    nii = nib.load("./rigid.nii", mmap=False)
    data = nii.get_fdata()
    data = data / cerebellar_gray_suv
    nii = nib.Nifti1Image(data, nii.affine)
    # clean up
    if clean_up:
        if Path("./tmp.nii").exists():
            os.remove("./tmp.nii")
        if Path("./rigid.nii").exists():
            os.remove("./rigid.nii")
    return nii


def decouple_img(model, image_item):
    discriminator = ADOnlyDiscriminator(model.discriminator)
    target_layer = discriminator.discriminator.model.net.layer_3
    guided_grad_cam = GuidedGradCam(discriminator, target_layer)
    model.eval()
    img, label = image_item["image"], image_item["label"]
    img_path = img.meta["filename_or_obj"]
    img = img.unsqueeze(0).to(model.device)
    affine = img.affine.cpu().numpy()
    with torch.no_grad():
        stripped_image, prob_map, ad_component = model(img)
        pred_AD_prob, pred_fake_prob = model.discriminator(img).squeeze().cpu().numpy()
        out = model.discriminator(stripped_image).squeeze()
        stripped_AD_prob, stripped_fake_prob = out[0].item(), out[1].item()
    ad_attribution = (
        guided_grad_cam.attribute(
            img, target=TARGET_CLASS, interpolate_mode="trilinear"
        )
        .detach()
        .squeeze()
        .cpu()
        .numpy()
    )
    ad_attribution = nib.Nifti1Image(ad_attribution, affine)
    stripped_image = stripped_image.squeeze().cpu().numpy()
    stripped_image = nib.Nifti1Image(stripped_image, affine)
    prob_map = prob_map.squeeze().cpu().numpy()
    prob_map = nib.Nifti1Image(prob_map, affine)
    ad_component = ad_component.squeeze().cpu().numpy()
    ad_component = nib.Nifti1Image(ad_component, affine)
    original_img = img.squeeze().cpu().numpy()
    original_img = nib.Nifti1Image(original_img, affine)
    return DecoupledImage(
        original_img,
        stripped_image,
        prob_map,
        ad_component,
        img_path,
        label,
        pred_AD_prob=pred_AD_prob,
        pred_fake_prob=pred_fake_prob,
        pred_striped_AD_prob=stripped_AD_prob,
        pred_striped_fake_prob=stripped_fake_prob,
        grad_cam=ad_attribution,
    )


def diagnose_decouple_raw_pet(model, img_path: Path, label=None):
    discriminator = ADOnlyDiscriminator(model.discriminator)
    target_layer = discriminator.discriminator.model.net.layer_3
    guided_grad_cam = GuidedGradCam(discriminator, target_layer)
    img = load_cerebellar_gray_intensity_normalized(img_path)
    template = nib.load("./data/ADNI_processed.nii")
    img = resample_to_img(img, template)
    affine = img.affine
    img = (
        torch.from_numpy(img.get_fdata())
        .unsqueeze(0)
        .unsqueeze(0)
        .to(dtype=torch.float32, device=model.device)
    )
    with torch.no_grad():
        out = model.discriminator(img).squeeze()
        AD_prob, fake_prob = out[0].item(), out[1].item()
        stripped_image, prob_map, ad_component = model(img)
        out = model.discriminator(stripped_image).squeeze()
        stripped_AD_prob, stripped_fake_prob = out[0].item(), out[1].item()
    ad_attribution = (
        guided_grad_cam.attribute(
            img, target=TARGET_CLASS, interpolate_mode="trilinear"
        )
        .detach()
        .squeeze()
        .cpu()
        .numpy()
    )
    stripped_image = stripped_image.squeeze().cpu().numpy()
    prob_map = prob_map.squeeze().cpu().numpy()
    ad_component = ad_component.squeeze().cpu().numpy()
    original_img = img.squeeze().cpu().numpy()
    ad_attribution = nib.Nifti1Image(ad_attribution, affine)
    stripped_image = nib.Nifti1Image(stripped_image, affine)
    prob_map = nib.Nifti1Image(prob_map, affine)
    ad_component = nib.Nifti1Image(ad_component, affine)
    original_img = nib.Nifti1Image(original_img, affine)
    return DecoupledImage(
        original_img,
        stripped_image,
        prob_map,
        ad_component,
        img_path,
        label,
        AD_prob,
        fake_prob,
        pred_striped_AD_prob=stripped_AD_prob,
        pred_striped_fake_prob=stripped_fake_prob,
        grad_cam=ad_attribution,
    )


def get_AD_CN_examples(
    dataset, model, n=1
) -> tuple[list[DecoupledImage], list[DecoupledImage]]:
    decoupled_ad, decoupled_cn = [], []
    for img in dataset:
        if img["label"] == 1 and len(decoupled_ad) < n:
            print("found AD", img["image"].meta.get("filename_or_obj"))
            decoupled_ad.append(decouple_img(model, img))
        elif img["label"] == 0 and len(decoupled_cn) < n:
            print("found CN", img["image"].meta.get("filename_or_obj"))
            decoupled_cn.append(decouple_img(model, img))
        if len(decoupled_ad) >= n and len(decoupled_cn) >= n:
            break
    else:
        warnings.warn(
            f"Not enough images. Only found {len(decoupled_ad)} AD and {len(decoupled_cn)} CN images"
        )
    decoupled_ad = sorted(decoupled_ad, key=lambda x: x.pred_AD_prob, reverse=True)
    decoupled_cn = sorted(decoupled_cn, key=lambda x: x.pred_AD_prob)
    return decoupled_ad, decoupled_cn


def save_decoupled_plots(decoupled_image: DecoupledImage, out_dir, patho="abeta"):
    if not out_dir.exists():
        print(f"{out_dir} not found. Created {out_dir}")
        out_dir.mkdir(parents=True)
    out_dir = out_dir / decoupled_image.name
    if not out_dir.exists():
        out_dir.mkdir()
    decoupled_image.save_prob_map(out_dir / f"{decoupled_image.name}_prob_map.nii")
    decoupled_image.save_ad_component(
        out_dir / f"{decoupled_image.name}_ad_component.nii"
    )
    decoupled_image.save_stripped_img(out_dir / f"{decoupled_image.name}_stripped.nii")
    decoupled_image.save_original_img(out_dir / f"{decoupled_image.name}_original.nii")
    decoupled_image.save_grad_cam(out_dir / f"{decoupled_image.name}_grad_cam.nii")
    original_path = out_dir / f"{decoupled_image.name}_original.nii"
    stripped_path = out_dir / f"{decoupled_image.name}_stripped.nii"
    normalizer.normalize([original_path], ["./temp/tmp.nii"])
    normalizer.normalize([stripped_path], ["./temp/tmp_stripped.nii"])
    if patho == "abeta":
        calc = cl_calc
    elif patho == "tau":
        calc = ctr_calc
    else:
        raise ValueError("patho should be either 'abeta' or 'tau'")
    metric_original = calc.calculate("./temp/tmp.nii", get_tracer_name(original_path))
    metric_stripped = calc.calculate(
        "./temp/tmp_stripped.nii", get_tracer_name(stripped_path)
    )
    txt = decoupled_image.summary(join_char="\n")
    txt += f"\n\nOriginal metric: {metric_original:.2f}\nStripped metric: {metric_stripped:.2f}"
    with open(out_dir / f"{decoupled_image.name}_summary.txt", "w") as f:
        f.write(txt)


def ADNI_PET_core_preprocess(input_path, output_path, calibration_factor=1):
    # tau
    # WC SUV: mean, std (1.0181183, 0.04550346)
    # Cerebellar gray SUV: mean, std (0.9960603, 0.030449256)

    # abeta
    # WC SUV: mean, std (1.237246, 0.070811674)
    # Cerebellar gray SUV: mean, std (1.0438542, 0.045583997)
    assert (
        output_path.name != "elastic_tmp.nii"
    ), "output_path should not be elastic_tmp.nii"
    normalizer.normalize(
        [input_path], [Path("./temp/elastic_tmp.nii")], rigid_paths=[output_path]
    )
    wc_suv = cl_calc.ref_SUV("./temp/elastic_tmp.nii")
    nii = nib.load(output_path)
    data = nii.get_fdata() / wc_suv * calibration_factor
    nii = nib.Nifti1Image(data, nii.affine)
    nii = reorder_img(nii)
    nii = resample_to_img(nii, "./data/ADNI_processed.nii")
    nib.save(nii, output_path)
    return wc_suv


def make_box_stripplot(result, Y, palette, ylabel):
    fig, ax = plt.subplots(figsize=(4, 2))
    DATASET_ORDER = ["Val", "Test", "External"]
    DIAGNOSIS_ORDER = ["CN", "AD"]
    ax = sns.stripplot(
        data=result,
        y="Dataset",
        x=Y,
        hue="Diagnosis",
        hue_order=DIAGNOSIS_ORDER,
        jitter=0.2,
        dodge=True,
        alpha=0.5,
        order=DATASET_ORDER,
        size=4,
        palette=palette,
        legend=False,
    )
    ax = sns.boxplot(
        data=result,
        y="Dataset",
        x=Y,
        hue="Diagnosis",
        hue_order=DIAGNOSIS_ORDER,
        showfliers=False,
        order=DATASET_ORDER,
        legend=True,
        ax=ax,
        palette=palette,
    )
    sns.despine(ax=ax)
    plt.setp(ax.patches, alpha=0.5)
    ax.set_xlabel(ylabel)
    return fig, ax
