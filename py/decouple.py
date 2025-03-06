from pathlib import Path
from collections import defaultdict, Counter
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from monai.transforms import (
    LoadImaged,
    RandFlipd,
    SignalFillEmptyd,
    Compose,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandGaussianSharpend,
)
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.blocks import UpSample
from monai.networks.nets import UNet, Classifier
from monai.data import Dataset, DataLoader
from monai.visualize import plot_2d_or_3d_image
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryConfusionMatrix,
    BinaryF1Score,
)
import matplotlib.pyplot as plt

from utils import get_tracer_name
from decouple_constants import (
    AV45_DIAGNOSIS_DICT,
    FBB_DIAGNOSIS_DICT,
    FMM_DIAGNOSIS_DICT,
)


VOXEL_VOLUME = (1.5 / 10) ** 3  # cm^3


def count_tracer(data_list):
    names = ["CN", "AD", "NA"]
    item_styles = [{"color": "#91cc75"}, {"color": "purple"}, {"color": "#afafaf"}]
    tracer_names = [[], [], []]
    for item in data_list:
        tracer_names[item["label"]].append(get_tracer_name(item["image"]))

    result = []
    for i, category in enumerate(tracer_names):
        if len(category) == 0:
            continue
        counter = Counter(category)
        result.append(
            {
                "name": f"{names[i]}",
                "value": len(category),
                "children": [
                    {"name": f"{k} ({v})", "value": v, "itemStyle": item_styles[i]}
                    for k, v in counter.items()
                ],
                "itemStyle": item_styles[i],
            }
        )
    print(result)


def split_based_on_individuals(data, train_size=0.8, random_state=42):
    d = defaultdict(list)
    for item in data:
        ptid = Path(item["image"]).stem.split("_")[:3]
        ptid = "_".join(ptid)
        d[ptid].append(item)
    d = list(d.values())

    train, val = train_test_split(d, train_size=train_size, random_state=random_state)
    val, test = train_test_split(val, train_size=0.5, random_state=random_state)
    counter = Counter(x[0]["label"] for x in d)
    print(
        "Patients:", len(d), "Train:", len(train), "Val:", len(val), "Test:", len(test)
    )
    print(counter)
    train = [item for sublist in train for item in sublist]
    val = [item for sublist in val for item in sublist]
    test = [item for sublist in test for item in sublist]
    print(
        "Scans:", len(data), "Train:", len(train), "Val:", len(val), "Test:", len(test)
    )
    return train, val, test


class ADCNDataModule(pl.LightningDataModule):
    """You should implement your own data module here."""

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        pathology="abeta",
        extend_dataset=True,
        demo_mode=False,
    ):
        super().__init__()
        self.demo_mode = demo_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pathology = pathology
        self.extend_dataset = extend_dataset

        self.transform = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                SignalFillEmptyd(keys=["image"]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandAffined(
                    keys=["image"],
                    prob=0.1,
                    rotate_range=10 / 180 * 3.1415926,
                    scale_range=0.1,
                    translate_range=10,
                    padding_mode="zeros",
                ),
                RandGaussianNoised(keys=["image"], prob=0.1),
                RandGaussianSmoothd(keys=["image"], prob=0.1),
                RandGaussianSharpend(keys=["image"], prob=0.1),
            ]
        )
        self.val_transform = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                SignalFillEmptyd(keys=["image"]),
            ]
        )

    @staticmethod
    def _read_lines(filename, label):
        with open(filename, "r") as f:
            data = f.readlines()
            data = [{"image": x.strip(), "label": label} for x in data]
        return data

    def _read_nii(self, root, label_fn):
        data = []
        for img in root.glob("*.nii"):
            data.append({"image": str(img), "label": label_fn(img)})
        for img in root.glob("*.nii.gz"):
            data.append({"image": str(img), "label": label_fn(img)})
        return data

    def setup(self, stage=None):
        if self.pathology == "abeta":
            datalist = []
            datalist.extend(self._read_lines("./AD-decouple-dataset/lccn.txt", 0))
            datalist.extend(self._read_lines("./AD-decouple-dataset/hccn.txt", 0))
            datalist.extend(self._read_lines("./AD-decouple-dataset/mci.txt", 1))
            datalist.extend(self._read_lines("./AD-decouple-dataset/cad.txt", 1))
        elif self.pathology == "tau":
            AD_root = Path("../tau/ALL_AD_tau")
            ad_list = [{"image": str(x), "label": 1} for x in AD_root.glob("*.nii")]
            CN_root = Path("../tau/CN_ALL_TIME_tau")
            cn_list = [{"image": str(x), "label": 0} for x in CN_root.glob("*.nii")]
            datalist = ad_list + cn_list
        else:
            raise ValueError("Unknown pathology")

        # print out distribution
        total_ad = sum([x["label"] for x in datalist])
        total_cn = len(datalist) - total_ad
        print(f"pathology: {self.pathology}")
        print(f"AD: {total_ad}, CN: {total_cn}")

        # test mode
        # datalist = random.sample(datalist, 100)
        self.train_data, self.val_data, self.test_data = split_based_on_individuals(
            datalist
        )
        self.datalist = datalist
        extended_list = []
        if self.extend_dataset:
            # extend training data!
            if self.pathology == "abeta":
                root = Path("./AD-decouple-dataset/extended_dataset/AV45")
                extended_list.extend(self._read_nii(root, lambda x: -1))
                root = Path("./AD-decouple-dataset/extended_dataset/PIB")
                extended_list.extend(self._read_nii(root, lambda x: -1))
                root = Path("./centiloid_eval/NAV4694/ADNI_pet_core_processed")
                extended_list.extend(self._read_nii(root, lambda x: -1))
            elif self.pathology == "tau":
                root = Path("./AD-decouple-dataset/extended_dataset/APN")
                extended_list.extend(self._read_nii(root, lambda x: -1))
                root = Path("./AD-decouple-dataset/extended_dataset/AV1451")
                extended_list.extend(self._read_nii(root, lambda x: -1))
            count_tracer(
                extended_list + self.train_data + self.val_data + self.test_data
            )
            self.train_data.extend(extended_list)
        self.train_data = Dataset(self.train_data, self.transform)
        self.val_data = Dataset(self.val_data, self.val_transform)
        self.test_data = Dataset(self.test_data, self.val_transform)

        # external validation
        external_test_datalist = []
        fn = lambda x: 1 if x.name.startswith("AD") else 0
        if self.pathology == "abeta":
            root = Path("./centiloid_eval/PiB/ADNI_pet_core_processed")
            external_test_datalist.extend(self._read_nii(root, fn))
            root = Path("./centiloid_eval/AV45/ADNI_pet_core_processed")
            external_test_datalist.extend(
                [
                    {"image": str(root / (k + ".nii")), "label": v}
                    for k, v in AV45_DIAGNOSIS_DICT.items()
                ]
            )
            root = Path("./centiloid_eval/FBB/ADNI_pet_core_processed")
            external_test_datalist.extend(
                [
                    {"image": str(root / (k + ".nii")), "label": v}
                    for k, v in FBB_DIAGNOSIS_DICT.items()
                ]
            )
            root = Path("./centiloid_eval/FMM/ADNI_pet_core_processed")
            external_test_datalist.extend(
                [
                    {"image": str(root / (k + ".nii")), "label": v}
                    for k, v in FMM_DIAGNOSIS_DICT.items()
                ]
            )

        elif self.pathology == "tau":
            root = Path("./centaur_eval/FTP/ADNI_pet_core_processed")
            external_test_datalist.extend(self._read_nii(root, fn))
        else:
            raise ValueError(f"Unknown pathology: {self.pathology}")
        total = len(external_test_datalist)
        ad = sum([x["label"] for x in external_test_datalist])
        print(f"External test data: {total}, AD: {ad}, CN: {total - ad}")
        count_tracer(external_test_datalist)
        self.external_test_data = Dataset(external_test_datalist, self.val_transform)

        if self.demo_mode:
            print("Demo mode")
            self.train_data = self.train_data[:10]
            self.val_data = self.val_data[:10]
            self.test_data = self.test_data[:10]
            self.external_test_data = self.external_test_data[:10]

    def train_dataloader(self):
        from collections import Counter

        cnt = Counter(x["label"] for x in self.train_data)
        weights = [1 / cnt[x["label"]] for x in self.train_data]
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            # shuffle=True,
            num_workers=self.num_workers,
            sampler=sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class CustomUNet(UNet):
    def _get_up_layer(
        self, in_channels: int, out_channels: int, strides: int, is_top: bool
    ) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Convolution | nn.Sequential

        # conv = Convolution(
        #     self.dimensions,
        #     in_channels,
        #     out_channels,
        #     strides=strides,
        #     kernel_size=self.up_kernel_size,
        #     act=self.act,
        #     norm=self.norm,
        #     dropout=self.dropout,
        #     bias=self.bias,
        #     conv_only=is_top and self.num_res_units == 0,
        #     is_transposed=True,
        #     adn_ordering=self.adn_ordering,
        # )

        # change deconv to upsampling + conv
        conv = nn.Sequential(
            UpSample(
                self.dimensions,
                in_channels=in_channels,
                out_channels=in_channels,
                scale_factor=2,
                mode="nontrainable",
            ),
            # nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
            Convolution(
                self.dimensions,
                in_channels,
                out_channels,
                strides=1,
                kernel_size=self.up_kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                conv_only=is_top and self.num_res_units == 0,
                adn_ordering=self.adn_ordering,
            ),
        )
        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv


class Decompositor(nn.Module):
    # decompositor will try to strip AD component from the input image
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            *[
                CustomUNet(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=1,
                    channels=(16, 32, 32, 32, 32, 32),
                    strides=(2, 2, 2, 2, 2),
                ),
                nn.Sigmoid(),
            ]
        )

    def forward(self, x):
        prob_map = self.model(x)
        return x - x * prob_map, x * prob_map, prob_map


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Classifier(
            in_shape=(1, 160, 160, 96),
            classes=2,  # AD or not AD + real or fake
            channels=(16, 32, 64, 128, 1),
            strides=(2, 2, 2, 2),
            dropout=0.25,
            last_act="sigmoid",
        )

    def forward(self, x):
        return self.model(x)  # 0 for CN/real, 1 for AD/fake


class ONNXGAN(pl.LightningModule):
    def __init__(self, lr=1e-4, beta=5, alpha=0.15, theta=5, pathology="abeta"):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Decompositor()
        self.discriminator = Discriminator()

    def forward(self, x):
        stripped_AD_images, stripped_component, AD_prob_map = self.generator(x)
        discriminator_res = self.discriminator(x)
        AD_prob, fake_prob = discriminator_res[:, 0], discriminator_res[:, 1]
        discriminator_res = self.discriminator(stripped_AD_images)
        stripped_AD_prob, stripped_fake_prob = (
            discriminator_res[:, 0],
            discriminator_res[:, 1],
        )
        ADAD_scores = (
            torch.sum(stripped_component, dim=tuple(range(1, x.dim()))) * VOXEL_VOLUME
        )
        return {
            "stripped_AD_images": stripped_AD_images,
            "stripped_component": stripped_component,
            "AD_prob_map": AD_prob_map,
            "AD_prob": AD_prob,
            "fake_prob": fake_prob,
            "stripped_AD_prob": stripped_AD_prob,
            "stripped_fake_prob": stripped_fake_prob,
            "ADAD_scores": ADAD_scores,
        }


class GAN(pl.LightningModule):
    def __init__(self, lr=1e-4, beta=5, alpha=0.15, theta=5, pathology="abeta"):
        # alpha: stripping cost factor, to prevent meanleaningless stripping
        # alpha = 0.1, AD loss will be 0.1, CN loss will be 0.9
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.generator = Decompositor()
        # discriminator will try to diagnose if the input image is AD or not
        self.discriminator = Discriminator()

        self.accuracy_calc = BinaryAccuracy()
        self.auc_calc = BinaryAUROC()
        self.confusion_matrix_calc = BinaryConfusionMatrix()
        self.f1_calc = BinaryF1Score()

        self.val_diagnosis_prob = []
        self.val_real_fake_prob = []
        self.val_gt_diagnosis = []
        self.val_gt_real_fake = []

        self.cn_img = None
        self.ad_img = None

    def forward(self, x):
        return self.generator(x)

    def adversarial_loss(self, y_hat, y, reduction="mean"):
        y = y.float()
        return F.binary_cross_entropy(y_hat, y, reduction=reduction)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch["image"], batch["label"]
        opt_g, opt_d = self.optimizers()

        self.toggle_optimizer(opt_g)
        stripped_AD_images, stripped_component, AD_prob_map = self.generator(imgs)

        # the AD component are stripped, so the discriminator should not be able to detect AD
        # moreover, the discriminator should not be able to detect if the image is stripped or not
        # besides, decompositor should not strip CN images
        g_loss = (
            self.adversarial_loss(
                self.discriminator(stripped_AD_images)[:, 0], torch.zeros_like(labels)
            )
            + self.adversarial_loss(
                self.discriminator(stripped_AD_images)[:, 1], torch.zeros_like(labels)
            )
            + (
                torch.mean(AD_prob_map, dim=tuple(range(1, AD_prob_map.dim())))
                + torch.mean(AD_prob_map**2, dim=tuple(range(1, AD_prob_map.dim())))
            )
            * self.hparams.theta
        )
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        self.toggle_optimizer(opt_d)
        # if the image is not touched by the decompositor, the discriminator should be able to tell that this image is not modified
        # if the image is touched by the decompositor, the discriminator should be able to tell how much it is modified
        d_real_loss = self.adversarial_loss(
            self.discriminator(imgs)[:, 1], torch.zeros_like(labels)
        )
        modified_target = (
            torch.mean(
                stripped_component.detach(), dim=tuple(range(1, AD_prob_map.dim()))
            )
            / torch.mean(imgs, dim=tuple(range(1, AD_prob_map.dim())))
            * self.hparams.beta
        )
        modified_target = torch.clip(modified_target, 0, 1)
        d_strip_loss = self.adversarial_loss(
            self.discriminator(stripped_AD_images.detach())[:, 1],
            modified_target,
        )

        mask = labels != -1
        if mask.sum() == 0:
            d_ad_original_loss = 0
            d_ad_strip_loss = 0
        else:
            imgs = imgs[mask]
            labels = labels[mask]
            stripped_AD_images = stripped_AD_images[mask]
            stripped_component = stripped_component[mask]
            AD_prob_map = AD_prob_map[mask]
            # the discriminator should be able to detect if the image is AD or not
            # even if the image is stripped
            d_ad_original_loss = self.adversarial_loss(
                self.discriminator(imgs)[:, 0], labels
            )
            d_ad_strip_loss = 0

        d_loss = (d_ad_original_loss + d_ad_strip_loss) / 2 + (
            d_real_loss + d_strip_loss
        ) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

    def configure_optimizers(self):
        lr = self.hparams.lr

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []

    def validation_step(self, batch, batch_idx):
        # here we only test the discriminator
        imgs, labels = batch["image"], batch["label"]
        diagnosis_prob = self.discriminator(imgs)[:, 0]
        real_fake_prob = self.discriminator(imgs)[:, 1]
        self.val_diagnosis_prob.append(
            diagnosis_prob.as_tensor()
        )  # to convert monai.metatensor to torch.tensor
        self.val_real_fake_prob.append(real_fake_prob.as_tensor())
        self.val_gt_diagnosis.append(labels)
        self.val_gt_real_fake.append(torch.zeros_like(labels))

        if self.ad_img is not None and self.cn_img is not None:
            return
        for img, label in zip(imgs, labels):
            if label == 1 and self.ad_img is None:
                self.ad_img = img.unsqueeze(0)
            elif label == 0 and self.cn_img is None:
                self.cn_img = img.unsqueeze(0)
            if self.ad_img is not None and self.cn_img is not None:
                break

    def _plot_confusion_matrix(self, confusion_matrix, display_labels):
        disp = ConfusionMatrixDisplay(confusion_matrix, display_labels=display_labels)
        fig, ax = plt.subplots(figsize=(4, 4))
        disp.plot(ax=ax)
        return fig

    def on_validation_epoch_end(self):
        # let's calculate metrics
        self.val_diagnosis_prob = torch.concat(self.val_diagnosis_prob)
        self.val_real_fake_prob = torch.concat(self.val_real_fake_prob)
        self.val_gt_diagnosis = torch.concat(self.val_gt_diagnosis)
        self.val_gt_real_fake = torch.concat(self.val_gt_real_fake)

        diagnosis_acc = self.accuracy_calc(
            self.val_diagnosis_prob, self.val_gt_diagnosis
        )
        diagnosis_auc = self.auc_calc(self.val_diagnosis_prob, self.val_gt_diagnosis)
        diagnosis_confusion_matrix = self.confusion_matrix_calc(
            self.val_diagnosis_prob, self.val_gt_diagnosis
        )
        diagnosis_f1 = self.f1_calc(self.val_diagnosis_prob, self.val_gt_diagnosis)
        diagnosis_cm_fig = self._plot_confusion_matrix(
            diagnosis_confusion_matrix.cpu().numpy(), ["CN", "AD"]
        )

        real_fake_acc = self.accuracy_calc(
            self.val_real_fake_prob, self.val_gt_real_fake
        )
        real_fake_auc = self.auc_calc(self.val_real_fake_prob, self.val_gt_real_fake)
        real_fake_confusion_matrix = self.confusion_matrix_calc(
            self.val_real_fake_prob, self.val_gt_real_fake
        )
        real_fake_cm_fig = self._plot_confusion_matrix(
            real_fake_confusion_matrix.cpu().numpy(), ["Real", "Fake"]
        )

        # log metrics
        self.log("Val Diagnosis Accuracy", diagnosis_acc, on_epoch=True, on_step=False)
        self.log("Val Diagnosis AUC", diagnosis_auc, on_epoch=True, on_step=False)
        self.logger.experiment.add_figure(
            "Val Diagnosis Confusion Matrix", diagnosis_cm_fig, self.current_epoch
        )
        self.log("Val Real-Fake Accuracy", real_fake_acc, on_epoch=True, on_step=False)
        self.log("Val Real-Fake AUC", real_fake_auc, on_epoch=True, on_step=False)
        self.log("Val Diagnosis F1", diagnosis_f1, on_epoch=True, on_step=False)
        self.logger.experiment.add_figure(
            "Val Real-Fake Confusion Matrix", real_fake_cm_fig, self.current_epoch
        )

        self.val_diagnosis_prob = []
        self.val_real_fake_prob = []
        self.val_gt_diagnosis = []
        self.val_gt_real_fake = []

        if self.ad_img is not None:
            stripped_AD_img, stripped_component, prob_map = self.generator(self.ad_img)
            plot_2d_or_3d_image(
                prob_map,
                self.current_epoch,
                self.logger.experiment,
                tag="AD/probmap",
            )
            plot_2d_or_3d_image(
                stripped_component,
                self.current_epoch,
                self.logger.experiment,
                tag="AD/stripped component",
            )
        if self.cn_img is not None:
            stripped_CN_img, stripped_component, prob_map = self.generator(self.cn_img)
            plot_2d_or_3d_image(
                prob_map,
                self.current_epoch,
                self.logger.experiment,
                tag="CN/probmap",
            )
            plot_2d_or_3d_image(
                stripped_component,
                self.current_epoch,
                self.logger.experiment,
                tag="CN/stripped component",
            )
        self.ad_img = None
        self.cn_img = None


if __name__ == "__main__":
    pathology = "tau"
    checkpoint_callback = ModelCheckpoint(
        monitor="Val Diagnosis F1",
        dirpath=f"./decomposition-exp-{pathology}",
        filename="best-diagnosis-f1-{epoch:02d}-{Val Diagnosis F1:.2f}",
        mode="max",
        save_last=True,
        save_top_k=3,
        save_weights_only=False,
    )
    data = ADCNDataModule(batch_size=1, num_workers=1, pathology=pathology)
    logger = TensorBoardLogger(f"./decomposition-exp-{pathology}", name="decomposition")
    trainer = pl.Trainer(
        max_epochs=30,
        default_root_dir=f"./decomposition-exp-{pathology}",
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        GAN(pathology=pathology), data, ckpt_path="./decomposition-exp-tau/last.ckpt"
    )
