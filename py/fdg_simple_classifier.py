from collections import defaultdict
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    RandFlipd,
    RandAffined,
    RandGaussianNoised,
    MapTransform,
    ResampleToMatch,
    LoadImage,
)
from monai.networks.nets import DenseNet121
from monai.networks.layers.factories import Act
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryConfusionMatrix,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import re


logger = TensorBoardLogger("tb_logs", name="my_model")


def split_based_on_individuals(data, train_size=0.8, random_state=42):
    d = defaultdict(list)
    for item in data:
        ptid = Path(item["image"]).stem.split("_")[:3]
        ptid = "_".join(ptid)
        d[ptid].append(item)
    d = list(d.values())
    train, val = train_test_split(
        d, test_size=1 - train_size, random_state=random_state
    )
    val, test = train_test_split(val, test_size=0.5, random_state=random_state)
    # flatten the list of lists
    print(
        f"Patients\nTotal: {len(d)}, Train: {len(train)}, Val: {len(val)}, Test: {len(test)}"
    )
    train = [item for sublist in train for item in sublist]
    val = [item for sublist in val for item in sublist]
    test = [item for sublist in test for item in sublist]
    print(
        f"Scans\nTotal: {len(train) + len(val) + len(test)}, Train: {len(train)}, Val: {len(val)}, Test: {len(test)}"
    )
    return train, val, test


def summarize_OASIS_data(datalist):
    pattern = re.compile(r"OAS\d+")
    d = defaultdict(list)
    for item in datalist:
        ptid = pattern.search(Path(item["image"]).stem).group()
        d[ptid].append(item)
    d = list(d.values())
    print(f"Patients: {len(d)}")
    print(f"Scans: {len(datalist)}")
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


class SimpleDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=8, num_workers=8):
        super(SimpleDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        AD_dir = Path("../ADNI/ALL_AD/FDG")
        CN_dir = Path("../ADNI/CN_ALL_TIME/FDG")
        AD_list = [{"image": str(f), "label": 1} for f in AD_dir.glob("*.nii")]
        CN_list = [{"image": str(f), "label": 0} for f in CN_dir.glob("*.nii")]
        self.train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandAffined(
                    keys=["image"], rotate_range=10, scale_range=0.1, mode="bilinear"
                ),
                RandGaussianNoised(keys=["image"]),
            ]
        )
        self.val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
            ]
        )
        self.test_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
            ]
        )
        self.external_test_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                ResampleToMatchTemplated(
                    keys=["image"], template_path="./ADNI_processed.nii"
                ),
            ]
        )
        total_list = AD_list + CN_list
        print("AD dataset:")
        split_based_on_individuals(AD_list)
        print("CN dataset:")
        split_based_on_individuals(CN_list)
        train, val, test = split_based_on_individuals(total_list)
        self.train = Dataset(data=train, transform=self.train_transforms)
        self.val = Dataset(data=val, transform=self.val_transforms)
        test = sorted(test, key=lambda x: Path(x["image"]).stem)
        self.test = Dataset(data=test, transform=self.test_transforms)

        internal_AD = Path("../ADNI/preprocessed_with_DCCC/AD")
        internal_CN = Path("../ADNI/preprocessed_with_DCCC/CN")
        internal_AD_list = [
            {"image": str(f), "label": 1} for f in internal_AD.glob("*.nii")
        ]
        internal_CN_list = [
            {"image": str(f), "label": 0} for f in internal_CN.glob("*.nii")
        ]
        internal_total_list = internal_AD_list + internal_CN_list
        internal_total_list = sorted(
            internal_total_list, key=lambda x: Path(x["image"]).stem
        )
        self.internal_test = Dataset(
            data=internal_total_list, transform=self.external_test_transforms
        )

        external_AD = Path("./OASIS3_FDG_nii/AD_AT_LAST_aligned")
        external_CN = Path("./OASIS3_FDG_nii/CN_ALL_TIME_aligned")
        external_AD_list = [
            {"image": str(f), "label": 1} for f in external_AD.glob("*.nii")
        ]
        external_CN_list = [
            {"image": str(f), "label": 0} for f in external_CN.glob("*.nii")
        ]
        print("OASIS3 AD dataset:")
        summarize_OASIS_data(external_AD_list)
        print("OASIS3 CN dataset:")
        summarize_OASIS_data(external_CN_list)
        external_total_list = external_AD_list + external_CN_list
        self.external_test = Dataset(
            data=external_total_list, transform=self.external_test_transforms
        )

    def train_dataloader(self):
        AD_total = sum(x["label"] for x in self.train)
        AD_weight = 1 / AD_total
        CN_weight = 1 / (len(self.train) - AD_total)
        weights = [AD_weight if x["label"] == 1 else CN_weight for x in self.train]
        weighted_sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            sampler=weighted_sampler,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def external_test_dataloader(self):
        return DataLoader(
            self.external_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def internal_test_dataloader(self):
        return DataLoader(
            self.internal_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class SimpleModel(pl.LightningModule):
    def __init__(self, plot=False):
        super(SimpleModel, self).__init__()
        self.model = nn.Sequential(
            *[
                DenseNet121(
                    spatial_dims=3, in_channels=1, out_channels=1, act=Act.PRELU
                ),
                nn.Sigmoid(),
            ]
        )
        self.loss = nn.BCELoss()

        self.auc_roc_calc = BinaryAUROC()
        self.accuracy_calc = BinaryAccuracy()
        self.precision_calc = BinaryPrecision()
        self.recall_calc = BinaryRecall()
        self.f1_calc = BinaryF1Score()
        self.confusion_matrix_calc = BinaryConfusionMatrix()

        self._val_pred_AD_prob = []
        self._val_gt = []

    @property
    def val_gt(self):
        if isinstance(self._val_gt, list):
            return torch.cat(self._val_gt).cpu().numpy()
        return self._val_gt.cpu().numpy()

    @property
    def val_pred_AD_prob(self):
        if isinstance(self._val_pred_AD_prob, list):
            return torch.cat(self._val_pred_AD_prob).cpu().numpy()
        return self._val_pred_AD_prob.cpu().numpy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch["image"], batch["label"]
        pred_AD_prob = self(imgs)
        labels = torch.unsqueeze(labels, 1).to(dtype=pred_AD_prob.dtype)
        loss = self.loss(pred_AD_prob, labels)
        self.log("Train/Loss", loss, prog_bar=True)
        return loss

    def binary_classification_report(self, y_true, y_pred_prob):

        auc_roc = self.auc_roc_calc(y_pred_prob, y_true)
        accuracy = self.accuracy_calc(y_pred_prob, y_true)
        precision = self.precision_calc(y_pred_prob, y_true)
        recall = self.recall_calc(y_pred_prob, y_true)
        f1 = self.f1_calc(y_pred_prob, y_true)
        confusion_matrix = self.confusion_matrix_calc(y_pred_prob, y_true)
        return {
            "Val/auc_roc": auc_roc,
            "Val/accuracy": accuracy,
            "Val/precision": precision,
            "Val/recall": recall,
            "Val/f1": f1,
            "Val/confusion_matrix": confusion_matrix,
        }

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch["image"], batch["label"]
        pred_AD_prob = self(imgs)
        labels = torch.unsqueeze(labels, 1).to(dtype=pred_AD_prob.dtype)
        loss = self.loss(pred_AD_prob, labels)
        self.log("Val/Loss", loss, prog_bar=True)
        self._val_pred_AD_prob.append(pred_AD_prob)
        self._val_gt.append(labels)

    def split_confusion_matrix(self, confusion_matrix):
        tn, fp, fn, tp = confusion_matrix.view(-1)
        return tn, fp, fn, tp

    def on_validation_epoch_start(self):
        self._val_pred_AD_prob = []
        self._val_gt = []

    def on_validation_epoch_end(self):
        y_pred_prob = torch.cat(self._val_pred_AD_prob)
        y_true = torch.cat(self._val_gt)
        metrics = self.binary_classification_report(y_true, y_pred_prob)
        confusion_matrix = metrics.pop("Val/confusion_matrix")
        fig = self._plot_confusion_matrix(
            confusion_matrix.cpu().numpy(), display_labels=["CN", "AD"]
        )
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)
        self.logger.experiment.add_figure(
            "Val/Confusion Matrix", fig, self.current_epoch
        )

    def test_step(self, batch, batch_idx):
        imgs, labels = batch["image"], batch["label"]
        pred_AD_prob = self(imgs)
        labels = torch.unsqueeze(labels, 1).to(dtype=pred_AD_prob.dtype)
        self._val_pred_AD_prob.append(pred_AD_prob)
        self._val_gt.append(labels)

    def on_test_epoch_end(self):
        y_pred_prob = torch.cat(self._val_pred_AD_prob)
        y_true = torch.cat(self._val_gt)
        metrics = self.binary_classification_report(y_true, y_pred_prob)
        confusion_matrix = metrics.pop("Val/confusion_matrix")

        tn, fp, fn, tp = self.split_confusion_matrix(confusion_matrix)
        metrics["Val/tn"] = tn
        metrics["Val/fp"] = fp
        metrics["Val/fn"] = fn
        metrics["Val/tp"] = tp
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)

    def on_test_epoch_start(self):
        self._val_pred_AD_prob = []
        self._val_gt = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _plot_confusion_matrix(self, confusion_matrix, display_labels):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix, display_labels=display_labels
        )
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42
        sns.set_context("paper")
        fig, ax = plt.subplots(figsize=(4, 4))
        disp.plot(ax=ax)
        return fig


def main():
    dm = SimpleDataModule()
    checkpoint_callback = ModelCheckpoint(
        monitor="Val/auc_roc",
        dirpath="logs",
        filename="simple-fdg-classification-{epoch:02d}-{Val/auc_roc:.2f}",
        save_last=True,
        mode="max",
        save_top_k=1,
        save_weights_only=False,
    )
    model = SimpleModel()
    logger = TensorBoardLogger("logs", name="simple-fdg-classification")
    trainer = pl.Trainer(
        max_epochs=50,
        default_root_dir="logs",
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=2,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
