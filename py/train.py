from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import nibabel as nib
from monai.visualize import plot_2d_or_3d_image
from nets import RegressorModel, AffineVoxelMorph
from constants import DEVICE
from dataloader import (
    train_rigid_loader,
    val_rigid_loader,
    train_elastic_loader,
    val_elastic_dataloader,
)


def inference(model, batch):
    if isinstance(model, AffineVoxelMorph):
        pred = model(batch["image"], batch["template"])
    else:
        batch["target"] = (
            batch["image_AC"],
            batch["image_PA"],
            batch["image_IS"],
        )
        pred = model(batch["image"])
    loss = model.loss(pred, batch["target"])
    return loss, pred


def train(model, exp_name, max_epochs, train_loader, val_loader, amp=True):
    save_path = Path(f"./exp/{exp_name}")
    if save_path.exists():
        raise ValueError(f"Experiment {exp_name} already exists")
    save_path.mkdir(parents=True)
    writer = SummaryWriter(save_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max_epochs, 1e-6
    )
    scaler = torch.amp.GradScaler()
    min_val_loss = float("inf")
    model_type = model.__class__.__name__
    for epoch in range(max_epochs):
        epoch_loss = {}
        train_loader_tqdm = tqdm(train_loader)
        for batch in train_loader_tqdm:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            if amp:
                with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16):
                    loss, pred = inference(model, batch)
                scaler.scale(loss["Total"]).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, pred = inference(model, batch)
                loss["Total"].backward()
                optimizer.step()
            loss_information = ""
            for k, v in loss.items():
                epoch_loss[k] = epoch_loss.get(k, 0) + v.item()
                loss_information += f"{k}: {v.item():.4f} "
            train_loader_tqdm.set_description(loss_information)
        lr_scheduler.step()
        for k, v in epoch_loss.items():
            writer.add_scalar(f"Train: {k}", v / len(train_loader), epoch)
        writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], epoch)
        model.eval()
        with torch.no_grad():
            val_loss = {}
            for batch in tqdm(val_loader):
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                if isinstance(model, RegressorModel):
                    batch["target"] = (
                        batch["image_AC"],
                        batch["image_PA"],
                        batch["image_IS"],
                    )
                    pred = model(batch["image"])
                else:
                    pred = model(batch["image"], batch["template"])
                loss = model.loss(pred, batch["target"])
                for k, v in loss.items():
                    key = f"{k}"
                    val_loss[key] = val_loss.get(key, 0) + v.item()
            for k, v in val_loss.items():
                writer.add_scalar(f"Val: {k}", v / len(val_loader), epoch)
            if val_loss["Total"] < min_val_loss:
                min_val_loss = val_loss["Total"]
                torch.save(model.state_dict(), save_path / f"best_{model_type}.pth")
            if isinstance(model, AffineVoxelMorph):
                elastic_warped = pred[2]
                plot_2d_or_3d_image(
                    batch["image"], epoch, writer, tag="Val: Original image"
                )
                plot_2d_or_3d_image(
                    elastic_warped, epoch, writer, tag="Val: Predicted image"
                )
                plot_2d_or_3d_image(
                    elastic_warped - batch["target"], epoch, writer, tag="Val: Diff"
                )
            torch.save(model.state_dict(), save_path / f"last_{model_type}.pth")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="rigid", choices=["rigid", "elastic"]
    )
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument(
        "--expname", type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    parser.add_argument("--maxepochs", type=int, default=100)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if args.model == "rigid":
        model = RegressorModel().to(DEVICE)
        train_loader = train_rigid_loader
        val_loader = val_rigid_loader
    else:
        mask = nib.load("./data/padded_mask.nii")
        mask = (
            torch.from_numpy(mask.get_fdata()).reshape((1, 1, *mask.shape)).to(DEVICE)
        )
        model = AffineVoxelMorph(mask).to(DEVICE)
        train_loader = train_elastic_loader
        val_loader = val_elastic_dataloader
    if args.load:
        model.load_state_dict(torch.load(args.load))

    train(
        model,
        args.expname + f"_{args.model}",
        args.maxepochs,
        train_loader,
        val_loader,
        args.amp,
    )
