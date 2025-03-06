from torch.utils.data.sampler import WeightedRandomSampler
from monai.data import Dataset, DataLoader
from constants import ELASTIC_BATCH_SIZE, RIGID_BTACH_SIZE, NUM_WORKER
from transforms import (
    train_rigid_transform,
    val_rigid_transform,
    train_elastic_transform,
    val_elastic_transform,
)
from data import train_rigid, val_rigid, train_elastic, val_elastic

train_rigid_dataset = Dataset(train_rigid, train_rigid_transform)
val_rigid_dataset = Dataset(val_rigid, val_rigid_transform)
train_elastic_dataset = Dataset(
    train_elastic,
    train_elastic_transform,
)
val_elastic_dataset = Dataset(val_elastic, val_elastic_transform)

rigid_sampler = WeightedRandomSampler(
    [item["weight"] for item in train_rigid], len(train_rigid), replacement=True
)
for item in train_rigid:
    del item["weight"]
    del item["tracer"]
for item in val_rigid:
    del item["tracer"]
train_rigid_loader = DataLoader(
    train_rigid_dataset,
    batch_size=RIGID_BTACH_SIZE,
    sampler=rigid_sampler,
    num_workers=NUM_WORKER,
)
val_rigid_loader = DataLoader(
    val_rigid_dataset,
    batch_size=RIGID_BTACH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKER,
)

sampler = WeightedRandomSampler(
    [item["weight"] for item in train_elastic], len(train_elastic), replacement=True
)
for item in train_elastic:
    del item["weight"]
    del item["tracer"]
for item in val_elastic:
    del item["tracer"]
train_elastic_loader = DataLoader(
    train_elastic_dataset,
    batch_size=ELASTIC_BATCH_SIZE,
    num_workers=NUM_WORKER,
    sampler=sampler,
)
val_elastic_dataloader = DataLoader(
    val_elastic_dataset,
    batch_size=ELASTIC_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKER,
)
