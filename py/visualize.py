import argparse
import pyvista as pv
import numpy as np
from numpy.typing import ArrayLike
import nibabel as nib

from transforms import (
    train_rigid_transform,
    val_rigid_transform,
    train_elastic_transform,
    val_elastic_transform,
)
from constants import RIGID_INSHAPE


class Visualizer:
    def __init__(self, vol: ArrayLike, resolution: ArrayLike = None, mode="simple"):
        self.plotter = pv.Plotter()
        if mode == "simple":
            self.plotter.add_volume(
                self.highlight_border(vol), cmap="gray_r", opacity="linear"
            )
        else:
            spacing = (
                np.array([1, 1, 1]) if resolution is None else np.array(resolution)
            )
            grid = pv.ImageData(dimensions=np.array(vol.shape) + 1, spacing=spacing)
            grid.cell_data["values"] = vol.flatten(order="F")
            self.plotter.add_mesh_clip_plane(grid, cmap="gray_r")
        self.vol_shape = vol.shape[::-1]
        self.points = []
        self.vectors = []

    def add_arrow(self, origin, direction, color="red", scale=50):
        self.vectors.append((origin, direction))
        arrow = pv.Arrow(start=origin, direction=direction, scale=scale)
        self.plotter.add_mesh(arrow, color=color)

    def add_point(self, point, color="red", radius=5):
        self.points.append(point)
        self.plotter.add_mesh(pv.Sphere(center=point, radius=radius), color=color)

    def show(self):
        self.plotter.show()

    def print(self):
        print("Points:")
        for p in self.points:
            print(p)
        print(f"Vectors:")
        for v in self.vectors:
            print(v)

    @staticmethod
    def highlight_border(vol: ArrayLike):
        max_val = np.max(vol)
        vol[0, ...] = max_val / 5
        vol[-1, ...] = max_val / 5
        vol[:, 0, :] = max_val / 5
        vol[:, -1, :] = max_val / 5
        vol[:, :, 0] = max_val / 5
        vol[:, :, -1] = max_val / 5
        return vol


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input volume file")
    parser.add_argument(
        "--aug",
        choices=["rt", "rv", "vt", "vv"],
        help="rt: rigid train, rv: rigid validation, vt: voxelmorph train, vv: voxelmorph validation",
    )
    parser.add_argument("--mode", type=str, default="simple", help="simple or clip")
    parser.add_argument(
        "--key", type=str, default="image", choices=["image", "target", "template"]
    )
    args = parser.parse_args()
    if not args.aug:
        nii = nib.load(args.input)
        vol = nii.get_fdata()
        res = nii.header.get_zooms()
        vis = Visualizer(vol, resolution=res, mode=args.mode)
    else:
        d = {"image": args.input}
        if args.aug == "rt":
            transform = train_rigid_transform
        elif args.aug == "rv":
            transform = val_rigid_transform
        elif args.aug == "vt":
            transform = train_elastic_transform
        elif args.aug == "vv":
            transform = val_elastic_transform
        else:
            raise ValueError("Invalid augmentation")
        if args.aug in ["vt", "vv"]:
            d["target"] = args.input
        img = transform(d)
        vol = img[args.key].squeeze().cpu().numpy()
        vis = Visualizer(vol, mode=args.mode)
        if args.aug in ["rt", "rv"]:
            point_AC = img["image_AC"].squeeze().cpu().numpy()
            point_AC = point_AC * RIGID_INSHAPE[1:]
            vec_PA = img["image_PA"].squeeze().cpu().numpy()
            vec_IS = img["image_IS"].squeeze().cpu().numpy()
            vis.add_arrow(point_AC, vec_PA, color="red")
            vis.add_arrow(point_AC, vec_IS, color="blue")
            vis.add_point(point_AC)
            vis.print()
    vis.show()
