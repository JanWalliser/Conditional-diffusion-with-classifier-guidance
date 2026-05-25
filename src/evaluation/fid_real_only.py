import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm
from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.models import inception_v3, Inception_V3_Weights


LABELS = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def load_cifar10_images(
    data_root: str,
    split: str,
    class_id: Optional[int] = None,
    max_images: Optional[int] = None,
    download: bool = True,
) -> torch.Tensor:
    train = split.lower() == "train"

    dataset = datasets.CIFAR10(
        root=data_root,
        train=train,
        download=download,
        transform=transforms.ToTensor(),
    )

    images = []

    for image, label in tqdm(dataset, desc=f"Loading CIFAR-10 {split}"):
        if class_id is not None and label != class_id:
            continue

        images.append(image)

        if max_images is not None and len(images) >= max_images:
            break

    if len(images) == 0:
        raise RuntimeError(f"No CIFAR-10 images found for split={split}, class_id={class_id}")

    return torch.stack(images, dim=0)


class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights, transform_input=False)
        model.fc = nn.Identity()
        model.eval()

        self.model = model

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=(299, 299),
            mode="bilinear",
            align_corners=False,
        )

        x = (x - self.mean) / self.std
        features = self.model(x)

        return features


@torch.no_grad()
def extract_inception_features(
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model = InceptionFeatureExtractor().to(device)
    model.eval()

    features = []

    for start in tqdm(range(0, images.shape[0], batch_size), desc="Extracting features"):
        batch = images[start:start + batch_size].to(device)
        batch_features = model(batch)
        features.append(batch_features.cpu())

    features = torch.cat(features, dim=0)
    return features.numpy()


def calculate_statistics(features: np.ndarray):
    mean = np.mean(features, axis=0)
    covariance = np.cov(features, rowvar=False)
    variance = np.diag(covariance)

    return mean, variance, covariance


def calculate_fid_from_stats(
    mean_1: np.ndarray,
    covariance_1: np.ndarray,
    mean_2: np.ndarray,
    covariance_2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    diff = mean_1 - mean_2

    covmean = sqrtm(covariance_1 @ covariance_2)

    if not np.isfinite(covmean).all():
        print("Non-finite covariance sqrt. Adding epsilon to diagonals.")
        offset = np.eye(covariance_1.shape[0]) * eps
        covmean = sqrtm((covariance_1 + offset) @ (covariance_2 + offset))

    if np.iscomplexobj(covmean):
        imaginary_part = np.max(np.abs(covmean.imag))

        if imaginary_part > 1e-3:
            raise ValueError(f"Large imaginary component in sqrtm: {imaginary_part}")

        covmean = covmean.real

    fid = (
        diff.dot(diff)
        + np.trace(covariance_1)
        + np.trace(covariance_2)
        - 2.0 * np.trace(covmean)
    )

    return float(fid)


def print_statistics(
    name: str,
    mean: np.ndarray,
    variance: np.ndarray,
    covariance: np.ndarray,
    preview_values: int = 10,
):
    print("=" * 80)
    print(f"Statistics: {name}")
    print("=" * 80)

    print(f"Mean shape:       {mean.shape}")
    print(f"Variance shape:   {variance.shape}")
    print(f"Covariance shape: {covariance.shape}")

    print()
    print(f"Mean first {preview_values}:")
    print(mean[:preview_values])

    print()
    print(f"Variance first {preview_values}:")
    print(variance[:preview_values])

    print()
    print("Scalar summaries:")
    print(f"mean(mean):           {mean.mean():.6f}")
    print(f"std(mean):            {mean.std():.6f}")
    print(f"mean(variance):       {variance.mean():.6f}")
    print(f"std(variance):        {variance.std():.6f}")
    print(f"trace(covariance):    {np.trace(covariance):.6f}")
    print(f"mean(covariance diag): {np.diag(covariance).mean():.6f}")

    print()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["global", "class"],
        required=True,
        help="global = all CIFAR-10 classes, class = selected CIFAR-10 class only",
    )

    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="CIFAR-10 class id. Required for --mode class.",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
    )

    parser.add_argument(
        "--split-a",
        type=str,
        default="train",
        choices=["train", "test"],
        help="First real split.",
    )

    parser.add_argument(
        "--split-b",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Second real split.",
    )

    parser.add_argument(
        "--max-a",
        type=int,
        default=None,
        help="Optional max images for split A.",
    )

    parser.add_argument(
        "--max-b",
        type=int,
        default=None,
        help="Optional max images for split B.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--save-stats",
        type=str,
        default=None,
        help="Optional .npz path to save mean, variance, covariance and FID.",
    )

    args = parser.parse_args()

    if args.mode == "class":
        if args.class_id is None:
            raise ValueError("--class-id is required for --mode class")

        if not 0 <= args.class_id <= 9:
            raise ValueError("--class-id must be between 0 and 9")

        class_id = args.class_id
        print(f"Mode: class-wise real-vs-real FID")
        print(f"Class: {class_id} = {LABELS[class_id]}")
    else:
        class_id = None
        print("Mode: global real-vs-real FID")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    images_a = load_cifar10_images(
        data_root=args.data_root,
        split=args.split_a,
        class_id=class_id,
        max_images=args.max_a,
        download=True,
    )

    images_b = load_cifar10_images(
        data_root=args.data_root,
        split=args.split_b,
        class_id=class_id,
        max_images=args.max_b,
        download=True,
    )

    print(f"{args.split_a} images: {tuple(images_a.shape)}")
    print(f"{args.split_b} images: {tuple(images_b.shape)}")

    features_a = extract_inception_features(
        images=images_a,
        batch_size=args.batch_size,
        device=device,
    )

    features_b = extract_inception_features(
        images=images_b,
        batch_size=args.batch_size,
        device=device,
    )

    mean_a, variance_a, covariance_a = calculate_statistics(features_a)
    mean_b, variance_b, covariance_b = calculate_statistics(features_b)

    fid = calculate_fid_from_stats(
        mean_1=mean_a,
        covariance_1=covariance_a,
        mean_2=mean_b,
        covariance_2=covariance_b,
    )

    print_statistics(
        name=args.split_a,
        mean=mean_a,
        variance=variance_a,
        covariance=covariance_a,
    )

    print_statistics(
        name=args.split_b,
        mean=mean_b,
        variance=variance_b,
        covariance=covariance_b,
    )

    print("=" * 80)
    if args.mode == "class":
        print(
            f"FID real-vs-real "
            f"[{args.split_a} vs {args.split_b}] "
            f"[class {class_id} - {LABELS[class_id]}]: {fid:.6f}"
        )
    else:
        print(f"FID real-vs-real [{args.split_a} vs {args.split_b}] [global]: {fid:.6f}")
    print("=" * 80)

    if args.save_stats is not None:
        np.savez(
            args.save_stats,
            mean_a=mean_a,
            variance_a=variance_a,
            covariance_a=covariance_a,
            mean_b=mean_b,
            variance_b=variance_b,
            covariance_b=covariance_b,
            fid=fid,
            split_a=args.split_a,
            split_b=args.split_b,
            class_id=-1 if class_id is None else class_id,
        )

        print(f"Saved statistics to: {args.save_stats}")


if __name__ == "__main__":
    main()