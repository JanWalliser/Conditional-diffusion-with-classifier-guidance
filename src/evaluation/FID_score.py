import argparse
import math
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.models import inception_v3, Inception_V3_Weights

from scipy.linalg import sqrtm


LABELS = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_images_from_directory(
    directory_path: str,
    class_id: Optional[int] = None,
    max_images: Optional[int] = None,
) -> torch.Tensor:
    root = Path(directory_path)

    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    if class_id is None:
        image_paths = [
            p for p in root.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    else:
        class_name = LABELS[class_id]

        possible_dirs = [
            root / str(class_id),
            root / class_name,
            root / f"class_{class_id}",
            root / f"{class_id}_{class_name}",
        ]

        class_dirs = [p for p in possible_dirs if p.exists() and p.is_dir()]

        if class_dirs:
            search_root = class_dirs[0]
            image_paths = [
                p for p in search_root.rglob("*")
                if p.suffix.lower() in IMAGE_EXTENSIONS
            ]
        else:
            image_paths = [
                p for p in root.rglob("*")
                if p.suffix.lower() in IMAGE_EXTENSIONS
                and (
                    f"class_{class_id}" in p.name.lower()
                    or f"label_{class_id}" in p.name.lower()
                    or class_name in p.name.lower()
                )
            ]

    image_paths = sorted(image_paths)

    if max_images is not None:
        image_paths = image_paths[:max_images]

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {root} for class_id={class_id}")

    to_tensor = transforms.ToTensor()
    images: List[torch.Tensor] = []

    for path in tqdm(image_paths, desc=f"Loading images from {root}"):
        img = Image.open(path).convert("RGB")
        images.append(to_tensor(img))

    return torch.stack(images, dim=0)


######TODO
def generate_images(
    num_images: int,
    class_id: Optional[int] = None,
) -> torch.Tensor:
    raise NotImplementedError(
        "Use --generated-dir for now, or connect your sampler here."
    )


def load_cifar10_images(
    data_root: str,
    split: str = "test",
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

    for img, label in tqdm(dataset, desc=f"Loading CIFAR-10 {split}"):
        if class_id is not None and label != class_id:
            continue

        images.append(img)

        if max_images is not None and len(images) >= max_images:
            break

    if len(images) == 0:
        raise RuntimeError(f"No CIFAR-10 images found for class_id={class_id}")

    return torch.stack(images, dim=0)


def calculate_statistics(features: np.ndarray):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


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
        return self.model(x)


@torch.no_grad()
def extract_inception_features(
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model = InceptionFeatureExtractor().to(device)
    model.eval()

    features = []

    for i in tqdm(range(0, images.shape[0], batch_size), desc="Extracting features"):
        batch = images[i:i + batch_size].to(device)
        feat = model(batch)
        features.append(feat.cpu())

    return torch.cat(features, dim=0).numpy()



def calculate_fid_from_features(
    real_features: np.ndarray,
    generated_features: np.ndarray,
    eps: float = 1e-6,
) -> float:
    mu_real, sigma_real = calculate_statistics(real_features)
    mu_gen, sigma_gen = calculate_statistics(generated_features)

    diff = mu_real - mu_gen

    # sqrtm can be numerically unstable if covariance matrices are nearly singular
    covmean = sqrtm(sigma_real @ sigma_gen)

    if not np.isfinite(covmean).all():
        print("FID calculation produced non-finite values. Adding epsilon to covariance diagonals.")

        offset = np.eye(sigma_real.shape[0]) * eps
        covmean = sqrtm((sigma_real + offset) @ (sigma_gen + offset))

    # sqrtm can return tiny imaginary components because of numerical error
    if np.iscomplexobj(covmean):
        imaginary_part = np.max(np.abs(covmean.imag))

        if imaginary_part > 1e-3:
            raise ValueError(f"Large imaginary component in sqrtm: {imaginary_part}")

        covmean = covmean.real

    fid = (
        diff.dot(diff)
        + np.trace(sigma_real)
        + np.trace(sigma_gen)
        - 2.0 * np.trace(covmean)
    )

    return float(fid)



def calculate_classwise_fid_score(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    batch_size: int = 64,
    device: str = "cuda",
) -> float:
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    real_features = extract_inception_features(real_images, batch_size, device)
    generated_features = extract_inception_features(generated_images, batch_size, device)

    return calculate_fid_from_features(real_features, generated_features)




def calculate_global_fid_score(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    batch_size: int = 64,
    device: str = "cuda",
) -> float:
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    real_features = extract_inception_features(real_images, batch_size, device)
    generated_features = extract_inception_features(generated_images, batch_size, device)

    return calculate_fid_from_features(real_features, generated_features)





def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["global", "class"],
        required=True,
        help="global = all classes, class = one selected CIFAR-10 class",
    )

    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="CIFAR-10 class id, required for --mode class",
    )

    parser.add_argument(
        "--real-source",
        choices=["cifar10", "directory"],
        default="cifar10",
        help="Where real images come from.",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="CIFAR-10 data root if --real-source cifar10.",
    )

    parser.add_argument(
        "--real-dir",
        type=str,
        default=None,
        help="Directory with real images if --real-source directory.",
    )

    parser.add_argument(
        "--generated-dir",
        type=str,
        default=None,
        help="Directory with generated images. If omitted, generate_images() is called.",
    )

    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help="CIFAR-10 split for real images.",
    )

    parser.add_argument(
        "--max-real",
        type=int,
        default=None,
        help="Optional limit for real images.",
    )

    parser.add_argument(
        "--max-generated",
        type=int,
        default=None,
        help="Optional limit for generated images.",
    )

    parser.add_argument(
        "--num-generate",
        type=int,
        default=1000,
        help="Number of images to generate if --generated-dir is omitted.",
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

    args = parser.parse_args()

    if args.mode == "class":
        if args.class_id is None:
            raise ValueError("--class-id is required when --mode class")

        if not 0 <= args.class_id <= 9:
            raise ValueError("--class-id must be between 0 and 9")

        print(f"Mode: class-wise FID")
        print(f"Class: {args.class_id} = {LABELS[args.class_id]}")
    else:
        print("Mode: global FID")

    class_id = args.class_id if args.mode == "class" else None

    # Load real images
    if args.real_source == "cifar10":
        real_images = load_cifar10_images(
            data_root=args.data_root,
            split=args.split,
            class_id=class_id,
            max_images=args.max_real,
            download=True,
        )
    else:
        if args.real_dir is None:
            raise ValueError("--real-dir is required when --real-source directory")

        real_images = load_images_from_directory(
            directory_path=args.real_dir,
            class_id=class_id,
            max_images=args.max_real,
        )

    # Load or generate fake images
    if args.generated_dir is not None:
        generated_images = load_images_from_directory(
            directory_path=args.generated_dir,
            class_id=class_id,
            max_images=args.max_generated,
        )
    else:
        generated_images = generate_images(
            num_images=args.num_generate,
            class_id=class_id,
        )

    print(f"Real images:      {tuple(real_images.shape)}")
    print(f"Generated images: {tuple(generated_images.shape)}")

    if args.mode == "global":
        fid = calculate_global_fid_score(
            real_images=real_images,
            generated_images=generated_images,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        fid = calculate_classwise_fid_score(
            real_images=real_images,
            generated_images=generated_images,
            batch_size=args.batch_size,
            device=args.device,
        )

    print("-" * 80)
    if args.mode == "global":
        print(f"Global FID: {fid:.4f}")
    else:
        print(
            f"Class-wise FID "
            f"[{args.class_id} - {LABELS[args.class_id]}]: {fid:.4f}"
        )
    print("-" * 80)


if __name__ == "__main__":
    main()