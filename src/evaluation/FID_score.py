import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
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

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# -----------------------------------------------------------------------------
# Loading images
# -----------------------------------------------------------------------------

def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def find_class_dir(root: Path, class_id: int) -> Optional[Path]:
    class_name = LABELS[class_id]
    candidates = [
        root / str(class_id),
        root / class_name,
        root / f"class_{class_id}",
        root / f"label_{class_id}",
        root / f"{class_id}_{class_name}",
        root / f"class_{class_id}_{class_name}",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    return None


def resolve_scale_dir(generated_root: Path, scale: int) -> Path:
    candidates = [
        generated_root / f"scale_{scale}",
        generated_root / f"guidance_{scale}",
        generated_root / f"guidance_scale_{scale}",
        generated_root / f"s{scale}",
        generated_root / str(scale),
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not find generated directory for guidance scale "
        f"{scale}. Tried: {', '.join(str(p) for p in candidates)}"
    )


def load_images_from_directory(
    directory_path: str,
    class_id: Optional[int] = None,
    max_images: Optional[int] = None,
    expected_size: tuple[int, int] = (32, 32),
) -> torch.Tensor:
    root = Path(directory_path)

    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    if class_id is None:
        search_roots = [root]
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
            search_roots = [class_dirs[0]]
        else:
            search_roots = [root]

    image_paths = []
    for search_root in search_roots:
        for p in search_root.rglob("*"):
            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            name = p.name.lower()
            if any(x in name for x in ["grid", "preview", "sample_grid", "overview"]):
                continue

            if class_id is not None:
                class_name = LABELS[class_id]
                parent_text = str(p.parent).lower()
                file_text = p.name.lower()

                matches_class = (
                    f"class_{class_id}" in parent_text
                    or f"class_{class_id}" in file_text
                    or f"label_{class_id}" in file_text
                    or f"{class_id}_{class_name}" in parent_text
                    or class_name in parent_text
                    or class_name in file_text
                )

                if not class_dirs and not matches_class:
                    continue

            image_paths.append(p)

    image_paths = sorted(image_paths)

    to_tensor = transforms.ToTensor()
    images: List[torch.Tensor] = []
    skipped = 0

    for path in tqdm(image_paths, desc=f"Loading images from {root}"):
        img = Image.open(path).convert("RGB")

        if img.size != expected_size:
            skipped += 1
            continue

        images.append(to_tensor(img))

        if max_images is not None and len(images) >= max_images:
            break

    if len(images) == 0:
        raise RuntimeError(
            f"No valid {expected_size} images found in {root} for class_id={class_id}. "
            f"Skipped {skipped} non-matching images."
        )

    if skipped > 0:
        print(f"Skipped {skipped} non-{expected_size} images in {root}")

    return torch.stack(images, dim=0)


def load_cifar10_images(
    data_root: str,
    split: str,
    class_id: Optional[int] = None,
    max_images: Optional[int] = None,
    download: bool = True,
) -> torch.Tensor:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    dataset = datasets.CIFAR10(
        root=data_root,
        train=(split == "train"),
        download=download,
        transform=transforms.ToTensor(),
    )

    images: List[torch.Tensor] = []

    for img, label in tqdm(dataset, desc=f"Loading CIFAR-10 {split}"):
        if class_id is not None and label != class_id:
            continue

        images.append(img)

        if max_images is not None and len(images) >= max_images:
            break

    if len(images) == 0:
        raise RuntimeError(f"No CIFAR-10 images found for split={split}, class_id={class_id}")

    return torch.stack(images, dim=0)


# -----------------------------------------------------------------------------
# FID
# -----------------------------------------------------------------------------

class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights, transform_input=False)
        model.fc = nn.Identity()
        model.eval()

        self.model = model
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.model(x)


@torch.no_grad()
def extract_inception_features(
    extractor: InceptionFeatureExtractor,
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    features: List[torch.Tensor] = []

    for i in tqdm(range(0, images.shape[0], batch_size), desc="Extracting features"):
        batch = images[i:i + batch_size].to(device)
        features.append(extractor(batch).cpu())

    return torch.cat(features, dim=0).numpy()


def calculate_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_fid_from_features(
    features_a: np.ndarray,
    features_b: np.ndarray,
    eps: float = 1e-6,
    smoke_mode: bool = False,
) -> float:
    """Calculate FID from feature arrays.

    A real FID estimate needs enough samples to estimate covariance matrices.
    For the reproducibility smoke test we intentionally use only one image, so
    smoke_mode falls back to the squared mean-feature distance and skips the
    covariance/sqrtm term. That number is only a pipeline sanity check, not a
    reportable FID value.
    """
    if features_a.ndim != 2 or features_b.ndim != 2:
        raise ValueError("Expected feature arrays with shape (num_images, feature_dim).")

    if features_a.shape[0] < 2 or features_b.shape[0] < 2:
        if not smoke_mode:
            raise ValueError(
                "FID covariance estimation requires at least 2 images per set. "
                "Use more samples for real evaluation, or pass --smoke-mode for "
                "the 1-image reproducibility pipeline check."
            )

        mu_a = np.mean(features_a, axis=0)
        mu_b = np.mean(features_b, axis=0)
        diff = mu_a - mu_b
        print(
            "Smoke-mode FID fallback: fewer than 2 images in at least one set; "
            "reporting squared mean-feature distance only, not real FID."
        )
        return float(diff.dot(diff))

    mu_a, sigma_a = calculate_statistics(features_a)
    mu_b, sigma_b = calculate_statistics(features_b)

    diff = mu_a - mu_b
    covmean = sqrtm(sigma_a @ sigma_b)

    if not np.isfinite(covmean).all():
        print("FID produced non-finite values. Adding epsilon to covariance diagonals.")
        offset = np.eye(sigma_a.shape[0]) * eps
        covmean = sqrtm((sigma_a + offset) @ (sigma_b + offset))

    if np.iscomplexobj(covmean):
        imaginary_part = np.max(np.abs(covmean.imag))
        if imaginary_part > 1e-3:
            raise ValueError(f"Large imaginary component in sqrtm: {imaginary_part}")
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_a) + np.trace(sigma_b) - 2.0 * np.trace(covmean)
    return float(fid)


# -----------------------------------------------------------------------------
# Evaluation modes
# -----------------------------------------------------------------------------

def get_features_cached(
    cache: Dict[str, np.ndarray],
    key: str,
    images: torch.Tensor,
    extractor: InceptionFeatureExtractor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if key not in cache:
        print(f"\nComputing features: {key} | images={tuple(images.shape)}")
        cache[key] = extract_inception_features(extractor, images, batch_size, device)
    return cache[key]


def evaluate_matrix(args) -> List[Dict[str, object]]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    extractor = InceptionFeatureExtractor().to(device).eval()
    feature_cache: Dict[str, np.ndarray] = {}
    results: List[Dict[str, object]] = []

    classes = parse_int_list(args.class_ids)
    scales = parse_int_list(args.guidance_scales)
    generated_root = Path(args.generated_root)

    for class_id in classes:
        class_name = LABELS[class_id]
        print("\n" + "=" * 80)
        print(f"Class {class_id}: {class_name}")
        print("=" * 80)

        train_images = load_cifar10_images(
            data_root=args.data_root,
            split="train",
            class_id=class_id,
            max_images=args.max_real,
            download=True,
        )
        test_images = load_cifar10_images(
            data_root=args.data_root,
            split="test",
            class_id=class_id,
            max_images=args.max_real,
            download=True,
        )

        train_features = get_features_cached(
            feature_cache, f"cifar10_train_class_{class_id}", train_images,
            extractor, args.batch_size, device,
        )
        test_features = get_features_cached(
            feature_cache, f"cifar10_test_class_{class_id}", test_images,
            extractor, args.batch_size, device,
        )

        train_test_fid = calculate_fid_from_features(
            train_features, test_features, smoke_mode=args.smoke_mode
        )
        results.append({
            "class_id": class_id,
            "class_name": class_name,
            "comparison": "train_vs_test",
            "scale": "real",
            "num_a": len(train_images),
            "num_b": len(test_images),
            "fid": train_test_fid,
        })
        print(f"FID train vs test [{class_id} - {class_name}]: {train_test_fid:.4f}")

        for scale in scales:
            scale_dir = resolve_scale_dir(generated_root, scale)
            generated_images = load_images_from_directory(
                directory_path=scale_dir,
                class_id=class_id,
                max_images=args.max_generated,
            )
            generated_features = get_features_cached(
                feature_cache, f"generated_scale_{scale}_class_{class_id}", generated_images,
                extractor, args.batch_size, device,
            )

            fid = calculate_fid_from_features(
                test_features, generated_features, smoke_mode=args.smoke_mode
            )
            results.append({
                "class_id": class_id,
                "class_name": class_name,
                "comparison": "test_vs_generated",
                "scale": scale,
                "num_a": len(test_images),
                "num_b": len(generated_images),
                "fid": fid,
            })
            print(f"FID test vs generated scale={scale} [{class_id} - {class_name}]: {fid:.4f}")

    return results


def print_results_table(results: List[Dict[str, object]]) -> None:
    print("\n" + "=" * 80)
    print("FID SUMMARY")
    print("=" * 80)
    print(f"{'class':<16} {'comparison':<20} {'scale':<8} {'num_a':>8} {'num_b':>8} {'FID':>12}")
    print("-" * 80)

    for row in results:
        class_label = f"{row['class_id']}:{row['class_name']}"
        print(
            f"{class_label:<16} "
            f"{row['comparison']:<20} "
            f"{str(row['scale']):<8} "
            f"{int(row['num_a']):>8} "
            f"{int(row['num_b']):>8} "
            f"{float(row['fid']):>12.4f}"
        )


def save_csv(results: List[Dict[str, object]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["class_id", "class_name", "comparison", "scale", "num_a", "num_b", "fid"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved CSV: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval-mode",
        choices=["matrix"],
        default="matrix",
        help="matrix = train/test sanity baseline plus test/generated FID per class and guidance scale.",
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--generated-root", type=str, required=True)
    parser.add_argument("--class-ids", type=str, default="0,1,5,8")
    parser.add_argument("--guidance-scales", type=str, default="0,5,10")
    parser.add_argument("--max-real", type=int, default=None)
    parser.add_argument("--max-generated", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-csv", type=str, default="outputs/fid_results.csv")
    parser.add_argument(
        "--smoke-mode",
        action="store_true",
        help=(
            "Allow one-image smoke tests by skipping covariance/sqrtm and "
            "reporting squared mean-feature distance. Not a reportable FID."
        ),
    )

    args = parser.parse_args()

    results = evaluate_matrix(args)
    print_results_table(results)
    save_csv(results, args.output_csv)


if __name__ == "__main__":
    main()
