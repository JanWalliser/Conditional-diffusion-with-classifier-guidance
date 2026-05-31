from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = "accuracy" in sample.splitlines()[0].lower()
        if has_header:
            return list(csv.DictReader(f))

        rows = []
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                rows.append({"timestep": row[0], "accuracy": row[1]})
        return rows


def read_timestep_accuracy_csv(csv_path: Path) -> tuple[list[int], list[float]]:
    """
    Read classifier timestep accuracy CSVs.

    Supports both old two-column files:
        timestep,accuracy

    and current training output:
        epoch,timestep,accuracy

    For epoch-based CSVs, only the final epoch is plotted because that is the
    report-relevant classifier curve.
    """
    rows = _read_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows found in CSV: {csv_path}")

    normalized = []
    for row in rows:
        lower = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
        if "accuracy" not in lower or "timestep" not in lower:
            continue

        try:
            epoch = int(float(lower.get("epoch", "0")))
            timestep = int(float(lower["timestep"]))
            accuracy = float(lower["accuracy"])
        except ValueError:
            continue

        normalized.append((epoch, timestep, accuracy))

    if not normalized:
        raise ValueError(f"No valid timestep/accuracy rows found in CSV: {csv_path}")

    max_epoch = max(epoch for epoch, _, _ in normalized)
    final_rows = [(t, a) for epoch, t, a in normalized if epoch == max_epoch]
    final_rows.sort(key=lambda item: item[0])

    timesteps = [t for t, _ in final_rows]
    accuracies = [a for _, a in final_rows]
    return timesteps, accuracies


def plot_timestep_accuracy(
    csv_paths: list[Path],
    output_path: Path,
    labels: list[str] | None = None,
) -> None:
    if labels is None or len(labels) == 0:
        labels = [path.stem for path in csv_paths]

    if len(labels) != len(csv_paths):
        raise ValueError("Number of --label entries must match number of --csv entries.")

    plt.figure(figsize=(9, 5))

    for csv_path, label in zip(csv_paths, labels):
        timesteps, accuracies = read_timestep_accuracy_csv(csv_path)
        plt.plot(
            timesteps,
            accuracies,
            marker="o",
            linewidth=2,
            markersize=4,
            label=label,
        )

    plt.xlabel("Diffusion timestep t")
    plt.ylabel("Validation accuracy")
    plt.title("Noisy-image classifier accuracy by exact diffusion timestep")
    plt.xlim(0, 1000)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot noisy-image classifier accuracy over exact diffusion timesteps."
    )
    parser.add_argument(
        "--csv",
        type=str,
        action="append",
        required=True,
        help="Path to a CSV with timestep accuracy. Can be passed multiple times.",
    )
    parser.add_argument(
        "--label",
        type=str,
        action="append",
        default=None,
        help="Legend label. Can be passed once per --csv.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/classifier_timestep_accuracy.png",
        help="Path where the plot image should be saved.",
    )

    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.csv]
    output_path = Path(args.out)

    plot_timestep_accuracy(
        csv_paths=csv_paths,
        output_path=output_path,
        labels=args.label,
    )


if __name__ == "__main__":
    main()
