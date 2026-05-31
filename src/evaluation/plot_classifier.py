import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def read_timestep_accuracy_csv(csv_path: Path) -> list[tuple[list[int], list[float]]]:
    """
    Reads a CSV file in the format:

        timestep,accuracy
        1,0.8532
        50,0.8346
        ...

    If the file contains multiple runs appended after each other, the function
    starts a new run whenever the timestep decreases or resets.
    """
    runs: list[tuple[list[int], list[float]]] = []

    current_timesteps: list[int] = []
    current_accuracies: list[float] = []
    previous_timestep: int | None = None

    with csv_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line.lower().startswith("timestep"):
            continue

        timestep_str, accuracy_str = line.split(",")

        timestep = int(timestep_str.strip())
        accuracy = float(accuracy_str.strip())

        if previous_timestep is not None and timestep <= previous_timestep:
            runs.append((current_timesteps, current_accuracies))
            current_timesteps = []
            current_accuracies = []

        current_timesteps.append(timestep)
        current_accuracies.append(accuracy)

        previous_timestep = timestep

    if current_timesteps:
        runs.append((current_timesteps, current_accuracies))

    return runs


def plot_timestep_accuracy(
    csv_path: Path,
    output_path: Path,
) -> None:
    runs = read_timestep_accuracy_csv(csv_path)

    if not runs:
        raise ValueError(f"No valid data found in CSV file: {csv_path}")

    plt.figure(figsize=(9, 5))

    labels = [
    "Accuracy Sine Sampling",
    "Accuracy Linear Decay Sampling",
]
    for idx, (timesteps, accuracies) in enumerate(runs, start=1):
       
        

        plt.plot(
            timesteps,
            accuracies,
            marker="o",
            linewidth=2,
            markersize=4,
            label=labels[idx-1],
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
    plt.show()

    print(f"Saved plot to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot noisy-image classifier accuracy over exact diffusion timesteps."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="outputs/classifier_timestep_accuracy.csv",
        help="Path to CSV file with columns: timestep,accuracy",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/classifier_timestep_accuracy.png",
        help="Path where the plot image should be saved.",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_path = Path(args.out)

    plot_timestep_accuracy(
        csv_path=csv_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
