"""
infer.py - Interactive MNIST inference visualizer:
1. Picks a test sample (by index or randomly).
2. Calls the pure-Rust neural network to run forward inference.
3. Displays the image and Rust probability distribution in a Python GUI window.
"""

import sys
import random
import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Resolve repository root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "python_mnist" else Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "MNIST" / "raw"

def load_sample(index: int):
    # Read the exact same raw binary files
    img_path = DATA_DIR / "t10k-images-idx3-ubyte"
    lbl_path = DATA_DIR / "t10k-labels-idx1-ubyte"
    with open(img_path, "rb") as f:
        f.seek(16 + index * 784)
        img_bytes = f.read(784)
    with open(lbl_path, "rb") as f:
        f.seek(8 + index)
        lbl_byte = f.read(1)

    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(28, 28)
    label = int(lbl_byte[0])
    return img, label

def main():
    if len(sys.argv) > 1:
        try:
            index = int(sys.argv[1])
        except ValueError:
            print("Usage: python infer.py [sample_index (0-9999)]")
            sys.exit(1)
    else:
        index = random.randint(0, 9999)

    print(f"Loading test sample #{index}...")
    img, actual = load_sample(index)

    print("Running inference with pure-Rust neural network...")
    result = subprocess.run(
        ["cargo", "run", "--release", "--bin", "mnist", "--", str(index)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    print(result.stdout)

    # Parse prediction, actual, confidence, and class probabilities from Rust output
    pred = None
    confidence = None
    probs = [0.0] * 10

    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("Digit ") and ":" in line:
            # e.g. "Digit 7: 100.0% | ..."
            parts = line.split(":")
            digit = int(parts[0].replace("Digit", "").strip())
            pct = float(parts[1].split("%")[0].strip())
            probs[digit] = pct
        elif line.startswith("PREDICTION:"):
            pred = int(line.split(":")[1].strip())
        elif line.startswith("CONFIDENCE:"):
            confidence = line.split(":")[1].strip()

    # Create visual figure
    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(8, 4))
    fig.canvas.manager.set_window_title(f"Rust MNIST Inference - Sample #{index}")

    # Left: Handwritten Image
    ax_img.imshow(img, cmap="gray")
    status = "CORRECT" if pred == actual else "INCORRECT"
    color = "green" if pred == actual else "red"
    ax_img.set_title(
        f"Test Image #{index}\nActual: {actual} | Rust Pred: {pred} ({status})",
        fontsize=12,
        fontweight="bold",
        color=color,
    )
    ax_img.axis("off")

    # Right: Probability Distribution from Rust
    digits = list(range(10))
    bar_colors = ["#2ecc71" if d == pred else "#3498db" for d in digits]
    bars = ax_bar.barh(digits, probs, color=bar_colors, edgecolor="black", height=0.6)
    ax_bar.set_yticks(digits)
    ax_bar.set_yticklabels([f"Digit {d}" for d in digits], fontsize=10)
    ax_bar.set_xlabel("Confidence (%)", fontsize=11)
    ax_bar.set_xlim(0, 105)
    ax_bar.set_title(f"Rust Model Output\nTop Confidence: {confidence}", fontsize=12, fontweight="bold")
    ax_bar.invert_yaxis()  # 0 at top

    # Add text labels on bars
    for bar in bars:
        width = bar.get_width()
        if width > 1.0:
            ax_bar.text(width + 1.5, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    print("Displaying interactive visualization window...")
    plt.show()

if __name__ == "__main__":
    main()
