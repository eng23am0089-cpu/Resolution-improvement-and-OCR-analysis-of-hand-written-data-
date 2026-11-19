from datasets import load_dataset
import os
import csv


# ============================================================
# SETTINGS
# ============================================================

DATASET_NAME = "Teklia/IAM-line"

# Number of handwritten samples we want initially
NUMBER_OF_SAMPLES = 100


# ============================================================
# PATHS
# ============================================================

# Current folder = backend
BACKEND_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)

# Project folder = handwriting-ocr
PROJECT_FOLDER = os.path.dirname(
    BACKEND_FOLDER
)

# Dataset folder
DATASET_FOLDER = os.path.join(
    PROJECT_FOLDER,
    "dataset",
    "iam_sample"
)

IMAGE_FOLDER = os.path.join(
    DATASET_FOLDER,
    "images"
)

LABEL_FILE = os.path.join(
    DATASET_FOLDER,
    "labels.csv"
)


# ============================================================
# CREATE FOLDERS
# ============================================================

os.makedirs(
    IMAGE_FOLDER,
    exist_ok=True
)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("IAM HANDWRITING DATASET DOWNLOADER")
    print("=" * 60)
    print()

    print(
        f"Dataset: {DATASET_NAME}"
    )

    print(
        f"Samples to download: {NUMBER_OF_SAMPLES}"
    )

    print()

    print(
        "Loading IAM dataset..."
    )

    print(
        "This may take a little time the first time."
    )

    print()

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True
    )

    print(
        "Dataset loaded successfully."
    )

    print()

    # --------------------------------------------------------
    # CSV FILE
    # --------------------------------------------------------

    with open(
        LABEL_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        writer.writerow(
            [
                "image",
                "text"
            ]
        )

        # ----------------------------------------------------
        # DOWNLOAD SAMPLES
        # ----------------------------------------------------

        count = 0

        for example in dataset:

            if count >= NUMBER_OF_SAMPLES:
                break

            image = example["image"]
            text = example["text"]

            if image is None:
                continue

            if text is None:
                continue

            text = str(
                text
            ).strip()

            if not text:
                continue

            filename = (
                f"line_{count + 1:03d}.png"
            )

            image_path = os.path.join(
                IMAGE_FOLDER,
                filename
            )

            # Save image
            image.save(
                image_path
            )

            # Save image filename + ground truth
            writer.writerow(
                [
                    filename,
                    text
                ]
            )

            count += 1

            print(
                f"[{count}/{NUMBER_OF_SAMPLES}] "
                f"{filename} -> {text}"
            )

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DATASET EXTRACTION COMPLETED")
    print("=" * 60)
    print()

    print(
        f"Images saved: {IMAGE_FOLDER}"
    )

    print(
        f"Labels saved: {LABEL_FILE}"
    )

    print()

    print(
        f"Total samples: {count}"
    )

    print()

    print(
        "Dataset structure:"
    )

    print(
        "dataset/"
    )

    print(
        "└── iam_sample/"
    )

    print(
        "    ├── images/"
    )

    print(
        "    │   ├── line_001.png"
    )

    print(
        "    │   ├── line_002.png"
    )

    print(
        "    │   └── ..."
    )

    print(
        "    └── labels.csv"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()