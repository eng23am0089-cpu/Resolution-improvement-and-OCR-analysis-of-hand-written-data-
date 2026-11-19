import os
import csv
import time

import torch
from PIL import Image
from jiwer import wer, cer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "microsoft/trocr-small-handwritten"

NUMBER_OF_TEST_IMAGES = 100


# ============================================================
# PROJECT PATHS
# ============================================================

BACKEND_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_FOLDER = os.path.dirname(
    BACKEND_FOLDER
)

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
# LOAD TrOCR MODEL
# ============================================================

def load_model():

    print()
    print("=" * 60)
    print("LOADING TrOCR")
    print("=" * 60)

    print(f"Model: {MODEL_NAME}")
    print()

    processor = TrOCRProcessor.from_pretrained(
        MODEL_NAME
    )

    model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_NAME
    )

    if torch.cuda.is_available():

        device = torch.device("cuda")

        model.to(device)

        print("Device: GPU")

    else:

        device = torch.device("cpu")

        model.to(device)

        print("Device: CPU")

    model.eval()

    print()
    print("TrOCR loaded successfully.")
    print()

    return processor, model, device


# ============================================================
# LOAD DATASET LABELS
# ============================================================

def load_labels():

    if not os.path.exists(LABEL_FILE):

        raise FileNotFoundError(
            f"Labels file not found:\n{LABEL_FILE}"
        )

    labels = []

    with open(
        LABEL_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            image_name = row["image"].strip()

            text = row["text"].strip()

            if not image_name or not text:
                continue

            labels.append(
                {
                    "image": image_name,
                    "text": text
                }
            )

    return labels


# ============================================================
# RECOGNIZE ONE IMAGE
# ============================================================

def recognize_image(
    image_path,
    processor,
    model,
    device
):

    image = Image.open(
        image_path
    ).convert("RGB")

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values

    pixel_values = pixel_values.to(
        device
    )

    with torch.no_grad():

        generated_ids = model.generate(
            pixel_values,
            max_length=64,
            num_beams=4,
            early_stopping=True
        )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return generated_text.strip()


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print()
    print("=" * 60)
    print("IAM TrOCR TEST")
    print("=" * 60)
    print()

    print("Dataset folder:")
    print(DATASET_FOLDER)
    print()

    # --------------------------------------------------------
    # LOAD LABELS
    # --------------------------------------------------------

    labels = load_labels()

    print(
        f"Total available samples: {len(labels)}"
    )

    print()

    if len(labels) == 0:

        print("No dataset samples found.")

        return


    # --------------------------------------------------------
    # SELECT TEST SAMPLES
    # --------------------------------------------------------

    test_samples = labels[
        :NUMBER_OF_TEST_IMAGES
    ]

    print(
        f"Testing samples: {len(test_samples)}"
    )

    print()


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    processor, model, device = load_model()


    # --------------------------------------------------------
    # RUN OCR
    # --------------------------------------------------------

    print("=" * 60)
    print("RUNNING OCR TEST")
    print("=" * 60)
    print()

    results = []


    for index, sample in enumerate(
        test_samples,
        start=1
    ):

        image_name = sample["image"]

        reference_text = sample["text"]

        image_path = os.path.join(
            IMAGE_FOLDER,
            image_name
        )


        # ----------------------------------------------------
        # CHECK IMAGE
        # ----------------------------------------------------

        if not os.path.exists(image_path):

            print(
                f"[{index}/{len(test_samples)}] "
                f"Image not found: {image_name}"
            )

            continue


        print(
            f"Test {index}/{len(test_samples)}"
        )

        print(
            f"Image : {image_name}"
        )

        print(
            f"Actual: {reference_text}"
        )


        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        start_time = time.time()

        predicted_text = recognize_image(
            image_path,
            processor,
            model,
            device
        )

        processing_time = (
            time.time() - start_time
        )


        print(
            f"OCR   : {predicted_text}"
        )

        print(
            f"Time  : {processing_time:.2f} seconds"
        )

        print("-" * 60)


        # ----------------------------------------------------
        # STORE RESULT
        # ----------------------------------------------------

        results.append(
            {
                "reference": reference_text,
                "prediction": predicted_text
            }
        )


    # --------------------------------------------------------
    # CHECK RESULTS
    # --------------------------------------------------------

    if not results:

        print()
        print("No OCR results available.")

        return


    # --------------------------------------------------------
    # CALCULATE WER / CER
    # --------------------------------------------------------

    references = [
        item["reference"]
        for item in results
    ]

    predictions = [
        item["prediction"]
        for item in results
    ]


    word_error_rate = wer(
        references,
        predictions
    )

    character_error_rate = cer(
        references,
        predictions
    )


    # --------------------------------------------------------
    # CALCULATE ACCURACY
    # --------------------------------------------------------

    word_accuracy = max(
        0,
        (1 - word_error_rate) * 100
    )

    character_accuracy = max(
        0,
        (1 - character_error_rate) * 100
    )


    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("IAM TrOCR TEST RESULTS")
    print("=" * 60)

    print()

    print(
        f"Samples tested          : {len(results)}"
    )

    print(
        f"Word Error Rate         : "
        f"{word_error_rate * 100:.2f}%"
    )

    print(
        f"Character Error Rate    : "
        f"{character_error_rate * 100:.2f}%"
    )

    print(
        f"Word Accuracy           : "
        f"{word_accuracy:.2f}%"
    )

    print(
        f"Character Accuracy     : "
        f"{character_accuracy:.2f}%"
    )

    print()

    print("=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)

    print()


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()