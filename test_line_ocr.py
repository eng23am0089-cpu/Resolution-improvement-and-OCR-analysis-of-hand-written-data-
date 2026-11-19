import os
import time

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "microsoft/trocr-small-handwritten"

BACKEND_FOLDER = os.path.dirname(os.path.abspath(__file__))

INPUT_FOLDER = os.path.join(
    BACKEND_FOLDER,
    "processed",
    "line_detection_test_v4",
    "lines"
)


# ============================================================
# LOAD TrOCR
# ============================================================

def load_model():

    print("=" * 60)
    print("LOADING TrOCR")
    print("=" * 60)

    print(f"Model: {MODEL_NAME}")
    print()

    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)

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
# OCR ONE LINE
# ============================================================

def recognize_line(
    image_path,
    processor,
    model,
    device
):

    image = Image.open(image_path).convert("RGB")

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values

    pixel_values = pixel_values.to(device)

    with torch.no_grad():

        generated_ids = model.generate(
            pixel_values,
            max_length=64,
            num_beams=4,
            early_stopping=True
        )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return text.strip()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("HANDWRITING LINE OCR TEST - VERSION 4")
    print("=" * 60)
    print()

    print("Input folder:")
    print(INPUT_FOLDER)
    print()

    # --------------------------------------------------------
    # Check folder
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FOLDER):

        print("ERROR: Line folder does not exist.")

        print()
        print("Expected folder:")
        print(INPUT_FOLDER)

        return

    # --------------------------------------------------------
    # Find line images
    # --------------------------------------------------------

    image_files = []

    for filename in os.listdir(INPUT_FOLDER):

        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):

            image_files.append(filename)

    image_files.sort()

    if not image_files:

        print("ERROR: No line images found.")

        return

    print(
        f"Line images found: {len(image_files)}"
    )

    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    processor, model, device = load_model()

    # --------------------------------------------------------
    # OCR each line
    # --------------------------------------------------------

    print("=" * 60)
    print("RUNNING OCR ON DETECTED LINES")
    print("=" * 60)
    print()

    results = []

    for index, filename in enumerate(
        image_files,
        start=1
    ):

        image_path = os.path.join(
            INPUT_FOLDER,
            filename
        )

        print(
            f"LINE {index}/{len(image_files)}"
        )

        print(
            f"Image: {filename}"
        )

        start_time = time.time()

        try:

            text = recognize_line(
                image_path,
                processor,
                model,
                device
            )

            processing_time = (
                time.time() - start_time
            )

            print(
                f"OCR: {text}"
            )

            print(
                f"Time: {processing_time:.2f} seconds"
            )

            results.append(
                {
                    "image": filename,
                    "text": text
                }
            )

        except Exception as error:

            print(
                f"OCR ERROR: {error}"
            )

        print("-" * 60)

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL OCR RESULTS")
    print("=" * 60)
    print()

    for result in results:

        print(
            f"{result['image']} -> "
            f"{result['text']}"
        )

    print()
    print("=" * 60)
    print("LINE OCR TEST COMPLETED")
    print("=" * 60)
    print()

    print("IMPORTANT:")
    print("ocr_service.py has NOT been modified.")
    print("This was only a separate OCR test.")
    print()


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()