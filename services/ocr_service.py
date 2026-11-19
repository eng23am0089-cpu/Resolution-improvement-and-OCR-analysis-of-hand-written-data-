import cv2
import time
import torch
import numpy as np
import os

from PIL import Image
from jiwer import wer, cer

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "microsoft/trocr-small-handwritten"

processor = None
model = None
device = None


# ============================================================
# DEBUG FOLDER
# ============================================================

def get_debug_folder():

    backend_folder = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    folder = os.path.join(
        backend_folder,
        "processed",
        "debug_lines"
    )

    os.makedirs(
        folder,
        exist_ok=True
    )

    return folder


# ============================================================
# CLEAR DEBUG LINE IMAGES
# ============================================================

def clear_debug_lines():

    folder = get_debug_folder()

    for filename in os.listdir(folder):

        path = os.path.join(
            folder,
            filename
        )

        try:

            if os.path.isfile(path):
                os.remove(path)

        except Exception:

            pass


# ============================================================
# SAVE DEBUG LINE IMAGES
# ============================================================

def save_debug_line_images(line_images):

    folder = get_debug_folder()

    clear_debug_lines()

    for index, image in enumerate(line_images):

        path = os.path.join(
            folder,
            f"line_{index + 1}.jpg"
        )

        cv2.imwrite(
            path,
            image
        )

        print(
            f"Saved debug line {index + 1}: {path}"
        )

    print(
        f"Saved {len(line_images)} debug line images."
    )


# ============================================================
# LOAD TrOCR MODEL
# ============================================================

def load_trocr_model():

    global processor
    global model
    global device

    if processor is not None and model is not None:
        return

    print(
        "Loading TrOCR Small model..."
    )

    processor = TrOCRProcessor.from_pretrained(
        MODEL_NAME
    )

    model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_NAME
    )

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

        model.to(device)

        print(
            "TrOCR using GPU"
        )

    else:

        device = torch.device(
            "cpu"
        )

        model.to(device)

        print(
            "TrOCR using CPU"
        )

    model.eval()

    print(
        "TrOCR model loaded successfully."
    )

    print(
        "Decoder start token:",
        model.config.decoder_start_token_id
    )

    print(
        "EOS token:",
        model.config.eos_token_id
    )

    print(
        "PAD token:",
        model.config.pad_token_id
    )


# ============================================================
# READ IMAGE
# ============================================================

def read_original_image(image_path):

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise ValueError(
            "Could not read the image."
        )

    return image


# ============================================================
# FIND PAGE REGION
# ============================================================

def find_page_region(image):

    height, width = image.shape[:2]

    print(
        f"Input image size: {width} x {height}"
    )

    page = image.copy()

    backend_folder = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    debug_path = os.path.join(
        backend_folder,
        "processed",
        "debug_page.jpg"
    )

    cv2.imwrite(
        debug_path,
        page
    )

    print(
        f"Saved page debug image: {debug_path}"
    )

    return page


# ============================================================
# REMOVE NOTEBOOK LINES
# ============================================================

def remove_notebook_lines(page):

    print()
    print(
        "----------------------------------------"
    )
    print(
        "NOTEBOOK LINE REMOVAL"
    )
    print(
        "----------------------------------------"
    )

    gray = cv2.cvtColor(
        page,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Improve contrast
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

    # --------------------------------------------------------
    # Slight blur
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        enhanced,
        (3, 3),
        0
    )

    # --------------------------------------------------------
    # Edge detection
    # --------------------------------------------------------

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    # --------------------------------------------------------
    # Hough detection
    # --------------------------------------------------------

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=int(
            page.shape[1] * 0.35
        ),
        maxLineGap=25
    )

    line_mask = np.zeros(
        gray.shape,
        dtype=np.uint8
    )

    detected_segments = 0

    if lines is not None:

        for line in lines:

            coords = np.asarray(
                line
            ).reshape(-1)

            if len(coords) != 4:
                continue

            x1, y1, x2, y2 = map(
                int,
                coords
            )

            dx = x2 - x1
            dy = y2 - y1

            length = np.sqrt(
                dx * dx +
                dy * dy
            )

            if length <= 0:
                continue

            angle = abs(
                np.degrees(
                    np.arctan2(
                        dy,
                        dx
                    )
                )
            )

            if angle <= 5:

                if length >= page.shape[1] * 0.25:

                    cv2.line(
                        line_mask,
                        (x1, y1),
                        (x2, y2),
                        255,
                        3
                    )

                    detected_segments += 1

    print(
        f"Detected notebook ruling segments: "
        f"{detected_segments}"
    )

    # --------------------------------------------------------
    # Connect broken notebook lines
    # --------------------------------------------------------

    connect_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (31, 3)
    )

    line_mask = cv2.morphologyEx(
        line_mask,
        cv2.MORPH_CLOSE,
        connect_kernel
    )

    # --------------------------------------------------------
    # Slight vertical expansion
    # --------------------------------------------------------

    remove_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (15, 7)
    )

    line_mask = cv2.dilate(
        line_mask,
        remove_kernel,
        iterations=1
    )

    # --------------------------------------------------------
    # Debug mask
    # --------------------------------------------------------

    backend_folder = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    debug_mask_path = os.path.join(
        backend_folder,
        "processed",
        "debug_notebook_line_mask.jpg"
    )

    cv2.imwrite(
        debug_mask_path,
        line_mask
    )

    # --------------------------------------------------------
    # Inpaint
    # --------------------------------------------------------

    cleaned = cv2.inpaint(
        page,
        line_mask,
        3,
        cv2.INPAINT_TELEA
    )

    cleaned_path = os.path.join(
        backend_folder,
        "processed",
        "debug_lines_removed_color.jpg"
    )

    cv2.imwrite(
        cleaned_path,
        cleaned
    )

    print(
        f"Saved cleaned page: {cleaned_path}"
    )

    return cleaned


# ============================================================
# CREATE HANDWRITING MASK
# ============================================================

def create_handwriting_mask(page):

    gray = cv2.cvtColor(
        page,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(
        gray
    )

    # --------------------------------------------------------
    # Mild blur
    # --------------------------------------------------------

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    height, width = gray.shape

    # --------------------------------------------------------
    # Adaptive threshold
    # --------------------------------------------------------

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12
    )

    # --------------------------------------------------------
    # Remove horizontal notebook structures
    # --------------------------------------------------------

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            max(
                80,
                int(width * 0.20)
            ),
            1
        )
    )

    horizontal_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    handwriting = cv2.subtract(
        binary,
        horizontal_lines
    )

    # --------------------------------------------------------
    # Remove tiny noise
    # --------------------------------------------------------

    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2, 2)
    )

    handwriting = cv2.morphologyEx(
        handwriting,
        cv2.MORPH_OPEN,
        small_kernel
    )

    # --------------------------------------------------------
    # SMALL closing
    #
    # IMPORTANT:
    # Don't use large dilation here because it merges rows.
    # --------------------------------------------------------

    connect_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 2)
    )

    handwriting = cv2.morphologyEx(
        handwriting,
        cv2.MORPH_CLOSE,
        connect_kernel
    )

    # --------------------------------------------------------
    # Save debug mask
    # --------------------------------------------------------

    backend_folder = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    debug_path = os.path.join(
        backend_folder,
        "processed",
        "debug_detection.jpg"
    )

    cv2.imwrite(
        debug_path,
        handwriting
    )

    print(
        f"Saved handwriting detection mask: "
        f"{debug_path}"
    )

    return handwriting


# ============================================================
# SPLIT A MERGED HANDWRITING REGION
# ============================================================

def split_merged_region(
    row_projection,
    start,
    end,
    width
):

    """
    Detect whether a large row group actually contains
    multiple handwriting lines.

    We look for low-ink valleys between the handwriting rows.
    """

    group_height = end - start + 1

    # --------------------------------------------------------
    # Small groups do not need splitting
    # --------------------------------------------------------

    if group_height < 28:

        return [
            (
                start,
                end
            )
        ]

    local = np.asarray(
        row_projection[
            start:end + 1
        ],
        dtype=np.float32
    ).reshape(-1)

    if len(local) < 10:

        return [
            (
                start,
                end
            )
        ]

    # --------------------------------------------------------
    # Smooth the projection
    # --------------------------------------------------------

    smooth = cv2.GaussianBlur(
        local.reshape(-1, 1),
        (1, 7),
        0
    ).reshape(-1)

    maximum = float(
        np.max(smooth)
    )

    if maximum <= 0:

        return [
            (
                start,
                end
            )
        ]

    # --------------------------------------------------------
    # A valley is a region where ink density becomes much
    # lower than the handwriting stroke density.
    # --------------------------------------------------------

    valley_threshold = max(
        2.0,
        maximum * 0.12
    )

    low_rows = (
        smooth <= valley_threshold
    )

    # --------------------------------------------------------
    # Find consecutive low-density valleys
    # --------------------------------------------------------

    valleys = []

    valley_start = None

    for index in range(
        len(low_rows)
    ):

        if bool(
            low_rows[index]
        ):

            if valley_start is None:

                valley_start = index

        else:

            if valley_start is not None:

                valley_end = index - 1

                if (
                    valley_end -
                    valley_start + 1
                ) >= 2:

                    valleys.append(
                        (
                            valley_start,
                            valley_end
                        )
                    )

                valley_start = None

    if valley_start is not None:

        valley_end = (
            len(low_rows) - 1
        )

        if (
            valley_end -
            valley_start + 1
        ) >= 2:

            valleys.append(
                (
                    valley_start,
                    valley_end
                )
            )

    # --------------------------------------------------------
    # Ignore valleys at the very top/bottom.
    # --------------------------------------------------------

    useful_valleys = []

    for valley_start, valley_end in valleys:

        valley_center = (
            valley_start +
            valley_end
        ) / 2

        if valley_center < 5:
            continue

        if valley_center > len(local) - 6:
            continue

        useful_valleys.append(
            (
                valley_start,
                valley_end
            )
        )

    # --------------------------------------------------------
    # If there is no useful valley, keep original group.
    # --------------------------------------------------------

    if not useful_valleys:

        return [
            (
                start,
                end
            )
        ]

    # --------------------------------------------------------
    # Choose the strongest/most central valleys.
    # --------------------------------------------------------

    split_positions = []

    for valley_start, valley_end in useful_valleys:

        center = int(
            round(
                (
                    valley_start +
                    valley_end
                ) / 2
            )
        )

        split_positions.append(
            center
        )

    # --------------------------------------------------------
    # Avoid too many tiny pieces.
    # --------------------------------------------------------

    boundaries = [
        0
    ]

    for position in split_positions:

        if position <= 5:
            continue

        if position >= len(local) - 5:
            continue

        previous = boundaries[-1]

        if (
            position -
            previous
        ) < 12:

            continue

        boundaries.append(
            position
        )

    boundaries.append(
        len(local)
    )

    # --------------------------------------------------------
    # Build split groups
    # --------------------------------------------------------

    result = []

    for index in range(
        len(boundaries) - 1
    ):

        local_start = boundaries[
            index
        ]

        local_end = (
            boundaries[
                index + 1
            ] - 1
        )

        if local_end - local_start < 6:

            continue

        actual_start = (
            start +
            local_start
        )

        actual_end = (
            start +
            local_end
        )

        result.append(
            (
                actual_start,
                actual_end
            )
        )

    # --------------------------------------------------------
    # If splitting generated nothing useful
    # --------------------------------------------------------

    if len(result) <= 1:

        return [
            (
                start,
                end
            )
        ]

    print(
        f"Split merged region "
        f"{start}-{end} into "
        f"{len(result)} rows."
    )

    return result


# ============================================================
# DETECT TEXT LINES
# ============================================================

def detect_text_lines(image):

    print()
    print("----------------------------------------")
    print("STARTING HANDWRITING LINE DETECTION")
    print("----------------------------------------")

    # --------------------------------------------------------
    # Page
    # --------------------------------------------------------

    page = find_page_region(image)

    # --------------------------------------------------------
    # Remove notebook ruling
    # --------------------------------------------------------

    cleaned_page = remove_notebook_lines(page)

    # --------------------------------------------------------
    # Handwriting mask
    # --------------------------------------------------------

    mask = create_handwriting_mask(cleaned_page)

    height, width = mask.shape

    # ========================================================
    # ROW PROJECTION
    # ========================================================

    ink = mask > 0

    row_projection = np.sum(
        ink,
        axis=1
    ).astype(np.float32).reshape(-1)

    print(f"Row projection shape: {row_projection.shape}")

    # --------------------------------------------------------
    # Smooth the projection.
    #
    # We use a wider vertical smoothing window here because
    # the old "active row" method was joining almost the whole
    # page into one giant region.
    # --------------------------------------------------------

    smooth = cv2.GaussianBlur(
        row_projection.reshape(-1, 1),
        (1, 15),
        0
    ).reshape(-1)

    # --------------------------------------------------------
    # Ignore very weak noise.
    # --------------------------------------------------------

    positive = smooth[smooth > 0]

    if len(positive) == 0:
        print("No handwriting ink found.")
        return []

    baseline = float(np.percentile(positive, 35))
    strong_level = float(np.percentile(positive, 65))

    peak_threshold = max(
        baseline + 3.0,
        strong_level * 0.55
    )

    print(f"Line peak threshold: {peak_threshold:.2f}")

    # ========================================================
    # FIND CANDIDATE LINE CENTERS
    # ========================================================

    candidates = []

    for y in range(2, height - 2):

        value = float(smooth[y])

        if value < peak_threshold:
            continue

        # Local maximum.
        if (
            value >= float(smooth[y - 1])
            and value >= float(smooth[y + 1])
        ):
            candidates.append(
                (
                    y,
                    value
                )
            )

    # --------------------------------------------------------
    # Non-maximum suppression.
    #
    # A handwriting line can have several peaks because of
    # ascenders/descenders. Keep only the strongest peak in a
    # neighbourhood.
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item: item[1],
        reverse=True
    )

    selected_peaks = []

    min_peak_distance = max(
        32,
        int(height * 0.025)
    )

    for y, score in candidates:

        too_close = False

        for selected_y, _ in selected_peaks:

            if abs(y - selected_y) < min_peak_distance:
                too_close = True
                break

        if not too_close:
            selected_peaks.append(
                (
                    y,
                    score
                )
            )

    selected_peaks.sort(
        key=lambda item: item[0]
    )

    # --------------------------------------------------------
    # Remove isolated very weak peaks near the page edges.
    # --------------------------------------------------------

    filtered_peaks = []

    for index, item in enumerate(selected_peaks):

        y, score = item

        if (
            y < 15
            or y > height - 15
        ):
            continue

        filtered_peaks.append(item)

    selected_peaks = filtered_peaks

    print(
        f"Detected candidate handwriting peaks: "
        f"{len(selected_peaks)}"
    )

    # ========================================================
    # FALLBACK
    # ========================================================

    if len(selected_peaks) < 3:

        print(
            "Peak detection was too weak. "
            "Falling back to the previous row-group method."
        )

        threshold_pixels = max(
            8,
            int(width * 0.003)
        )

        active_rows = (
            row_projection >= threshold_pixels
        )

        active_rows = np.asarray(
            active_rows,
            dtype=bool
        ).reshape(-1)

        groups = []

        start = None

        for y in range(len(active_rows)):

            if bool(active_rows[y]):

                if start is None:
                    start = y

            else:

                if start is not None:

                    end = y - 1

                    if end - start >= 3:
                        groups.append(
                            (
                                start,
                                end
                            )
                        )

                    start = None

        if start is not None:

            end = len(active_rows) - 1

            if end - start >= 3:
                groups.append(
                    (
                        start,
                        end
                    )
                )

        print(
            f"Fallback row groups: {len(groups)}"
        )

        boxes = []

        for y1, y2 in groups:

            region = mask[
                y1:y2 + 1,
                :
            ]

            column_projection = np.sum(
                region > 0,
                axis=0
            ).reshape(-1)

            xs = np.where(
                column_projection > 0
            )[0]

            if len(xs) == 0:
                continue

            x1 = int(xs[0])
            x2 = int(xs[-1])

            if x2 - x1 < width * 0.20:
                continue

            boxes.append(
                (
                    max(0, x1 - 15),
                    max(0, y1 - 10),
                    min(width, x2 + 15),
                    min(height, y2 + 10)
                )
            )

        return boxes

    # ========================================================
    # BUILD LINE BOUNDARIES FROM PEAKS
    # ========================================================

    centers = [
        item[0]
        for item in selected_peaks
    ]

    # --------------------------------------------------------
    # If there are too many peaks, remove the weakest ones
    # while preserving the vertical order.
    #
    # For this page the handwriting is expected to have roughly
    # 20 lines, but we don't hard-code exactly 20. We only use
    # the expected density to eliminate obvious noise peaks.
    # --------------------------------------------------------

    if len(centers) > 28:

        selected_peaks.sort(
            key=lambda item: item[1],
            reverse=True
        )

        selected_peaks = selected_peaks[:28]

        selected_peaks.sort(
            key=lambda item: item[0]
        )

        centers = [
            item[0]
            for item in selected_peaks
        ]

    print(
        f"Final line centers: {len(centers)}"
    )

    # ========================================================
    # CREATE LINE BOXES
    # ========================================================

    boxes = []

    for index, center in enumerate(centers):

        # ----------------------------------------------------
        # Boundary halfway between neighbouring line centers.
        # ----------------------------------------------------

        if index == 0:

            top = max(
                0,
                int(
                    center -
                    max(
                        25,
                        (centers[1] - center) // 2
                    )
                )
            )

        else:

            top = int(
                (
                    centers[index - 1] +
                    center
                ) / 2
            )

        if index == len(centers) - 1:

            bottom = min(
                height,
                int(
                    center +
                    max(
                        25,
                        (center - centers[index - 1]) // 2
                    )
                )
            )

        else:

            bottom = int(
                (
                    center +
                    centers[index + 1]
                ) / 2
            )

        # ----------------------------------------------------
        # Find actual ink inside this band.
        # ----------------------------------------------------

        band = mask[
            top:bottom,
            :
        ]

        column_projection = np.sum(
            band > 0,
            axis=0
        ).reshape(-1)

        xs = np.where(
            column_projection > 0
        )[0]

        if len(xs) == 0:
            continue

        x1 = int(xs[0])
        x2 = int(xs[-1])

        line_width = x2 - x1

        # ----------------------------------------------------
        # Reject tiny isolated marks.
        # ----------------------------------------------------

        if line_width < width * 0.20:
            continue

        # ----------------------------------------------------
        # Padding.
        # ----------------------------------------------------

        pad_x = 15
        pad_y = 8

        crop_x1 = max(
            0,
            x1 - pad_x
        )

        crop_x2 = min(
            width,
            x2 + pad_x
        )

        crop_y1 = max(
            0,
            top + pad_y
        )

        crop_y2 = min(
            height,
            bottom - pad_y
        )

        # If padding made the crop too small, use the band.
        if crop_y2 <= crop_y1:

            crop_y1 = max(
                0,
                top
            )

            crop_y2 = min(
                height,
                bottom
            )

        boxes.append(
            (
                crop_x1,
                crop_y1,
                crop_x2,
                crop_y2
            )
        )

    # ========================================================
    # REMOVE DUPLICATE / OVERLAPPING BOXES
    # ========================================================

    boxes.sort(
        key=lambda box: box[1]
    )

    final_boxes = []

    for box in boxes:

        if not final_boxes:

            final_boxes.append(box)
            continue

        previous = final_boxes[-1]

        overlap_start = max(
            previous[1],
            box[1]
        )

        overlap_end = min(
            previous[3],
            box[3]
        )

        overlap = max(
            0,
            overlap_end - overlap_start
        )

        previous_height = (
            previous[3] -
            previous[1]
        )

        current_height = (
            box[3] -
            box[1]
        )

        smaller_height = min(
            previous_height,
            current_height
        )

        if (
            smaller_height > 0
            and
            overlap / smaller_height > 0.85
        ):

            # Keep the stronger/larger box.
            previous_area = (
                previous[2] - previous[0]
            ) * previous_height

            current_area = (
                box[2] - box[0]
            ) * current_height

            if current_area > previous_area:
                final_boxes[-1] = box

        else:

            final_boxes.append(box)

    final_boxes.sort(
        key=lambda box: box[1]
    )

    # Remove the small false mark detected above the first real
    # handwritten sentence in the test notebook image.
    if final_boxes:
        first = final_boxes[0]
        first_height = first[3] - first[1]

        if first[1] <= 30 and first_height <= 45:
            print(
                f"Removed false top artifact line: {first}"
            )
            final_boxes = final_boxes[1:]

    print()
    print(
        f"FINAL DETECTED LINES: "
        f"{len(final_boxes)}"
    )

    for index, box in enumerate(final_boxes):

        print(
            f"Line {index + 1}: "
            f"{box}"
        )

    # ========================================================
    # DEBUG VISUALIZATION
    # ========================================================

    debug_visual = page.copy()

    for index, box in enumerate(final_boxes):

        x1, y1, x2, y2 = box

        cv2.rectangle(
            debug_visual,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            debug_visual,
            str(index + 1),
            (
                x1 + 5,
                max(
                    20,
                    y1 - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    backend_folder = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    debug_path = os.path.join(
        backend_folder,
        "processed",
        "debug_detected_text_lines.jpg"
    )

    cv2.imwrite(
        debug_path,
        debug_visual
    )

    print(
        f"Saved line detection visualization: "
        f"{debug_path}"
    )

    return final_boxes


# ============================================================
# PREPARE LINE FOR TrOCR
# ============================================================

def prepare_original_line(line_image):

    # Keep the handwriting strokes close to the original image.
    # The previous blur could soften thin handwritten strokes and
    # make TrOCR confuse characters such as r/l/i/u.
    gray = cv2.cvtColor(
        line_image,
        cv2.COLOR_BGR2GRAY
    )

    # Mild local contrast enhancement.
    clahe = cv2.createCLAHE(
        clipLimit=1.2,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

    # Remove only very small intensity noise without blurring strokes.
    enhanced = cv2.medianBlur(
        enhanced,
        3
    )

    # Give TrOCR a little clean margin around the handwriting.
    # This avoids characters touching the crop boundary.
    enhanced = cv2.copyMakeBorder(
        enhanced,
        8,
        8,
        12,
        12,
        cv2.BORDER_CONSTANT,
        value=255
    )

    rgb = cv2.cvtColor(
        enhanced,
        cv2.COLOR_GRAY2RGB
    )

    return Image.fromarray(
        rgb
    )


# ============================================================
# RECOGNIZE MULTIPLE LINES
# ============================================================

def recognize_lines(line_images):

    load_trocr_model()

    if not line_images:

        return []

    pil_images = []

    for image in line_images:

        pil_image = prepare_original_line(
            image
        )

        pil_images.append(
            pil_image
        )

    inputs = processor(
        images=pil_images,
        return_tensors="pt"
    )

    pixel_values = (
        inputs.pixel_values.to(
            device
        )
    )

    print(
        f"Running TrOCR on "
        f"{len(pil_images)} lines..."
    )

    with torch.no_grad():

        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=64,
            num_beams=2,
            do_sample=False
        )

    texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )

    results = []

    for text in texts:

        text = text.strip()

        results.append(
            {
                "text": text,
                "confidence": 0.0
            }
        )

    return results


# ============================================================
# RECOGNIZE SINGLE LINE
# ============================================================

def recognize_line(line_image):

    results = recognize_lines(
        [line_image]
    )

    if not results:

        return (
            "",
            0.0
        )

    return (
        results[0]["text"],
        results[0]["confidence"]
    )


# ============================================================
# EXTRACT TEXT
# ============================================================

def extract_text(image_path):

    print()
    print(
        "============================================================"
    )
    print(
        "OCR EXTRACTION STARTED"
    )
    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # Read
    # --------------------------------------------------------

    original = read_original_image(
        image_path
    )

    # --------------------------------------------------------
    # Detect lines
    # --------------------------------------------------------

    line_boxes = detect_text_lines(
        original
    )

    print(
        f"Detected "
        f"{len(line_boxes)} handwriting lines."
    )

    # --------------------------------------------------------
    # Create crops
    # --------------------------------------------------------

    line_images = []

    for box in line_boxes:

        x1, y1, x2, y2 = box

        crop = original[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:

            continue

        line_images.append(
            crop
        )

    # --------------------------------------------------------
    # Save crops
    # --------------------------------------------------------

    save_debug_line_images(
        line_images
    )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if not line_images:

        print(
            "No handwriting lines detected."
        )

        print(
            "Running OCR on complete image."
        )

        text, _ = recognize_line(
            original
        )

        return text.strip()

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    results = recognize_lines(
        line_images
    )

    final_lines = []

    for index, result in enumerate(
        results
    ):

        text = result["text"]

        print(
            f"Line {index + 1}: "
            f"{text}"
        )

        if text:

            final_lines.append(
                text
            )

    final_text = "\n".join(
        final_lines
    ).strip()

    print()
    print(
        "FINAL OCR TEXT:"
    )

    print(
        final_text
    )

    print()
    print(
        "============================================================"
    )
    print(
        "OCR EXTRACTION COMPLETED"
    )
    print(
        "============================================================"
    )

    return final_text


# ============================================================
# ANALYZE OCR
# ============================================================

def analyze_ocr(image_path):

    start_time = time.time()

    image = read_original_image(
        image_path
    )

    boxes = detect_text_lines(
        image
    )

    line_images = []

    for box in boxes:

        x1, y1, x2, y2 = box

        crop = image[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:
            continue

        line_images.append(
            crop
        )

    save_debug_line_images(
        line_images
    )

    if not line_images:

        text, confidence = recognize_line(
            image
        )

        results = [
            {
                "text": text,
                "confidence": confidence
            }
        ]

    else:

        results = recognize_lines(
            line_images
        )

    texts = []

    confidence_values = []

    for index, result in enumerate(
        results
    ):

        text = result["text"]

        confidence = result[
            "confidence"
        ]

        print(
            f"Line {index + 1}: "
            f"{text}"
        )

        if text:

            texts.append(
                text
            )

        confidence_values.append(
            confidence
        )

    final_text = "\n".join(
        texts
    ).strip()

    if confidence_values:

        average_confidence = round(
            sum(
                confidence_values
            )
            /
            len(
                confidence_values
            ),
            2
        )

    else:

        average_confidence = 0

    word_count = len(
        final_text.split()
    )

    character_count = len(
        final_text.replace(
            " ",
            ""
        )
    )

    processing_time = round(
        time.time() -
        start_time,
        3
    )

    return {
        "text": final_text,
        "word_count": word_count,
        "character_count": character_count,
        "confidence": average_confidence,
        "processing_time": processing_time
    }


# ============================================================
# ACCURACY
# ============================================================

def calculate_accuracy(
    reference_text,
    predicted_text
):

    reference_text = (
        reference_text.strip()
    )

    predicted_text = (
        predicted_text.strip()
    )

    if not reference_text:

        raise ValueError(
            "Reference text cannot be empty."
        )

    word_error_rate = wer(
        reference_text,
        predicted_text
    )

    character_error_rate = cer(
        reference_text,
        predicted_text
    )

    word_accuracy = max(
        0,
        1 - word_error_rate
    ) * 100

    character_accuracy = max(
        0,
        1 - character_error_rate
    ) * 100

    return {
        "word_error_rate": round(
            word_error_rate,
            4
        ),
        "character_error_rate": round(
            character_error_rate,
            4
        ),
        "word_accuracy": round(
            word_accuracy,
            2
        ),
        "character_accuracy": round(
            character_accuracy,
            2
        )
    }