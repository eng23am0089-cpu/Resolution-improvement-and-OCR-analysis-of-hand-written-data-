import cv2
import numpy as np
import os
import math


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_IMAGE = r"C:\Users\yoges\OneDrive\Desktop\handwriting-ocr\backend\test_handwriting.jpg"

BACKEND_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)

OUTPUT_FOLDER = os.path.join(
    BACKEND_FOLDER,
    "processed",
    "preprocessing_test_v3"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(filename, image):

    path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )

    success = cv2.imwrite(
        path,
        image
    )

    if success:
        print(f"Saved: {filename}")
    else:
        print(f"ERROR: Could not save {filename}")


# ============================================================
# DETECT LONG NOTEBOOK LINES
# ============================================================

def detect_notebook_lines(gray):

    height, width = gray.shape

    print()
    print("Detecting long straight notebook lines...")

    # --------------------------------------------------------
    # Improve contrast slightly
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # --------------------------------------------------------
    # Detect edges
    # --------------------------------------------------------

    edges = cv2.Canny(
        enhanced,
        40,
        120
    )

    save_image(
        "05_edges.png",
        edges
    )

    # --------------------------------------------------------
    # Hough line detection
    # --------------------------------------------------------

    min_line_length = int(
        width * 0.35
    )

    max_line_gap = 25

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 1800,
        threshold=50,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    line_mask = np.zeros_like(
        gray
    )

    detected_count = 0

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line

            dx = x2 - x1
            dy = y2 - y1

            length = math.sqrt(
                dx * dx + dy * dy
            )

            if length == 0:
                continue

            angle = math.degrees(
                math.atan2(
                    dy,
                    dx
                )
            )

            # Normalize angle
            if angle < -90:
                angle += 180

            if angle > 90:
                angle -= 180

            # ------------------------------------------------
            # Only accept lines that are almost horizontal.
            #
            # This prevents handwriting strokes from being
            # treated as notebook ruling.
            # ------------------------------------------------

            if abs(angle) > 2.5:
                continue

            # ------------------------------------------------
            # Require a long line.
            # ------------------------------------------------

            if length < min_line_length:
                continue

            # ------------------------------------------------
            # Draw the detected line into the mask.
            # Thickness is intentionally small.
            # ------------------------------------------------

            cv2.line(
                line_mask,
                (x1, y1),
                (x2, y2),
                255,
                2
            )

            detected_count += 1

    print(
        f"Candidate notebook lines detected: "
        f"{detected_count}"
    )

    return line_mask


# ============================================================
# CONNECT BROKEN NOTEBOOK LINES
# ============================================================

def connect_notebook_lines(line_mask):

    print(
        "Connecting small gaps in notebook lines..."
    )

    height, width = line_mask.shape

    kernel_width = max(
        15,
        width // 30
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_width, 1)
    )

    connected = cv2.morphologyEx(
        line_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return connected


# ============================================================
# REMOVE SMALL FALSE DETECTIONS
# ============================================================

def remove_small_components(mask):

    number_of_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    clean_mask = np.zeros_like(
        mask
    )

    height, width = mask.shape

    minimum_length = int(
        width * 0.25
    )

    for label in range(
        1,
        number_of_labels
    ):

        component_width = stats[
            label,
            cv2.CC_STAT_WIDTH
        ]

        component_height = stats[
            label,
            cv2.CC_STAT_HEIGHT
        ]

        if component_width >= minimum_length:

            if component_height <= 8:

                clean_mask[
                    labels == label
                ] = 255

    return clean_mask


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 65)
    print("HANDWRITING PREPROCESSING TEST - VERSION 3")
    print("=" * 65)
    print()

    # --------------------------------------------------------
    # 1. LOAD IMAGE
    # --------------------------------------------------------

    print("Loading image...")

    image = cv2.imread(
        INPUT_IMAGE
    )

    if image is None:

        print()
        print("ERROR: Could not load image.")
        print()
        print("Expected:")
        print(INPUT_IMAGE)
        print()

        return

    height, width = image.shape[:2]

    print(
        f"Image loaded successfully: "
        f"{width} x {height}"
    )

    print()

    save_image(
        "01_original.png",
        image
    )


    # --------------------------------------------------------
    # 2. GRAYSCALE
    # --------------------------------------------------------

    print("Step 2: Converting to grayscale...")

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    save_image(
        "02_grayscale.png",
        gray
    )


    # --------------------------------------------------------
    # 3. CONTRAST ENHANCEMENT
    # --------------------------------------------------------

    print(
        "Step 3: Improving local contrast..."
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

    save_image(
        "03_contrast_enhanced.png",
        enhanced
    )


    # --------------------------------------------------------
    # 4. DETECT NOTEBOOK LINES
    # --------------------------------------------------------

    print(
        "Step 4: Detecting notebook ruling..."
    )

    line_mask = detect_notebook_lines(
        gray
    )

    save_image(
        "06_raw_notebook_lines.png",
        line_mask
    )


    # --------------------------------------------------------
    # 5. CONNECT BROKEN LINES
    # --------------------------------------------------------

    print(
        "Step 5: Connecting broken notebook lines..."
    )

    connected_lines = connect_notebook_lines(
        line_mask
    )

    save_image(
        "07_connected_notebook_lines.png",
        connected_lines
    )


    # --------------------------------------------------------
    # 6. REMOVE FALSE DETECTIONS
    # --------------------------------------------------------

    print(
        "Step 6: Removing short false detections..."
    )

    clean_line_mask = remove_small_components(
        connected_lines
    )

    save_image(
        "08_final_notebook_line_mask.png",
        clean_line_mask
    )


    # --------------------------------------------------------
    # 7. VISUALIZE DETECTED LINES
    # --------------------------------------------------------

    print(
        "Step 7: Creating line visualization..."
    )

    visualization = image.copy()

    visualization[
        clean_line_mask > 0
    ] = (0, 0, 255)

    save_image(
        "09_detected_lines_visualization.png",
        visualization
    )


    # --------------------------------------------------------
    # 8. INPAINT NOTEBOOK LINES
    # --------------------------------------------------------

    print(
        "Step 8: Removing notebook lines using inpainting..."
    )

    # Slightly expand the mask.
    #
    # This covers the complete thickness of the faint
    # notebook ruling without aggressively affecting
    # handwriting.

    inpaint_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    inpaint_mask = cv2.dilate(
        clean_line_mask,
        inpaint_kernel,
        iterations=1
    )

    save_image(
        "10_inpaint_mask.png",
        inpaint_mask
    )


    # --------------------------------------------------------
    # 9. PERFORM INPAINTING
    # --------------------------------------------------------

    cleaned_image = cv2.inpaint(
        image,
        inpaint_mask,
        3,
        cv2.INPAINT_TELEA
    )

    save_image(
        "11_lines_removed_color.png",
        cleaned_image
    )


    # --------------------------------------------------------
    # 10. GRAYSCALE CLEANED IMAGE
    # --------------------------------------------------------

    cleaned_gray = cv2.cvtColor(
        cleaned_image,
        cv2.COLOR_BGR2GRAY
    )

    save_image(
        "12_lines_removed_grayscale.png",
        cleaned_gray
    )


    # --------------------------------------------------------
    # 11. CREATE BINARY HANDWRITING IMAGE
    # --------------------------------------------------------

    print(
        "Step 11: Creating handwriting mask..."
    )

    cleaned_blurred = cv2.GaussianBlur(
        cleaned_gray,
        (3, 3),
        0
    )

    handwriting_binary = cv2.adaptiveThreshold(
        cleaned_blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12
    )

    save_image(
        "13_handwriting_binary.png",
        handwriting_binary
    )


    # --------------------------------------------------------
    # 12. SMALL NOISE REMOVAL
    # --------------------------------------------------------

    print(
        "Step 12: Cleaning small noise..."
    )

    noise_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2)
    )

    handwriting_clean = cv2.morphologyEx(
        handwriting_binary,
        cv2.MORPH_OPEN,
        noise_kernel
    )

    save_image(
        "14_clean_handwriting_mask.png",
        handwriting_clean
    )


    # --------------------------------------------------------
    # 13. FINAL WHITE BACKGROUND
    # --------------------------------------------------------

    final_handwriting = np.full_like(
        cleaned_gray,
        255
    )

    final_handwriting[
        handwriting_clean > 0
    ] = 0

    save_image(
        "15_final_clean_handwriting.png",
        final_handwriting
    )


    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("VERSION 3 PREPROCESSING COMPLETED")
    print("=" * 65)
    print()

    print("Output folder:")
    print(OUTPUT_FOLDER)
    print()

    print("Important files to inspect:")
    print()
    print("08_final_notebook_line_mask.png")
    print("09_detected_lines_visualization.png")
    print("11_lines_removed_color.png")
    print("12_lines_removed_grayscale.png")
    print("15_final_clean_handwriting.png")
    print()

    print("IMPORTANT:")
    print("ocr_service.py has NOT been modified.")
    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()