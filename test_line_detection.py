import os
import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BACKEND_FOLDER = os.path.dirname(os.path.abspath(__file__))

INPUT_IMAGE = os.path.join(
    BACKEND_FOLDER,
    "processed",
    "preprocessing_test_v3",
    "15_final_clean_handwriting.png"
)

OUTPUT_FOLDER = os.path.join(
    BACKEND_FOLDER,
    "processed",
    "line_detection_test_v4"
)

LINES_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "lines"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(LINES_FOLDER, exist_ok=True)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image():

    image = cv2.imread(INPUT_IMAGE)

    if image is None:

        print()
        print("ERROR: Could not load image.")
        print()
        print("Expected:")
        print(INPUT_IMAGE)
        print()

        return None

    return image


# ============================================================
# CREATE INK IMAGE
# ============================================================

def create_ink_image(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Convert dark handwriting into white foreground
    # and white background into black.
    _, ink = cv2.threshold(
        gray,
        180,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Remove tiny isolated noise.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2, 2)
    )

    ink = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        kernel
    )

    return gray, ink


# ============================================================
# CALCULATE ROW PROJECTION
# ============================================================

def calculate_row_projection(ink):

    # Make absolutely sure ink is 2D.
    if len(ink.shape) != 2:

        raise ValueError(
            f"Expected 2D ink image, got shape: {ink.shape}"
        )

    projection = np.sum(
        ink > 0,
        axis=1
    )

    # Force projection to one-dimensional.
    projection = np.asarray(
        projection
    ).reshape(-1)

    return projection


# ============================================================
# SAVE ROW PROJECTION
# ============================================================

def save_projection_image(
    projection,
    width
):

    projection = np.asarray(
        projection
    ).reshape(-1)

    height = len(projection)

    canvas = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    maximum = float(
        np.max(projection)
    )

    if maximum <= 0:

        path = os.path.join(
            OUTPUT_FOLDER,
            "01_row_projection_v4.png"
        )

        cv2.imwrite(
            path,
            canvas
        )

        return canvas

    for y in range(height):

        value = float(
            projection[y]
        )

        line_width = int(
            (value / maximum) *
            (width - 1)
        )

        if line_width > 0:

            canvas[
                y,
                0:line_width
            ] = 255

    path = os.path.join(
        OUTPUT_FOLDER,
        "01_row_projection_v4.png"
    )

    cv2.imwrite(
        path,
        canvas
    )

    return canvas


# ============================================================
# FIND HANDWRITING ROWS
# ============================================================

def detect_handwriting_rows(
    projection,
    image_height,
    image_width
):

    # Force everything to simple 1D arrays.
    projection = np.asarray(
        projection
    ).reshape(-1)

    # Make sure image height matches projection.
    image_height = min(
        image_height,
        len(projection)
    )

    # Percentage of image width occupied by ink.
    ink_ratio = (
        projection[:image_height]
        / float(image_width)
    )

    # A small amount of ink is enough to consider
    # that Y-coordinate part of handwriting.
    active_threshold = 0.015

    active = (
        ink_ratio > active_threshold
    )

    # IMPORTANT:
    # Force active to one-dimensional boolean array.
    active = np.asarray(
        active
    ).reshape(-1)

    # --------------------------------------------------------
    # Fill very small gaps inside a handwriting stroke.
    # --------------------------------------------------------

    active_uint8 = (
        active.astype(np.uint8) * 255
    )

    gap_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, 5)
    )

    active_uint8 = cv2.morphologyEx(
        active_uint8,
        cv2.MORPH_CLOSE,
        gap_kernel
    )

    active = (
        active_uint8 > 0
    )

    active = np.asarray(
        active
    ).reshape(-1)

    # --------------------------------------------------------
    # Find continuous active Y regions.
    # --------------------------------------------------------

    raw_regions = []

    start = None

    for y in range(
        len(active)
    ):

        # Convert explicitly to Python bool.
        is_active = bool(
            active[y]
        )

        if is_active:

            if start is None:
                start = y

        else:

            if start is not None:

                end = y - 1

                raw_regions.append(
                    [start, end]
                )

                start = None

    # Handle region reaching bottom.
    if start is not None:

        raw_regions.append(
            [
                start,
                len(active) - 1
            ]
        )

    # --------------------------------------------------------
    # Remove tiny regions.
    # --------------------------------------------------------

    filtered_regions = []

    for start, end in raw_regions:

        region_height = (
            end - start + 1
        )

        if region_height >= 3:

            filtered_regions.append(
                [start, end]
            )

    # --------------------------------------------------------
    # Merge ONLY extremely small gaps.
    # --------------------------------------------------------

    merged_regions = []

    for start, end in filtered_regions:

        if not merged_regions:

            merged_regions.append(
                [start, end]
            )

            continue

        previous_start, previous_end = (
            merged_regions[-1]
        )

        gap = (
            start -
            previous_end -
            1
        )

        # Only merge tiny gaps.
        if gap <= 3:

            merged_regions[-1][1] = end

        else:

            merged_regions.append(
                [start, end]
            )

    # --------------------------------------------------------
    # If no regions were found.
    # --------------------------------------------------------

    if not merged_regions:

        return []

    # --------------------------------------------------------
    # Calculate median line height.
    # --------------------------------------------------------

    heights = []

    for start, end in merged_regions:

        heights.append(
            end - start + 1
        )

    median_height = float(
        np.median(heights)
    )

    # --------------------------------------------------------
    # Split suspiciously large regions.
    # --------------------------------------------------------

    final_regions = []

    for start, end in merged_regions:

        region_height = (
            end - start + 1
        )

        # A normal handwriting line shouldn't
        # occupy a huge part of the page.
        if region_height > (
            median_height * 2.5
        ):

            sub_projection = projection[
                start:end + 1
            ]

            sub_threshold = (
                image_width * 0.008
            )

            sub_active = (
                sub_projection >
                sub_threshold
            )

            sub_active = np.asarray(
                sub_active
            ).reshape(-1)

            sub_start = None

            for index in range(
                len(sub_active)
            ):

                is_active = bool(
                    sub_active[index]
                )

                if is_active:

                    if sub_start is None:
                        sub_start = index

                else:

                    if sub_start is not None:

                        sub_end = index - 1

                        if (
                            sub_end -
                            sub_start +
                            1 >= 3
                        ):

                            final_regions.append(
                                [
                                    start + sub_start,
                                    start + sub_end
                                ]
                            )

                        sub_start = None

            if sub_start is not None:

                sub_end = (
                    len(sub_active) - 1
                )

                if (
                    sub_end -
                    sub_start +
                    1 >= 3
                ):

                    final_regions.append(
                        [
                            start + sub_start,
                            start + sub_end
                        ]
                    )

        else:

            final_regions.append(
                [start, end]
            )

    # --------------------------------------------------------
    # Sort top to bottom.
    # --------------------------------------------------------

    final_regions.sort(
        key=lambda region: region[0]
    )

    return final_regions


# ============================================================
# CROP INDIVIDUAL LINES
# ============================================================

def crop_lines(
    original_image,
    ink,
    regions
):

    height, width = ink.shape

    # Remove old crops.
    if os.path.exists(
        LINES_FOLDER
    ):

        for filename in os.listdir(
            LINES_FOLDER
        ):

            file_path = os.path.join(
                LINES_FOLDER,
                filename
            )

            if os.path.isfile(
                file_path
            ):

                try:
                    os.remove(
                        file_path
                    )
                except:
                    pass

    saved_lines = []

    for index, (start, end) in enumerate(
        regions,
        start=1
    ):

        # Vertical padding.
        y1 = max(
            0,
            start - 7
        )

        y2 = min(
            height,
            end + 8
        )

        crop = original_image[
            y1:y2,
            :
        ]

        crop_ink = ink[
            y1:y2,
            :
        ]

        # ----------------------------------------------------
        # Find actual horizontal handwriting area.
        # ----------------------------------------------------

        column_projection = np.sum(
            crop_ink > 0,
            axis=0
        )

        column_projection = np.asarray(
            column_projection
        ).reshape(-1)

        active_columns = np.where(
            column_projection > 0
        )[0]

        if len(active_columns) > 0:

            x1 = max(
                0,
                int(
                    active_columns[0]
                ) - 8
            )

            x2 = min(
                width,
                int(
                    active_columns[-1]
                ) + 9
            )

            crop = crop[
                :,
                x1:x2
            ]

        filename = (
            f"line_{index:03d}.png"
        )

        output_path = os.path.join(
            LINES_FOLDER,
            filename
        )

        cv2.imwrite(
            output_path,
            crop
        )

        saved_lines.append(
            output_path
        )

    return saved_lines


# ============================================================
# VISUALIZATION
# ============================================================

def create_visualization(
    image,
    regions
):

    visualization = image.copy()

    height, width = (
        visualization.shape[:2]
    )

    for index, (start, end) in enumerate(
        regions,
        start=1
    ):

        y1 = max(
            0,
            start - 7
        )

        y2 = min(
            height - 1,
            end + 7
        )

        cv2.rectangle(
            visualization,
            (0, y1),
            (width - 1, y2),
            (0, 0, 255),
            1
        )

        cv2.putText(
            visualization,
            str(index),
            (
                5,
                max(
                    15,
                    y1 - 2
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        "02_line_detection_visualization_v4.png"
    )

    cv2.imwrite(
        output_path,
        visualization
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("HANDWRITING LINE DETECTION - VERSION 4")
    print("=" * 60)
    print()

    print("Input image:")
    print(INPUT_IMAGE)
    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    image = load_image()

    if image is None:
        return

    print("Clean handwriting image loaded.")

    # --------------------------------------------------------
    # Create ink image
    # --------------------------------------------------------

    gray, ink = create_ink_image(
        image
    )

    binary_display = cv2.bitwise_not(
        ink
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            "00_binary_handwriting_v4.png"
        ),
        binary_display
    )

    print("Ink image created.")

    # --------------------------------------------------------
    # Row projection
    # --------------------------------------------------------

    projection = calculate_row_projection(
        ink
    )

    save_projection_image(
        projection,
        ink.shape[1]
    )

    print("Row projection calculated.")

    # --------------------------------------------------------
    # Detect lines
    # --------------------------------------------------------

    regions = detect_handwriting_rows(
        projection,
        ink.shape[0],
        ink.shape[1]
    )

    print()
    print(
        f"Detected handwriting lines: "
        f"{len(regions)}"
    )

    # --------------------------------------------------------
    # Crop lines
    # --------------------------------------------------------

    saved_lines = crop_lines(
        image,
        ink,
        regions
    )

    print(
        f"Saved line crops: "
        f"{len(saved_lines)}"
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    visualization_path = (
        create_visualization(
            image,
            regions
        )
    )

    print()
    print("Visualization:")
    print(
        visualization_path
    )

    print()
    print("Line crops:")
    print(
        LINES_FOLDER
    )

    print()
    print("=" * 60)
    print("LINE DETECTION VERSION 4 COMPLETED")
    print("=" * 60)
    print()

    print("IMPORTANT:")
    print(
        "ocr_service.py has NOT been modified."
    )
    print(
        "Do not modify it yet."
    )
    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()