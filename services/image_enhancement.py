import cv2
import os


def enhance_image(input_path, output_path):
    """
    Improves the quality of a handwritten image.

    Steps:
    1. Read image
    2. Upscale image
    3. Denoise
    4. Improve contrast
    5. Sharpen
    6. Save enhanced image
    """

    # Read the original image
    image = cv2.imread(input_path)

    if image is None:
        raise ValueError("Could not read the input image.")

    # -----------------------------------
    # 1. Upscale image
    # -----------------------------------

    height, width = image.shape[:2]

    scale_factor = 2

    enhanced = cv2.resize(
        image,
        (width * scale_factor, height * scale_factor),
        interpolation=cv2.INTER_CUBIC
    )

    # -----------------------------------
    # 2. Denoise
    # -----------------------------------

    enhanced = cv2.fastNlMeansDenoisingColored(
        enhanced,
        None,
        10,
        10,
        7,
        21
    )

    # -----------------------------------
    # 3. Improve contrast
    # -----------------------------------

    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(l_channel)

    lab = cv2.merge(
        (l_channel, a_channel, b_channel)
    )

    enhanced = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    # -----------------------------------
    # 4. Sharpen
    # -----------------------------------

    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        3
    )

    enhanced = cv2.addWeighted(
        enhanced,
        1.5,
        blurred,
        -0.5,
        0
    )

    # -----------------------------------
    # 5. Save enhanced image
    # -----------------------------------

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    cv2.imwrite(
        output_path,
        enhanced
    )

    return output_path