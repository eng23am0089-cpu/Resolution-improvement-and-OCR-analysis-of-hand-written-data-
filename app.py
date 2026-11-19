import os
import traceback

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import werkzeug.utils

from services.image_enhancement import enhance_image

from services.ocr_service import (
    extract_text,
    analyze_ocr,
    calculate_accuracy
)


# ==========================================
# Flask Application
# ==========================================

app = Flask(__name__)

CORS(app)


# ==========================================
# Configuration
# ==========================================

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER


# ==========================================
# Create Required Folders
# ==========================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    PROCESSED_FOLDER,
    exist_ok=True
)


# ==========================================
# Helper Function
# ==========================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ==========================================
# Home Route
# ==========================================

@app.route("/")
def home():

    return jsonify({
        "message":
            "Handwriting Resolution Improvement & OCR API is running!"
    })


# ==========================================
# Upload Image
# ==========================================

@app.route("/upload", methods=["POST"])
def upload_image():

    # Check whether image was provided

    if "image" not in request.files:

        return jsonify({
            "error": "No image file provided"
        }), 400

    file = request.files["image"]

    # Check filename

    if file.filename == "":

        return jsonify({
            "error": "No file selected"
        }), 400

    # Check file extension

    if not allowed_file(file.filename):

        return jsonify({
            "error":
                "Only PNG, JPG and JPEG files are allowed"
        }), 400

    # Secure filename

    filename = werkzeug.utils.secure_filename(
        file.filename
    )

    # Create file path

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    # Save image

    file.save(filepath)

    return jsonify({

        "message":
            "Image uploaded successfully",

        "filename":
            filename,

        "path":
            filepath

    }), 200


# ==========================================
# Resolution Enhancement
# ==========================================

@app.route("/enhance/<filename>", methods=["POST"])
def enhance_uploaded_image(filename):

    # Original image path

    input_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    # Check if original image exists

    if not os.path.exists(input_path):

        return jsonify({
            "error": "Image not found"
        }), 404

    # Enhanced filename

    output_filename = f"enhanced_{filename}"

    # Enhanced image path

    output_path = os.path.join(
        app.config["PROCESSED_FOLDER"],
        output_filename
    )

    try:

        # Enhance image

        enhance_image(
            input_path,
            output_path
        )

        return jsonify({

            "message":
                "Image enhanced successfully",

            "original":
                filename,

            "enhanced":
                output_filename,

            "path":
                output_path

        }), 200

    except Exception as e:

        print()
        print("=" * 80)
        print("!!!!!!!!!!!! IMAGE ENHANCEMENT ERROR !!!!!!!!!!!!")
        print("=" * 80)

        print(
            f"Error type: {type(e).__name__}"
        )

        print(
            f"Error message: {str(e)}"
        )

        print()

        traceback.print_exc()

        print()
        print("=" * 80)
        print("!!!!!!!! END IMAGE ENHANCEMENT ERROR !!!!!!!!!!")
        print("=" * 80)
        print()

        return jsonify({
            "error": str(e)
        }), 500


# ==========================================
# Serve Enhanced / Processed Images
# ==========================================

@app.route("/processed/<filename>")
def get_processed_image(filename):

    return send_from_directory(
        app.config["PROCESSED_FOLDER"],
        filename
    )


# ==========================================
# OCR
# ==========================================

@app.route("/ocr/<filename>", methods=["POST"])
def perform_ocr(filename):

    print()
    print("=" * 70)
    print("OCR REQUEST RECEIVED")
    print("=" * 70)

    # Look for enhanced image

    image_path = os.path.join(
        app.config["PROCESSED_FOLDER"],
        filename
    )

    print(
        f"OCR image: {image_path}"
    )

    print()

    # Check if processed image exists

    if not os.path.exists(image_path):

        print(
            "ERROR: Processed image not found."
        )

        return jsonify({
            "error":
                "Processed image not found"
        }), 404

    try:

        print()
        print("=" * 60)
        print("OCR EXTRACTION STARTED")
        print("=" * 60)
        print()

        # ----------------------------------
        # Extract text
        # ----------------------------------

        text = extract_text(
            image_path
        )

        # ----------------------------------
        # OCR RESPONSE DEBUG
        # ----------------------------------

        print()
        print(
            "========== OCR RESPONSE DEBUG =========="
        )

        print(
            "TEXT TYPE:",
            type(text)
        )

        print(
            "TEXT REPR:",
            repr(text)
        )

        try:

            print(
                "TEXT LENGTH:",
                len(text)
            )

        except Exception:

            print(
                "TEXT LENGTH: Could not calculate"
            )

        print(
            "========================================"
        )

        print()

        # ----------------------------------
        # Return OCR result
        # ----------------------------------

        response_data = {

            "message":
                "OCR completed successfully",

            "filename":
                filename,

            "text":
                text

        }

        print(
            "OCR JSON RESPONSE:"
        )

        print(
            repr(response_data)
        )

        print()

        return jsonify(
            response_data
        ), 200

    except Exception as e:

        print()
        print("=" * 80)
        print("!!!!!!!!!!!!!!!! OCR ERROR !!!!!!!!!!!!!!!!")
        print("=" * 80)

        print(
            f"Error type: {type(e).__name__}"
        )

        print(
            f"Error message: {str(e)}"
        )

        print()
        print(
            "FULL PYTHON TRACEBACK:"
        )
        print()

        traceback.print_exc()

        print()
        print("=" * 80)
        print("!!!!!!!!!!!! END OCR ERROR !!!!!!!!!!!!!!")
        print("=" * 80)
        print()

        return jsonify({
            "error": str(e)
        }), 500


# ==========================================
# OCR Analysis
# ==========================================

@app.route("/analyze/<filename>", methods=["POST"])
def analyze_image(filename):

    # Original image path

    original_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    # Enhanced image filename

    enhanced_filename = f"enhanced_{filename}"

    # Enhanced image path

    enhanced_path = os.path.join(
        app.config["PROCESSED_FOLDER"],
        enhanced_filename
    )

    # Check original image

    if not os.path.exists(original_path):

        return jsonify({
            "error":
                "Original image not found"
        }), 404

    # Check enhanced image

    if not os.path.exists(enhanced_path):

        return jsonify({
            "error":
                "Enhanced image not found. Run enhancement first."
        }), 404

    try:

        # ----------------------------------
        # Analyze Original Image
        # ----------------------------------

        original_result = analyze_ocr(
            original_path
        )

        # ----------------------------------
        # Analyze Enhanced Image
        # ----------------------------------

        enhanced_result = analyze_ocr(
            enhanced_path
        )

        # ----------------------------------
        # Calculate Improvement
        # ----------------------------------

        confidence_improvement = round(
            enhanced_result["confidence"]
            -
            original_result["confidence"],
            2
        )

        # ----------------------------------
        # Return Results
        # ----------------------------------

        return jsonify({

            "message":
                "OCR analysis completed successfully",

            "original": {

                "filename":
                    filename,

                "text":
                    original_result["text"],

                "word_count":
                    original_result["word_count"],

                "character_count":
                    original_result["character_count"],

                "confidence":
                    original_result["confidence"],

                "processing_time":
                    original_result["processing_time"]

            },

            "enhanced": {

                "filename":
                    enhanced_filename,

                "text":
                    enhanced_result["text"],

                "word_count":
                    enhanced_result["word_count"],

                "character_count":
                    enhanced_result["character_count"],

                "confidence":
                    enhanced_result["confidence"],

                "processing_time":
                    enhanced_result["processing_time"]

            },

            "comparison": {

                "confidence_improvement":
                    confidence_improvement

            }

        }), 200

    except Exception as e:

        print()
        print("=" * 80)
        print("!!!!!!!!!!!! OCR ANALYSIS ERROR !!!!!!!!!!!!")
        print("=" * 80)

        print(
            f"Error type: {type(e).__name__}"
        )

        print(
            f"Error message: {str(e)}"
        )

        print()

        traceback.print_exc()

        print()
        print("=" * 80)
        print("!!!!!!!! END OCR ANALYSIS ERROR !!!!!!!!!")
        print("=" * 80)
        print()

        return jsonify({
            "error": str(e)
        }), 500


# ==========================================
# OCR Accuracy
# ==========================================

@app.route("/accuracy/<filename>", methods=["POST"])
def calculate_image_accuracy(filename):

    # The filename should be the ORIGINAL filename.
    #
    # Example:
    #
    # /accuracy/handwritten_ocr_test.jpg
    #
    # Backend automatically looks for:
    #
    # processed/enhanced_handwritten_ocr_test.jpg

    enhanced_filename = f"enhanced_{filename}"

    enhanced_path = os.path.join(
        app.config["PROCESSED_FOLDER"],
        enhanced_filename
    )

    # Check enhanced image

    if not os.path.exists(enhanced_path):

        return jsonify({
            "error":
                "Enhanced image not found. Run enhancement first."
        }), 404

    # Get JSON request body

    data = request.get_json()

    if not data or "reference_text" not in data:

        return jsonify({
            "error":
                "Reference text is required"
        }), 400

    reference_text = data["reference_text"]

    # Validate reference text

    if not isinstance(reference_text, str):

        return jsonify({
            "error":
                "Reference text must be a string"
        }), 400

    if not reference_text.strip():

        return jsonify({
            "error":
                "Reference text cannot be empty"
        }), 400

    try:

        # ----------------------------------
        # Get OCR Prediction
        # ----------------------------------

        predicted_text = extract_text(
            enhanced_path
        )

        # ----------------------------------
        # Calculate Accuracy
        # ----------------------------------

        accuracy = calculate_accuracy(
            reference_text,
            predicted_text
        )

        return jsonify({

            "message":
                "OCR accuracy calculated successfully",

            "reference_text":
                reference_text,

            "predicted_text":
                predicted_text,

            "accuracy":
                accuracy

        }), 200

    except Exception as e:

        print()
        print("=" * 80)
        print("!!!!!!!!!! OCR ACCURACY ERROR !!!!!!!!!!!")
        print("=" * 80)

        print(
            f"Error type: {type(e).__name__}"
        )

        print(
            f"Error message: {str(e)}"
        )

        print()

        traceback.print_exc()

        print()
        print("=" * 80)
        print("!!!!!!!! END OCR ACCURACY ERROR !!!!!!!!")
        print("=" * 80)
        print()

        return jsonify({
            "error": str(e)
        }), 500


# ==========================================
# Start Flask Application
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )