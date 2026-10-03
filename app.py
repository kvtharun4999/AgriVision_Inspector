"""
AgriVision Turmeric Inspector - Flask Application
-------------------------------------------------
Academic Mini-Project for Turmeric Farmers in Erode & Perundurai, Tamil Nadu.
Integrated with Google Stitch UI templates.

Routes:
  /         : Upload studio page (upload.html)
  /predict  : Analyzes uploaded leaf, performs 3-tier validation, returns result.html
              or invalid_warning.html
  /history  : Shows past inspection scans (history.html)
  /report/pdf: Downloads printable PDF diagnostic dossier
"""

import os
import io
import json
import uuid
import random
import logging
from datetime import datetime
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from dotenv import load_dotenv

import api_client

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# Strict 5 MB upload limit
MAX_FILE_BYTES = 5 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_BYTES
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Directories & Storage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "turmeric_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

UPLOADS_DIR = os.path.join(BASE_DIR, "static", "uploads")
DATA_DIR = os.path.join(BASE_DIR, "data")
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Model state
tf_model = None
class_names = []


def load_model_and_classes():
    """Load trained Keras model and class names if available."""
    global tf_model, class_names

    if os.path.exists(CLASS_NAMES_PATH):
        try:
            with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                class_names = json.load(f)
                logger.info(f"Loaded class names ({len(class_names)}): {class_names}")
        except Exception as e:
            logger.warning(f"Could not load class names from {CLASS_NAMES_PATH}: {e}")

    if os.path.exists(MODEL_PATH):
        try:
            import tensorflow as tf
            tf_model = tf.keras.models.load_model(MODEL_PATH)
            logger.info(f"Loaded trained model from {MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Failed to load model from {MODEL_PATH}: {e}. Running in mock mode.")
            tf_model = None
    else:
        logger.info("Model file not yet found at 'model/turmeric_model.keras'. Running in mock mode.")
        tf_model = None


load_model_and_classes()


# -------------------------------------------------------------
# History Management
# -------------------------------------------------------------
def load_history():
    """Load stored scan records from data/history.json."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed reading history.json: {e}")
    return []


def save_history_scan(entry):
    """Prepend new scan to history.json and persist."""
    scans = load_history()
    scans.insert(0, entry)
    if len(scans) > 50:
        scans.pop()
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(scans, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed saving scan to history.json: {e}")


def wants_json():
    """Check if the requesting client expects JSON response (AJAX/API)."""
    accept = request.headers.get("Accept", "")
    x_req = request.headers.get("X-Requested-With", "")
    return "application/json" in accept or x_req == "XMLHttpRequest"


# -------------------------------------------------------------
# OpenCV Plant Color & Area Checks
# -------------------------------------------------------------
def check_plant_foliage(img_bgr, min_leaf_ratio=0.05):
    """
    Step 2: Plant check (OpenCV)
    Checks if image contains sufficient green, yellow, or brown leaf-colored pixels.
    Returns (has_leaf_color, ratio).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Green / yellow-green foliage
    lower_green = np.array([15, 20, 20], dtype=np.uint8)
    upper_green = np.array([95, 255, 255], dtype=np.uint8)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Yellow / tan / brown leaf tones (including necrotic spots)
    lower_brown = np.array([5, 25, 20], dtype=np.uint8)
    upper_brown = np.array([30, 255, 240], dtype=np.uint8)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    leaf_mask = cv2.bitwise_or(mask_green, mask_brown)
    total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
    leaf_pixels = cv2.countNonZero(leaf_mask)
    ratio = leaf_pixels / max(1, total_pixels)

    logger.info(f"Plant foliage pixel ratio: {ratio:.3f} (min required: {min_leaf_ratio})")
    return (ratio >= min_leaf_ratio), ratio


def calculate_infected_area(img_bgr):
    """
    Calculates infected leaf area percentage using OpenCV HSV segmentation:
    infected % = (diseased pixels / total leaf pixels) * 100.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Healthy / green foliage
    lower_green = np.array([20, 30, 30], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)
    leaf_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Diseased brown / necrotic spots
    lower_spot1 = np.array([5, 40, 25], dtype=np.uint8)
    upper_spot1 = np.array([24, 255, 210], dtype=np.uint8)
    spot_mask1 = cv2.inRange(hsv, lower_spot1, upper_spot1)

    lower_spot2 = np.array([0, 25, 20], dtype=np.uint8)
    upper_spot2 = np.array([180, 255, 65], dtype=np.uint8)
    spot_mask2 = cv2.inRange(hsv, lower_spot2, upper_spot2)

    disease_mask = cv2.bitwise_or(spot_mask1, spot_mask2)
    total_leaf = cv2.bitwise_or(leaf_mask, disease_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    total_leaf = cv2.morphologyEx(total_leaf, cv2.MORPH_CLOSE, kernel)
    spots_on_leaf = cv2.bitwise_and(disease_mask, total_leaf)

    leaf_pixels = cv2.countNonZero(total_leaf)
    infected_pixels = cv2.countNonZero(spots_on_leaf)

    if leaf_pixels == 0:
        return 0.0

    pct = (infected_pixels / float(leaf_pixels)) * 100.0
    return round(float(pct), 1)


def mock_predict(img_bgr, leaf_ratio):
    """Deterministic fallback predictor if TensorFlow model is missing."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_brown = np.array([5, 40, 25], dtype=np.uint8)
    upper_brown = np.array([24, 255, 210], dtype=np.uint8)
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    brown_ratio = cv2.countNonZero(brown_mask) / max(1, img_bgr.shape[0] * img_bgr.shape[1])

    if brown_ratio > 0.08:
        return "Leaf Spot", 88.5
    elif brown_ratio > 0.03:
        return "Leaf Blotch", 84.2
    else:
        return "Healthy Leaf", 93.8


# -------------------------------------------------------------
# Routes
# -------------------------------------------------------------
@app.route("/")
def upload_page():
    """
    Route 1: / shows the upload studio page.
    Renders Stitch-designed upload.html with drag-and-drop & camera capture.
    """
    return render_template("upload.html")


@app.route("/history")
def history_page():
    """
    Route 3: /history shows past scans log.
    Renders Stitch-designed history.html with saved scans.
    """
    scans = load_history()
    return render_template("history.html", scans=scans)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Route 2: /predict takes the uploaded image and returns the result page.
    Implements 3 sequential checks:
      a. File check (valid image, under 5 MB)
      b. Plant check (OpenCV leaf color pixels)
      c. Model check (confidence >= 70%, not negative/rhizome)

    On rejection: returns "Please upload a correct turmeric leaf image."
    On success: returns result.html with OpenCV severity & TNAU disease advice.
    """
    reject_msg = "Please upload a correct turmeric leaf image."

    def reject(reason=""):
        logger.warning(f"Rejection: {reason}")
        if wants_json():
            return jsonify({
                "is_turmeric_leaf": False,
                "message": reject_msg,
                "reason": reason
            }), 200
        return render_template("invalid_warning.html", message=reject_msg, reason=reason)

    # --- Check A: File Check ---
    if "image" not in request.files and "file" not in request.files:
        return reject("No photo file was found in the upload request.")

    file = request.files.get("image") or request.files.get("file")
    if not file or file.filename == "":
        return reject("No image file was selected.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return reject(f"File format '{ext}' is not supported. Please upload a JPG, PNG, or WEBP photo.")

    file_bytes = file.read()
    if len(file_bytes) > MAX_FILE_BYTES:
        return reject("File exceeds the 5 MB limit. Please upload a photo under 5 MB.")

    try:
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Corrupt image data")
    except Exception as e:
        return reject(f"Image could not be decoded: {e}")

    # Save uploaded image for viewing on result page
    safe_filename = f"{uuid.uuid4().hex[:12]}{ext}"
    saved_path = os.path.join(UPLOADS_DIR, safe_filename)
    try:
        with open(saved_path, "wb") as f:
            f.write(file_bytes)
        image_url = f"/static/uploads/{safe_filename}"
    except Exception as e:
        logger.warning(f"Could not persist upload: {e}")
        image_url = "/static/stitch_assets/result_dashboard.png"

    # --- Check B: Plant Check (OpenCV) ---
    has_foliage, leaf_ratio = check_plant_foliage(img_bgr, min_leaf_ratio=0.05)
    if not has_foliage:
        return reject("Insufficient leaf foliage pixels detected. Please photograph a green, yellow, or brown turmeric leaf.")

    # --- Check C: Model Check ---
    global tf_model, class_names
    if tf_model is None and os.path.exists(MODEL_PATH):
        load_model_and_classes()

    is_mock = False
    top_class = None
    top_confidence = 0.0

    if tf_model is not None and class_names:
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_array = np.expand_dims(img_resized, axis=0).astype(np.float32)
            preds = tf_model.predict(img_array, verbose=0)[0]
            top_idx = int(np.argmax(preds))
            top_confidence = float(preds[top_idx] * 100.0)
            top_class = class_names[top_idx]
            logger.info(f"Model prediction: '{top_class}' with {top_confidence:.1f}% confidence.")
        except Exception as e:
            logger.error(f"Inference failure: {e}. Falling back to mock engine.")
            top_class, top_confidence = mock_predict(img_bgr, leaf_ratio)
            is_mock = True
    else:
        top_class, top_confidence = mock_predict(img_bgr, leaf_ratio)
        is_mock = True

    # 1. Reject if confidence < 70%
    if top_confidence < 70.0:
        return reject(f"Low match confidence ({top_confidence:.1f}% < 70%). Please photograph under uniform daylight.")

    # 2. Reject if predicted class is Not_turmeric_leaf
    cls_lower = top_class.lower().replace("_", " ")
    if "not turmeric" in cls_lower:
        return reject("Identified as non-turmeric material. Please upload a valid turmeric leaf or plant image.")

    # =========================================================
    # Crop Confirmed: Proceed to Disease Analysis and API call
    # =========================================================
    clean_disease = top_class.replace("_", " ")
    if "rhizome" in clean_disease.lower() and "disease" in clean_disease.lower():
        clean_disease = "Rhizome Disease Root"
    elif "rhizome" in clean_disease.lower() and "healthy" in clean_disease.lower():
        clean_disease = "Rhizome Healthy Root"
    elif "healthy" in clean_disease.lower():
        clean_disease = "Healthy Leaf"
    elif "blotch" in clean_disease.lower():
        clean_disease = "Leaf Blotch"
    elif "spot" in clean_disease.lower():
        clean_disease = "Leaf Spot"
    elif "dry" in clean_disease.lower():
        clean_disease = "Dry Leaf"
    elif "aphid" in clean_disease.lower():
        clean_disease = "Aphids Disease"

    # Severity and Infected Area %
    raw_infected_pct = calculate_infected_area(img_bgr)
    if "healthy" in clean_disease.lower():
        severity = "Healthy"
        infected_area_pct = 0.0
    else:
        infected_area_pct = raw_infected_pct
        if infected_area_pct < 10.0:
            severity = "Mild"
        elif 10.0 <= infected_area_pct <= 30.0:
            severity = "Moderate"
        else:
            severity = "Severe"

    # Call external API (or cached / local fallback from diseases.json)
    disease_info = api_client.fetch_disease_info(clean_disease)

    scan_id = f"AV-{random.randint(1000, 9999)}"
    scan_time = datetime.now().strftime("%b %d, %Y • %I:%M %p")

    # Record scan in history
    history_entry = {
        "id": scan_id,
        "disease": clean_disease,
        "scientific_name": disease_info.get("scientific_name", "Curcuma longa"),
        "confidence": round(top_confidence, 1),
        "severity": severity,
        "infected_area_pct": infected_area_pct,
        "yield_loss": disease_info.get("yield_loss", "Cited: 15% - 25%"),
        "date": scan_time,
        "image_url": image_url
    }
    save_history_scan(history_entry)

    payload = {
        "is_turmeric_leaf": True,
        "leaf_confirmed_message": "Turmeric leaf confirmed",
        "disease": clean_disease,
        "confidence": round(top_confidence, 1),
        "severity": severity,
        "scientific_name": disease_info.get("scientific_name", "Curcuma longa"),
        "cause": disease_info.get("cause", ""),
        "symptoms": disease_info.get("symptoms", []),
        "infected_area_pct": infected_area_pct,
        "yield_loss": disease_info.get("yield_loss", "15% - 25%"),
        "solution": disease_info.get("solution", []),
        "prevention": disease_info.get("prevention", []),
        "sources": disease_info.get("sources", []),
        "tamil": disease_info.get("tamil", {}),
        "is_offline": disease_info.get("is_offline", False),
        "offline_note": disease_info.get("offline_note", None),
        "is_mock": is_mock,
        "image_url": image_url,
        "scan_id": scan_id,
        "scan_time": scan_time
    }

    if wants_json():
        return jsonify(payload), 200

    return render_template("result.html", **payload)


@app.route("/report/pdf", methods=["POST"])
def generate_pdf_report():
    """Generate and download a printable PDF report using ReportLab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        data = request.get_json() or {}
        disease = data.get("disease", "Turmeric Leaf Inspection")
        scientific_name = data.get("scientific_name", "")
        confidence = data.get("confidence", "N/A")
        severity = data.get("severity", "N/A")
        infected_pct = data.get("infected_area_pct", "0")
        yield_loss = data.get("yield_loss") or "Not cited"
        cause = data.get("cause", "N/A")
        symptoms = data.get("symptoms", [])
        solution = data.get("solution", [])
        prevention = data.get("prevention", [])

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "TitleStyle",
            parent=styles["Heading1"],
            fontSize=22,
            textColor=colors.HexColor("#2F6B3A"),
            spaceAfter=4
        )
        subtitle_style = ParagraphStyle(
            "SubTitleStyle",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.HexColor("#566952"),
            spaceAfter=12
        )
        h2_style = ParagraphStyle(
            "H2Style",
            parent=styles["Heading2"],
            fontSize=13,
            textColor=colors.HexColor("#2F6B3A"),
            spaceBefore=10,
            spaceAfter=4
        )
        body_style = ParagraphStyle(
            "BodyStyle",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.HexColor("#23301F"),
            leading=14
        )

        elements = []
        elements.append(Paragraph("AgriVision Turmeric Inspector", title_style))
        elements.append(Paragraph("Official Turmeric Leaf Diagnostic Report &bull; Erode &amp; Perundurai", subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#C98A0B"), spaceAfter=14))

        elements.append(Paragraph(f"<b>Diagnosis:</b> {disease} <i>({scientific_name})</i>", body_style))
        elements.append(Paragraph(f"<b>Confidence:</b> {confidence}% &nbsp;|&nbsp; <b>Severity:</b> {severity} &nbsp;|&nbsp; <b>Infected Leaf Area:</b> {infected_pct}%", body_style))
        if yield_loss and yield_loss != "Not cited":
            elements.append(Paragraph(f"<b>Estimated Yield Loss:</b> {yield_loss}", body_style))
        elements.append(Spacer(1, 8))

        elements.append(Paragraph("<b>Cause:</b>", h2_style))
        elements.append(Paragraph(cause, body_style))
        elements.append(Spacer(1, 6))

        elements.append(Paragraph("<b>Symptoms Seen:</b>", h2_style))
        for s in symptoms:
            elements.append(Paragraph(f"&bull; {s}", body_style))
        elements.append(Spacer(1, 6))

        elements.append(Paragraph("<b>Recommended Treatment (Solution):</b>", h2_style))
        for step in solution:
            elements.append(Paragraph(step, body_style))
        elements.append(Spacer(1, 6))

        elements.append(Paragraph("<b>Preventive Measures:</b>", h2_style))
        for p in prevention:
            elements.append(Paragraph(f"&bull; {p}", body_style))
        elements.append(Spacer(1, 14))

        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#DCCEBA"), spaceAfter=8))
        elements.append(Paragraph("<i>Advisory Note: For guidance only. Confirm treatment with your local agriculture officer or Krishi Vigyan Kendra (KVK).</i>", subtitle_style))

        doc.build(elements)
        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"AgriVision_Report_{disease.replace(' ', '_')}.pdf",
            mimetype="application/pdf"
        )
    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        return jsonify({"error": "Failed to generate PDF", "detail": str(e)}), 500


@app.errorhandler(413)
def handle_large_file(error):
    msg = "Please upload a correct turmeric leaf image."
    if wants_json():
        return jsonify({
            "is_turmeric_leaf": False,
            "message": msg,
            "reason": "File size exceeds the 5 MB limit."
        }), 200
    return render_template("invalid_warning.html", message=msg, reason="File exceeds the 5 MB limit. Please select a photo under 5 MB.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting AgriVision Turmeric Inspector on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
