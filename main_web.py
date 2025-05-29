import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify, url_for
import re
from main import PhotoStylizer
import os

app = Flask(__name__, static_folder="static", template_folder="templates")
for sub in ("uploads", "stylized", "qr"):
    path = os.path.join(app.static_folder, sub)
    os.makedirs(path, exist_ok=True)

# ———————— Routes ————————
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    data_url = request.json["image"]

    payload = request.get_json()
    email   = payload.get("email", "").strip() 
    safe_email = re.sub(r"[^\w\-\.]", "_", email)

    header,encoded = data_url.split(",",1)
    jpg = base64.b64decode(encoded)
    arr = np.frombuffer(jpg, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)


    stylizer = PhotoStylizer(
        upload_dir=os.path.join(app.static_folder, "uploads"),
        stylized_dir=os.path.join(app.static_folder, "stylized"),
        qr_dir=os.path.join(app.static_folder, "qr"),
    )
    presigned_url, styl_path, qr_path = stylizer.run(frame, safe_email)
    qr_url = url_for("static", filename=f"qr/{qr_path}")
    return jsonify({
        "stylized_url": presigned_url,
        "stylized_path": styl_path,
        "qr_path": qr_url
    })

if __name__=="__main__":
    app.run(debug=True)
