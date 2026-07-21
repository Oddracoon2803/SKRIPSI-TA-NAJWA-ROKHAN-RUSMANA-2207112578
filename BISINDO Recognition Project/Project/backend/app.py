from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import keras
import joblib

# =======================
# CONFIG
# =======================
CONF_THRESHOLD = 0.6

# =======================
# INIT APP
# =======================
app = Flask(__name__)
CORS(app)

# =======================
# LOAD MODEL & ASSETS
# =======================
model = keras.models.load_model("bisindo_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

labels = label_encoder.classes_

print("✅ Model ready")
print("Classes:", labels)
print("Input shape:", model.input_shape)


# =======================
# PREPROCESS (CONSISTENT DENGAN TRAINING)
# =======================
def preprocess_landmarks(data):
    try:
        X = np.array(data, dtype=np.float32)

        # =========================
        # VALIDASI PANJANG DATA
        # =========================
        if len(X) != 126:
            raise ValueError(f"Invalid landmark length: {len(X)} (expected 126)")

        # =========================
        # RESHAPE
        # =========================
        X = X.reshape(1, 126)

        # =========================
        # SCALING (WAJIB)
        # =========================
        X = scaler.transform(X)

        return X

    except Exception as e:
        raise ValueError(f"Preprocess error: {str(e)}")


# =======================
# ENDPOINT PREDICT
# =======================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json.get("landmarks", [])
        hand_count = request.json.get("hand_count", None)

        # =====================
        # VALIDASI INPUT
        # =====================
        if not data or len(data) != 126:
            return jsonify({
                "label": "-",
                "conf": 0.0
            })

        # =====================
        # PREPROCESS
        # =====================
        X = preprocess_landmarks(data)

        # =====================
        # PREDIKSI
        # =====================
        pred = model.predict(X, verbose=0)[0]

        idx = int(np.argmax(pred))
        conf = float(np.max(pred))
        label = str(labels[idx])

        # =====================
        # CONFIDENCE FILTER
        # =====================
        if conf < CONF_THRESHOLD:
            return jsonify({
                "label": "-",
                "conf": conf
            })

        return jsonify({
            "label": label,
            "conf": conf
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })


# =======================
# HEALTH CHECK (OPSIONAL)
# =======================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "model_input_shape": model.input_shape,
        "num_classes": len(labels)
    })


# =======================
# RUN SERVER
# =======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)