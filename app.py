"""
Flask web app for transaction fraud detection (from try.ipynb / transactions.csv).
Run train_model.py once to create model.pkl and feature_columns.pkl, then: flask run
"""
import os
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
COLS_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")
DATA_PATH = os.path.join(BASE_DIR, "transactions.csv")

model = None
feature_columns = None


def load_model():
    global model, feature_columns
    if os.path.isfile(MODEL_PATH) and os.path.isfile(COLS_PATH):
        model = joblib.load(MODEL_PATH)
        feature_columns = joblib.load(COLS_PATH)
        return True
    return False


def encode_single(row_df):
    """Encode one transaction the same way as in the notebook."""
    row_df = row_df.copy()
    row_df["transaction_time"] = pd.to_datetime(row_df["transaction_time"])
    row_df["hour"] = row_df["transaction_time"].dt.hour
    row_df["day_of_week"] = row_df["transaction_time"].dt.dayofweek
    row_df["day"] = row_df["transaction_time"].dt.day
    row_df["month"] = row_df["transaction_time"].dt.month
    row_df["is_weekend"] = (row_df["day_of_week"] >= 5).astype(int)
    row_df = row_df.drop(columns=["transaction_time"])

    categorical_cols = ["country", "bin_country", "channel", "merchant_category"]
    row_df = pd.get_dummies(row_df, columns=categorical_cols, drop_first=True)

    for col in feature_columns:
        if col not in row_df.columns:
            row_df[col] = 0
    row_df = row_df[feature_columns]
    return row_df


@app.route("/")
def index():
    if not load_model():
        return render_template("index.html", model_loaded=False, stats=None)

    stats = None
    if os.path.isfile(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        total = len(df)
        fraud_count = int(df["is_fraud"].sum()) if "is_fraud" in df.columns else 0
        fraud_rate = (df["is_fraud"].mean() * 100) if "is_fraud" in df.columns else 0
        legitimate_count = total - fraud_count
        countries_series = df["country"].value_counts().head(10) if "country" in df.columns else pd.Series()
        channels_series = df["channel"].value_counts() if "channel" in df.columns else pd.Series()
        stats = {
            "total_transactions": int(total),
            "fraud_count": int(fraud_count),
            "legitimate_count": int(legitimate_count),
            "fraud_rate_pct": round(fraud_rate, 2),
            "countries": countries_series.to_dict(),
            "channels": channels_series.to_dict(),
            "country_labels": [str(x) for x in countries_series.index],
            "country_values": [int(x) for x in countries_series.values],
            "channel_labels": [str(x) for x in channels_series.index],
            "channel_values": [int(x) for x in channels_series.values],
        }

    return render_template("index.html", model_loaded=True, stats=stats)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not load_model():
        return render_template("predict.html", model_loaded=False, result=None)

    if request.method == "GET":
        return render_template("predict.html", model_loaded=True, result=None)

    try:
        data = {
            "transaction_id": 0,
            "account_age_days": int(request.form.get("account_age_days", 0)),
            "total_transactions_user": int(request.form.get("total_transactions_user", 0)),
            "avg_amount_user": float(request.form.get("avg_amount_user", 0)),
            "amount": float(request.form.get("amount", 0)),
            "country": request.form.get("country", "FR").strip() or "FR",
            "bin_country": request.form.get("bin_country", "FR").strip() or "FR",
            "channel": request.form.get("channel", "web").strip() or "web",
            "merchant_category": request.form.get("merchant_category", "travel").strip() or "travel",
            "promo_used": int(request.form.get("promo_used", 0)),
            "avs_match": int(request.form.get("avs_match", 1)),
            "cvv_result": int(request.form.get("cvv_result", 1)),
            "three_ds_flag": int(request.form.get("three_ds_flag", 1)),
            "transaction_time": request.form.get("transaction_time", "2024-01-15T12:00:00Z"),
            "shipping_distance_km": float(request.form.get("shipping_distance_km", 0)),
        }
        row_df = pd.DataFrame([data])
        X = encode_single(row_df)
        pred = model.predict(X)[0]
        label = "Fraud" if pred == 1 else "Legitimate"
        proba = model.predict_proba(X)[0]
        fraud_prob = float(proba[1]) if len(proba) > 1 else 0.0
        return render_template(
            "predict.html",
            model_loaded=True,
            result={"label": label, "fraud_probability": round(fraud_prob * 100, 2), "raw": int(pred)},
        )
    except Exception as e:
        return render_template(
            "predict.html",
            model_loaded=True,
            result=None,
            error=str(e),
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API for prediction."""
    if not load_model():
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    try:
        data = request.get_json() or request.form
        row = {
            "transaction_id": data.get("transaction_id", 0),
            "account_age_days": int(data.get("account_age_days", 0)),
            "total_transactions_user": int(data.get("total_transactions_user", 0)),
            "avg_amount_user": float(data.get("avg_amount_user", 0)),
            "amount": float(data.get("amount", 0)),
            "country": str(data.get("country", "FR")).strip() or "FR",
            "bin_country": str(data.get("bin_country", "FR")).strip() or "FR",
            "channel": str(data.get("channel", "web")).strip() or "web",
            "merchant_category": str(data.get("merchant_category", "travel")).strip() or "travel",
            "promo_used": int(data.get("promo_used", 0)),
            "avs_match": int(data.get("avs_match", 1)),
            "cvv_result": int(data.get("cvv_result", 1)),
            "three_ds_flag": int(data.get("three_ds_flag", 1)),
            "transaction_time": str(data.get("transaction_time", "2024-01-15T12:00:00Z")),
            "shipping_distance_km": float(data.get("shipping_distance_km", 0)),
        }
        row_df = pd.DataFrame([row])
        X = encode_single(row_df)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        return jsonify({
            "prediction": "fraud" if pred == 1 else "legitimate",
            "fraud_probability": round(float(proba[1]) * 100, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
