# 🛡️ Fraud Shield: Transaction Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Vercel](https://img.shields.io/badge/Vercel-Deployment-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://vercel.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

A professional end-to-end machine learning application for detecting fraudulent transactions. This project features a clean Flask-based dashboard, real-time risk scoring, and a pre-trained Random Forest model optimized for imbalanced datasets.

## 🚀 Key Features

- **Interactive Dashboard**: Visualize transaction statistics, fraud rates, and distribution across channels/countries.
- **Real-time Prediction**: specialized form to input transaction metadata and receive an instant probability-based risk score.
- **Robust ML Pipeline**: 
    - Random Forest Classifier for high-accuracy detection.
    - SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance.
    - Automated feature engineering for temporal data.
- **RESTful API**: Ready-to-use JSON endpoint for external integrations.
- **Vercel Optimized**: Pre-configured for seamless serverless deployment.

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Data Engineering**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, Imbalanced-learn
- **Model Persistence**: Joblib
- **Frontend**: HTML5, Vanilla CSS3 (Modern Glassmorphism UI)

## 📦 Project Structure

```text
Fraud/
├── app.py                  # Main Flask Server & API
├── train_model.py          # Model Training & Serialization Script
├── model.pkl               # Saved Random Forest Model
├── feature_columns.pkl      # Stored Feature Names for Consistency
├── templates/              # Beautiful Responsive UI Templates
├── transactions.csv        # Dataset for Training
├── requirements.txt        # Project Dependencies
└── vercel.json             # Vercel Deployment Configuration
```

## 🏁 Getting Started

### 1. Prerequisites
- Python 3.9 or higher
- Pip (Python package manager)

### 2. Installation
```bash
# Clone the repository (if applicable)
# git clone https://github.com/yourusername/fraud-shield.git
# cd fraud-shield

# Install dependencies
pip install -r requirements.txt
```

### 3. Training the Model
If you need to retrain the model with fresh data:
```bash
python train_model.py
```

### 4. Running Locally
```bash
# Using the integrated runner
python app.py

# OR using Flask CLI
flask run
```
The app will be available at `http://127.0.0.1:5000`.

## 🌐 API Usage

**POST** `/api/predict`

Request Body:
```json
{
  "amount": 50.00,
  "account_age_days": 100,
  "country": "FR",
  "channel": "web",
  "transaction_time": "2024-01-15T12:00:00Z"
}
```

Response:
```json
{
  "prediction": "legitimate",
  "fraud_probability": 12.45
}
```

## ☁️ Deployment

This project is configured for **Vercel**. To deploy:
1. Push this code to a GitHub repository.
2. Connect the repository to Vercel.
3. Vercel will automatically detect the `vercel.json` and deploy using the `@vercel/python` builder.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Note: The model included (`model.pkl`) is highly optimized and compressed (~2.5MB), making it ideal for standard GitHub pushes and Vercel serverless deployments.*
