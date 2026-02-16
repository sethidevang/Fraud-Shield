"""
Train fraud detection model and save it for the Flask app.
Run this once (or when you retrain) before starting the web app.
"""
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier


def encode_transactions(df):
    df = df.copy()
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    df['hour'] = df['transaction_time'].dt.hour
    df['day_of_week'] = df['transaction_time'].dt.dayofweek
    df['day'] = df['transaction_time'].dt.day
    df['month'] = df['transaction_time'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df.drop(columns=['transaction_time'], inplace=True)

    categorical_cols = ['country', 'bin_country', 'channel', 'merchant_category']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "transactions.csv")

    print("Loading data...")
    df = pd.read_csv(csv_path)

    print("Encoding features...")
    df_encoded = encode_transactions(df)

    X = df_encoded.drop(columns=['is_fraud', 'user_id'])
    y = df_encoded['is_fraud']
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Applying SMOTE...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    print("Training Random Forest (Optimized for size)...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_res, y_train_res)

    model_path = os.path.join(base_dir, "model.pkl")
    cols_path = os.path.join(base_dir, "feature_columns.pkl")

    print(f"Saving model with compression...")
    joblib.dump(model, model_path, compress=3)
    joblib.dump(feature_columns, cols_path)

    print(f"Model saved to {model_path}")
    print(f"Feature columns ({len(feature_columns)}) saved to {cols_path}")


if __name__ == "__main__":
    main()
