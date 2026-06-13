import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_customer_churn(input_path, output_path):
    """
    Melakukan preprocessing dataset Telco Customer Churn secara otomatis.
    Tahapan dibuat konsisten dengan notebook eksperimen:
    1. load data
    2. membersihkan kolom customerID
    3. konversi TotalCharges menjadi numerik
    4. imputasi missing value
    5. encoding target Churn
    6. one-hot encoding fitur kategorikal
    7. scaling fitur numerik
    8. menyimpan dataset siap latih
    """
    df = pd.read_csv(input_path)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Imputasi missing value
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encoding target
    if "Churn" in df.columns:
        if df["Churn"].dtype == "object":
            df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)

    # One-hot encoding fitur kategorikal selain target
    cat_features = [c for c in df.select_dtypes(include=["object", "category", "bool"]).columns if c != "Churn"]
    df = pd.get_dummies(df, columns=cat_features, drop_first=True)

    # Scaling fitur numerik utama
    scale_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in df.columns]
    if scale_cols:
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    preprocess_customer_churn(
        "customer_churn_raw/customer_churn_raw.csv",
        "preprocessing/customer_churn_preprocessing/churn_preprocessed.csv"
    )
    print("Preprocessing selesai. File tersimpan di preprocessing/customer_churn_preprocessing/churn_preprocessed.csv")
