import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Đường dẫn file dữ liệu đầu vào và kết quả đầu ra
CSV_PATH = "urfd_dataset.csv"
MODEL_PATH = "models/fall_classifier.pkl"
SCALER_PATH = "models/scaler.pkl"
os.makedirs("models", exist_ok=True)

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: Data file not found '{CSV_PATH}'. Please run 'python prepare_dataset.py' first!")
        return

    print("--- Loading data from CSV ---")
    df = pd.read_csv(CSV_PATH)
    print(f"Dataset shape: {df.shape}")

    # Tách đặc trưng (X) và nhãn (y)
    # Đặc trưng gồm 34 tọa độ khớp + angle + aspect_ratio
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    # Chia tập dữ liệu thành tập huấn luyện (80%) và tập kiểm thử (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # Chuẩn hóa đặc trưng (Standardization)
    print("\n--- Scaling features ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Huấn luyện mô hình SVM (Support Vector Machine) với kernel RBF (tiêu chuẩn cho phi tuyến tính)
    print("\n--- Training SVM model ---")
    clf = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Dự đoán trên tập kiểm thử để đánh giá mô hình
    y_pred = clf.predict(X_test_scaled)

    # Tính toán các chỉ số đánh giá
    accuracy = accuracy_score(y_test, y_pred)
    print("\n=== MODEL EVALUATION RESULTS ===")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal (ADL)", "Fall"]))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"[[True Normal: {cm[0][0]:4d}, False Fall:      {cm[0][1]:4d}]")
    print(f" [False Normal: {cm[1][0]:4d}, True Fall:       {cm[1][1]:4d}]]")

    # Lưu mô hình huấn luyện và scaler lại thành file .pkl
    print("\n--- Saving model and scaler ---")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"Model saved at: '{MODEL_PATH}'")
    print(f"Scaler saved at: '{SCALER_PATH}'")
    print("\nTraining completed successfully! You now have a trained Fall Detection ML model.")

if __name__ == '__main__':
    main()
