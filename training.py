import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import numpy as np

# 1. Baca dataset dari file CSV
data = pd.read_excel('D:\\Training New\\Extraksi_Warna_Cabai(dataset_new).xlsx')

# 2. Ekstrak fitur HSV serta label kondisi
hsv_features = data[['Avg H', 'Avg S', 'Avg V']].values
labels = data['Label'].values

# 3. Pisahkan data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(hsv_features, labels, test_size=0.3, random_state=42)

# 4. Normalisasi data untuk meningkatkan performa model
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Inisialisasi model Naive Bayes dan latih model
nb = GaussianNB()
nb.fit(X_train, y_train)

# 6. Prediksi dan evaluasi model
y_pred = nb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Menghitung probabilitas prediksi untuk menghitung log_loss
y_pred_proba = nb.predict_proba(X_test)

# Menghitung log loss
loss = log_loss(y_test, y_pred_proba)

# Menampilkan akurasi
print("Accuracy:", accuracy)

# Menampilkan log loss
print("Log Loss:", loss)

# Menghitung dan menampilkan matriks konfusi
print("\nConfusion Matrix:")
print(conf_matrix)

# Menampilkan laporan klasifikasi (precision, recall, f1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Busuk', 'Sehat']))

# 7. Tampilkan Confusion Matrix dalam bentuk heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Busuk', 'Sehat'], yticklabels=['Busuk', 'Sehat'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix of Naive Bayes Model")
plt.show()

# 8. Fungsi untuk memprediksi kondisi air dari gambar
def predict_image(image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (224, 224))  # Sesuaikan ukuran sesuai kebutuhan
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
    avg_hsv = np.average(np.average(hsv_image, axis=0), axis=0)
    features = avg_hsv.reshape(1, -1)
    features = scaler.transform(features)
    prediction = nb.predict(features)
    return prediction[0]

# 9. Contoh prediksi dari gambar input
image_path = 'test-busuk2.jpg'  # Ganti dengan path gambar Anda
prediksi_kondisi = predict_image(image_path)
print("Cabai ini:", prediksi_kondisi)
