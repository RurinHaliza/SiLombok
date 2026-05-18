import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from rembg import remove  # Library untuk menghapus latar belakang
from PIL import Image
import io

# Memuat model dan scaler yang telah disimpan
model = joblib.load('model_cabai.pkl')
scaler = joblib.load('scaler_cabai.pkl')

# Fungsi untuk menampilkan gambar dengan Matplotlib
def show_image_with_matplotlib(image, title="Hasil Preprocessing"):
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')  # Menyembunyikan sumbu
    plt.show()

# Fungsi untuk menghapus latar belakang menggunakan rembg
def remove_background(image_path):
    # Membaca file gambar sebagai byte
    with open(image_path, 'rb') as input_file:
        input_image = input_file.read()
        output_image = remove(input_image)  # Menghapus latar belakang
    
    # Memuat gambar hasil penghapusan latar belakang ke dalam PIL
    image_no_bg = Image.open(io.BytesIO(output_image)).convert("RGB")
    return np.array(image_no_bg)  # Konversi ke format numpy array

# Fungsi untuk mengekstrak nilai rata-rata HSV dari gambar
def extract_hsv_features(image_path):
    # Menghapus latar belakang gambar
    image_no_bg = remove_background(image_path)

    # Menampilkan gambar hasil penghapusan latar belakang
    show_image_with_matplotlib(image_no_bg, title="Gambar Setelah rembg (RGB)")

    # Mengonversi gambar dari RGB ke HSV melalui BGR (OpenCV memerlukan BGR)
    image_no_bg_bgr = cv2.cvtColor(image_no_bg, cv2.COLOR_RGB2BGR)  # RGB ke BGR
    hsv_image = cv2.cvtColor(image_no_bg_bgr, cv2.COLOR_BGR2HSV)    # BGR ke HSV

    # Menampilkan hasil konversi HSV
    hsv_preview = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)  # HSV ke RGB untuk ditampilkan
    show_image_with_matplotlib(hsv_preview, title="Gambar Setelah Konversi ke HSV")

    # Menghitung nilai rata-rata H, S, dan V (hanya area dengan cabai)
    avg_hue = np.mean(hsv_image[:, :, 0][hsv_image[:, :, 0] > 0])
    avg_saturation = np.mean(hsv_image[:, :, 1][hsv_image[:, :, 1] > 0])
    avg_value = np.mean(hsv_image[:, :, 2][hsv_image[:, :, 2] > 0])

    return [avg_hue, avg_saturation, avg_value]

# Fungsi untuk prediksi kondisi cabai berdasarkan gambar
def predict_from_image(image_path):
    # Ekstrak fitur HSV dari gambar
    hsv_features = extract_hsv_features(image_path)
    if hsv_features is None:
        return "Gagal memproses gambar."

    # Menstandarisasi data baru dengan scaler yang telah disimpan
    hsv_features_scaled = scaler.transform([hsv_features])

    # Melakukan prediksi
    prediction = model.predict(hsv_features_scaled)
    
    # Mengembalikan hasil prediksi (sehat atau busuk)
    return "Sehat" if prediction[0] == 1 else "Busuk"

# Path gambar cabai untuk diuji (gantilah dengan path gambar Anda)
image_path = 'test-sehat4.jpg'

# Melakukan prediksi kondisi cabai dari gambar
result = predict_from_image(image_path)

# Menampilkan hasil prediksi
print(f"Hasil Prediksi: Cabai ini {result}")
