from rembg import remove
from PIL import Image
import os

# Folder input dan output
input_folder = 'D:\\Training New\\dataset_tomat\\mentah'   # Ganti dengan nama folder input Anda
output_folder = 'D:\\Training New\\mentah_remove_bg' # Ganti dengan nama folder output Anda

# Membuat folder output jika belum ada
os.makedirs(output_folder, exist_ok=True)

# Memproses semua file gambar dalam folder input
for filename in os.listdir(input_folder):
    if filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"no_bg_{filename}")
        
        # Membaca gambar
        with open(input_path, 'rb') as i:
            input_image = i.read()
        
        # Menghapus background
        output_image = remove(input_image)
        
        # Menyimpan hasil ke file
        with open(output_path, 'wb') as o:
            o.write(output_image)
        
        print(f"Background berhasil dihapus untuk {filename}")

print("Proses selesai.")
