import pandas as pd
import numpy as np
import cv2
import os

# Kondisi tomat matang
path_matang = "D:\\Training New\\matang_remove_bg"
data_matang = os.listdir(path_matang)
NamaFile = []
AvgH = []
AvgS = []
AvgV = []
Label = []

for gbr in data_matang:
    gbr_read = cv2.imread(os.path.join(path_matang, gbr))
    gbr_hsv = cv2.cvtColor(gbr_read, cv2.COLOR_BGR2HSV)
    
    # Ekstraksi rata-rata HSV
    meanH = np.mean(gbr_hsv[:, :, 0])
    meanS = np.mean(gbr_hsv[:, :, 1])
    meanV = np.mean(gbr_hsv[:, :, 2])
    
    NamaFile.append(gbr)  # Tambahkan nama file
    AvgH.append(meanH)
    AvgS.append(meanS)
    AvgV.append(meanV)
    Label.append(1)  # Matang

# Dataframe untuk kondisi matang
data_matang = pd.DataFrame({
    'Nama File': NamaFile,
    'Avg H': AvgH,
    'Avg S': AvgS,
    'Avg V': AvgV,
    'Label': Label
})

# Kondisi tomat mentah
path_mentah = "D:\\Training New\\mentah_remove_bg"
data_mentah = os.listdir(path_mentah)
NamaFile = []
AvgH = []
AvgS = []
AvgV = []
Label = []

for gbr in data_mentah:
    gbr_read = cv2.imread(os.path.join(path_mentah, gbr))
    gbr_hsv = cv2.cvtColor(gbr_read, cv2.COLOR_BGR2HSV)
    
    # Ekstraksi rata-rata HSV
    meanH = np.mean(gbr_hsv[:, :, 0])
    meanS = np.mean(gbr_hsv[:, :, 1])
    meanV = np.mean(gbr_hsv[:, :, 2])
    
    NamaFile.append(gbr)  # Tambahkan nama file
    AvgH.append(meanH)
    AvgS.append(meanS)
    AvgV.append(meanV)
    Label.append(0)  # Mentah

# Dataframe untuk kondisi mentah
data_mentah = pd.DataFrame({
    'Nama File': NamaFile,
    'Avg H': AvgH,
    'Avg S': AvgS,
    'Avg V': AvgV,
    'Label': Label
})

# Gabungkan kedua dataframe
total = pd.concat([data_matang, data_mentah], ignore_index=True)

# Ekspor ke file Excel
total.to_excel("Extraksi_Warna_Tomat(rembg).xlsx", index=False)

print("Ekstraksi fitur berhasil, hasil disimpan dalam file 'Extraksi_Warna_Tomat(rembg).xlsx'")
