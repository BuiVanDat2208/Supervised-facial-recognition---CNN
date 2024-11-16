# # Test_model (Kiểm tra mô hình)

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder

# Tải mô hình đã huấn luyện
model = load_model("face_recognition_model.h5")

# Tải bộ mã hóa nhãn (label encoder)
label_encoder_classes = np.load("label_encoder_classes.npy")
label_encoder = LabelEncoder()
label_encoder.classes_ = label_encoder_classes

# Đường dẫn đến thư mục dữ liệu testing
testing_dir = 'D:/BaiTapXu Ly Anh/Supervised facial recognition - CNN/data'

# Tạo danh sách để chứa ảnh và nhãn từ dữ liệu testing
testing_images = []
testing_labels = []

# Đọc ảnh từ thư mục testing
for person_name in os.listdir(testing_dir):
    person_folder = os.path.join(testing_dir, person_name)
    if os.path.isdir(person_folder):  # Kiểm tra xem đây có phải là thư mục của một người không
        for img_name in os.listdir(person_folder):  # Duyệt tất cả các ảnh trong thư mục của mỗi người
            img_path = os.path.join(person_folder, img_name)
            
            # Đọc ảnh và chuyển thành ảnh grayscale
            img = cv2.imread(img_path)
            if img is None:
                continue  # Nếu không đọc được ảnh, bỏ qua
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Phát hiện khuôn mặt trong ảnh
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (128, 128))  # Đảm bảo kích thước giống như lúc huấn luyện
                face = img_to_array(face) / 255.0  # Chuẩn hóa ảnh (tỉ lệ 0-1)
                testing_images.append(face)
                testing_labels.append(person_name)  # Gán nhãn tương ứng với tên người

# Chuyển dữ liệu thành numpy array
testing_images = np.array(testing_images)
testing_labels = np.array(testing_labels)

# Chuyển nhãn thành số (label encoding) để có thể sử dụng trong việc đánh giá mô hình
testing_labels = label_encoder.transform(testing_labels)

# Đánh giá mô hình với dữ liệu testing
loss, accuracy = model.evaluate(testing_images, testing_labels, verbose=1)

# In ra cả loss và accuracy
print(f"Loss on testing data: {loss:.4f}")
print(f"Accuracy on testing data: {accuracy * 100:.2f}%")

# Vẽ biểu đồ độ chính xác (accuracy) của mô hình trên dữ liệu testing
# Tạo danh sách chứa độ chính xác
accuracies = [accuracy * 100]  # Độ chính xác đạt được trên dữ liệu testing

# Vẽ biểu đồ
plt.plot(accuracies, marker='o')  # Thêm marker để dễ thấy điểm trên biểu đồ
plt.title("Accuracy on Testing Data")
plt.xlabel("Test")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.show()
