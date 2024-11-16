# # train_model (Huấn luyện mô hình nhận diện khuôn mặt.)

import os
from pyexpat import model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt  # Thư viện vẽ biểu đồ

# Hàm tải và tiền xử lý dữ liệu ảnh
def load_images(data_dir="dataz", target_size=(128, 128)):
    images, labels = [], []
    for person in os.listdir(data_dir):
        person_folder = os.path.join(data_dir, person)
        if os.path.isdir(person_folder):
            for filename in os.listdir(person_folder):
                file_path = os.path.join(person_folder, filename)
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img = load_img(file_path, target_size=target_size, color_mode='grayscale')
                        img_array = img_to_array(img)
                        images.append(img_array)
                        labels.append(person)
                    except Exception as e:
                        print(f"Lỗi khi tải ảnh {filename}: {e}")
    images = np.array(images) / 255.0  # Chuẩn hóa ảnh
    return images, labels

# Hàm xây dựng mô hình CNN
def build_model(input_shape=(128, 128, 1), num_classes=2):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')  # Số lớp đầu ra bằng số nhãn duy nhất
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Hàm huấn luyện mô hình
def train_model(data_dir="data"):
    # Tải và tiền xử lý dữ liệu
    images, labels = load_images(data_dir)
    
    # Kiểm tra và đảm bảo rằng ảnh có độ sâu kênh là 1 (grayscale)
    if len(images.shape) == 3:
        images = images.reshape(images.shape[0], 128, 128, 1)
    
    # Chuyển nhãn thành dạng số và phân loại theo dạng one-hot
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)
    
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

    # Xây dựng mô hình
    model = build_model(input_shape=(128, 128, 1), num_classes=len(label_encoder.classes_))

    # Huấn luyện mô hình và lưu lịch sử huấn luyện
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Lưu mô hình và bộ mã hóa nhãn
    model.save("face_recognition_model.h5")
    np.save("label_encoder_classes.npy", label_encoder.classes_)

    print("Mô hình đã được huấn luyện và lưu thành công!")

    # Vẽ biểu đồ Loss và Accuracy
    # Biểu đồ Loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Biểu đồ Accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Đánh giá mô hình trên dữ liệu kiểm tra
    score = model.evaluate(X_test, y_test, verbose=1)
    print(f"Độ chính xác trên dữ liệu kiểm tra: {score[1] * 100:.2f}%")

# Gọi hàm huấn luyện mô hình khi chạy script
if __name__ == "__main__":
    train_model()
    