# recognize (Sử dụng mô hình đã huấn luyện để nhận diện khuôn mặt trong thời gian thực và hiển thị tên người.)

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder

# Tải mô hình và bộ mã hóa nhãn
model = load_model("face_recognition_model.h5")
label_encoder_classes = np.load("label_encoder_classes.npy")
label_encoder = LabelEncoder()
label_encoder.classes_ = label_encoder_classes

# Mở webcam
cap = cv2.VideoCapture(0)

# Tải bộ phân loại khuôn mặt (haarcascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()

    if not ret:
        print("Không thể đọc từ webcam.")
        break

    # Chuyển ảnh sang ảnh xám chỉ một lần để giảm độ trễ
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Phát hiện khuôn mặt trong ảnh
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Nếu có khuôn mặt
    if len(faces) > 0:
        # Xử lý tất cả khuôn mặt
        for (x, y, w, h) in faces:
            # Cắt khuôn mặt từ ảnh
            face = gray[y:y+h, x:x+w]
            
            # Đảm bảo khuôn mặt có kích thước (128, 128) và thêm chiều kênh
            face_resized = cv2.resize(face, (128, 128))
            face_resized = np.expand_dims(face_resized, axis=-1)  # Thêm chiều cho kênh màu xám
            face_resized = face_resized.astype("float32") / 255.0  # Chuẩn hóa giá trị pixel
            
            # Dự đoán nhãn cho khuôn mặt
            face_resized = np.expand_dims(face_resized, axis=0)  # Thêm chiều batch
            prediction = model.predict(face_resized)
            label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
            
            # Vẽ hộp bao quanh khuôn mặt và ghi nhãn lên khuôn mặt
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Hiển thị ảnh kết quả
    cv2.imshow("Face Recognition", frame)

    # Thoát khỏi vòng lặp khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()