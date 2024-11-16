#preprocess_data (Xử lý dữ liệu ảnh để chuẩn bị cho việc huấn luyện.)

import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def load_images():
    images, labels = [], []
    for person in os.listdir("data"):
        person_folder = os.path.join("data", person)
        # Kiểm tra xem đây có phải là thư mục không
        if os.path.isdir(person_folder):
            for filename in os.listdir(person_folder):
                file_path = os.path.join(person_folder, filename)
                # Kiểm tra xem file có phải là ảnh không (theo phần mở rộng)
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        # Tải ảnh và chuyển thành dạng mảng numpy
                        img = load_img(file_path, target_size=(128, 128), color_mode='grayscale')
                        img_array = img_to_array(img)
                        images.append(img_array)
                        labels.append(person)
                    except Exception as e:
                        print(f"Lỗi khi tải ảnh {filename}: {e}")
                else:
                    print(f"Đã bỏ qua file không phải ảnh: {filename}")
        else:
            print(f"Thư mục không hợp lệ: {person_folder}")
    
    # Chuyển mảng ảnh thành numpy array và chuẩn hóa giá trị ảnh (giảm xuống từ 0-255 thành 0-1)
    images = np.array(images) / 255.0
    return images, labels

# Kiểm tra hàm load_images
if __name__ == "__main__":
    images, labels = load_images()
    print(f"Số lượng ảnh đã tải: {images.shape[0]}")
    print(f"Số lượng nhãn duy nhất: {len(set(labels))}")
