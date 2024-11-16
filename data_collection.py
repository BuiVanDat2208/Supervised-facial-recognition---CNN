# data_collection
import cv2
import os

def collect_data(person_name):
    # Tạo thư mục riêng cho người dùng nếu chưa tồn tại
    person_folder = f"data/{person_name}"
    os.makedirs(person_folder, exist_ok=True)
    
    # Khởi tạo webcam
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    count = 0
    images_captured = []  # Danh sách lưu tên ảnh đã chụp
    
    while count < 100:  # Chụp 100 ảnh cho mỗi người
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            image_path = f"{person_folder}/{count}.jpg"
            cv2.imwrite(image_path, face)
            images_captured.append(image_path)  # Thêm đường dẫn ảnh vào danh sách
            count += 1
        
        cv2.imshow("Collecting Data", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Ghi thông tin ảnh đã chụp vào file metadata.txt
    with open(f"{person_folder}/metadata.txt", "w") as metadata_file:
        for image_path in images_captured:
            metadata_file.write(f"{image_path}\n")

# Chạy chương trình với tên của người muốn thêm vào dataset
person_name = input("Nhập tên người: ")
collect_data(person_name)
