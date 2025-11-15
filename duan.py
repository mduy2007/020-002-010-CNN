# Dự án nhận diện khuôn mặt
# Thành viên: Đặng Minh Duy, Lê Hoàng Ân, Nguyễn Hoàng Gia Minh

# Cài đặt thư viện
!pip install -q insightface onnxruntime-gpu opencv-python matplotlib

# Kết nối Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Import thư viện
import insightface
import cv2
import os
import numpy as np
from google.colab import files
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Đường dẫn dataset
drive_path = "/content/drive/MyDrive/DuAnAI"
dataset_path = os.path.join(drive_path, "dataset")
model_path = os.path.join(drive_path, "model.pkl")

# Tạo dataset và thư mục con cho từng người
os.makedirs(dataset_path, exist_ok=True)
for name in ["duy", "aan", "minh", "unknown"]:
    os.makedirs(os.path.join(dataset_path, name), exist_ok=True)
    print(f"Đã tạo thư mục: {name}")

# Khởi tạo model
model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
model.prepare(ctx_id=0)

# Hàm upload ảnh
def upload_images_simple(folder_name):
    print(f"up ảnh lên cho {folder_name}...")
    uploaded = files.upload()

    if not uploaded:
        print("ko có ảnh nào được tải lên")
        return 0

    count = 0
    for filename in uploaded.keys():
        new_name = f"{folder_name}_{count+1}_{filename}"
        file_content = uploaded[filename]
        new_path = os.path.join(dataset_path, folder_name, new_name)
        with open(new_path, 'wb') as f:
          f.write(file_content)
        count += 1
        if count % 10 == 0:
            print(f"đã upload {count} ảnh...")

    print(f"đã up ảnh cho: {folder_name}")
    return count

# Hàm xây dựng database
def build_face_database():
    print("tiến hành xây dựng database")
    face_database = []

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"đang xử lý: {person_name}")
        image_files = [f for f in os.listdir(person_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        processed_count = 0
        for image_file in image_files[:100]:
            image_path = os.path.join(person_dir, image_file)
            try:
                img = cv2.imread(image_path)
                if img is None:
                    continue

                # Phát hiện và mã hóa khuôn mặt
                faces = model.get(img)
                if len(faces) > 0:
                    face = faces[0]  # Lấy khuôn mặt đầu tiên
                    embedding = face.embedding

                    face_database.append({
                        'name': person_name,
                        'embedding': embedding,
                        'bbox': face.bbox,
                        'image_path': image_path
                    })
                    processed_count += 1

            except Exception as e:
                continue

        print(f" {person_name}: {processed_count} khuôn mặt")

    print(f"khởi tạo database thành công")
    return face_database

# Hàm nhận diện khuôn mặt
def recognize_faces_insightface():
    print("\nup ảnh lên...")
    uploaded = files.upload()

    if not uploaded:
        print("không có ảnh nào được tải lên")
        return

    image_filename = list(uploaded.keys())[0]
    print(f"Đang xử lý: {image_filename}")


    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            face_database = pickle.load(f)
        print("Đã load database từ file")
    else:
        face_database = build_face_database()
        with open(model_path, 'wb') as f:
            pickle.dump(face_database, f)
        print("Đã xây dựng database mới")

    # Đọc và xử lý ảnh
    img = cv2.imread(image_filename)
    if img is None:
        print("không thể đọc ảnh")
        return

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Phát hiện khuôn mặt
    print("đang tìm khuôn mặt...")
    faces = model.get(img)

    if len(faces) == 0:
        print("Không tìm thấy khuôn mặt nào!")
        plt.figure(figsize=(10, 6))
        plt.imshow(rgb_img)
        plt.axis('off')
        plt.title("Không tìm thấy khuôn mặt")
        plt.show()
        return

    print(f"Đã Tìm thấy {len(faces)} khuôn mặt")

    # Nhận diện từng khuôn mặt
    for i, face in enumerate(faces):
        current_embedding = face.embedding
        bbox = face.bbox.astype(int)

        # So sánh với database
        best_match = "Unknown"
        best_score = 0

        for db_face in face_database:
            similarity = cosine_similarity(
                [current_embedding],
                [db_face['embedding']]
            )[0][0]

            if similarity > best_score and similarity > 0.65:  # Ngưỡng similarity
                best_score = similarity
                best_match = db_face['name']

        # Hiển thị kết quả
        if best_match == "Unknown":
            color = (0, 0, 255)  # Đỏ
            status = "X"
        else:
            color = (0, 255, 0)  # Xanh
            status = "+"

        print(f"{status} khuôn mặt {i+1}: {best_match} (độ tương đồng: {best_score:.3f})")

        # Vẽ kết quả
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        label = f"{best_match} ({best_score:.2f})"
        y = y1 - 10 if y1 - 10 > 10 else y1 + 30
        cv2.putText(img, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Hiển thị ảnh kết quả
    plt.figure(figsize=(12, 8))
    result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(result_img)
    plt.axis('off')
    plt.title("kết quả nhận diện khuôn mặt")
    plt.show()

# Hàm kiểm tra dữ liệu
def check_database():
    print("\nKIỂM TRA DATABASE:")
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            face_database = pickle.load(f)

        name_counts = {}
        for face in face_database:
            name = face['name']
            name_counts[name] = name_counts.get(name, 0) + 1

        for name, count in name_counts.items():
            print(f"{name}: {count} khuôn mặt")
        print(f"Tổng: {len(face_database)} khuôn mặt")
    else:
        print("ko có dâtbase")

# MENU CHÍNH
print("MENU HỆ THỐNG ")
print("1.Upload ảnh training")
print("2.Nhận diện ảnh")
print("3.Kiểm tra database")

choice = input("Chọn đi: ")

if choice == "1":
    person = input("nhập tên: ")
    upload_images_simple(person)

elif choice == "2":
    recognize_faces_insightface()

elif choice == "3":
    check_database()

else:
    print("lựa chọn không hợp lệ")