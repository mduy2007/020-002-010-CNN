# Hệ Thống Nhận Diện Khuôn Mặt

# **Mục Lục** 
- [Giới thiệu](#giới-thiệu)
- [Hướng dẫn cài đặt](#hướng-dẫn-cài-đặt---chuẩn-bị-môi-trường)
- [Giải Thích Code](#giải-thích-code)
- [Hướng Dẫn Sử Dụng Code](#hướng-dẫn-sử-dụng-code)

## **Giới thiệu:**
- Được Thực Hiện Bởi Nhóm 1:
    + Đặng Minh Duy.
    + Lê Hoàng Ân.
    + Nguyễn Hoàng Gia Minh.

- **Đề tài:** Sử dụng công nghệ CNN ứng dụng cho nhận diện khuôn mặt.

- **Môi Trường:** Google Colab
- **Ngôn ngữ:** Python
- **Tính Năng Chính:** 
    + Đa khuôn mặt: Phát hiện nhiều khuôn mặt trong 1 ảnh.
    + So sánh thông minh: Được so sánh với mức ngưỡng cao nên rất chính xác
    + Lưu Trữ Qua Database và được liên kết với google drive: Lưu trữ qua database có sẵn giúp không phải train lại mỗi lần chạy mà chỉ cần quét thẳng dữ liệu với lần train đầu - liên kết google drive giúp dữ liệu trong local không phải reset.
- **Công nghệ được sử dụng:** 
    - Model: InsightFace
    - Computer Vision: OpenCV
    - Similarity: Cosine Similarity từ scikit-learn
    - Storage: Google Drive + Pickle serialization
    - Visualization: Matplotlib
## **Hướng dẫn cài đặt - chuẩn bị môi trường**
- Bước đầu hãy truy cập *[google colab](https://colab.research.google.com/)* và tạo 1 sổ tay mới.
- Tiếp theo hãy tuy chỉnh môi trường *google colab* của bạn thành sử dụng GPU **(để tốc độ load nhanh hơn)** 
- Hãy tạo 1 file để chuẩn bị chứa dữ liệu, ví dụ như ``` DuAnAI ``` trong phần code

## **Cài Đặt 1 Số Công Nghệ Cần Thiết Cho Môi Trường**
- **Cài Đặt Môi Trường:** 
    ```python
     !pip install -q insightface onnxruntime-gpu opencv-python matplotlib
     ```
     Cài Đặt Môi trường được đưa vào code vì google colab có **giới hạn** thời gian mỗi ngày, và khi hết time hoặc rời đi quá lâu - môi trường sẽ tự động **reset** mất hết dữ liệu local.

- **Liên kết *google drive:***

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
    Phần `code` này sẽ giúp chúng ta liên kết với google drive.

- **Tiếp theo chạy code như bình thường**

## **Cấu Trúc Dự Án:**

```python
/content/drive/MyDrive/DuAnAI/
├── dataset/
│   ├── duy/           # Ảnh training Duy
│   ├── aan/           # Ảnh training Ân
│   ├── minh/          # Ảnh training Minh
│   └── unknown/       # Ảnh unknown faces
└── model.pkl          # Database embeddings
```
## **Giải Thích Code:**
- **Note:** Phần Này Khá Dài, bạn có thể bỏ qua phần này để xem phần hướng dẫn dùng code, bạn có thể quay lại phần [Mục Lục](#mục-lục) hoặc [Cách Dùng Code](#hướng-dẫn-sử-dụng-code) tại đây.


### **Kết nối Thư Viện**
``` python
import insightface # Kết Nối Với model
import cv2 # xử lí ảnh
import os  # tải tệp
import numpy as np # xử lý các dữ liệu của bài toán phức tạp
from google.colab import files  # up  ảnh lên môi trường collab
import matplotlib.pyplot as plt # hiển thị ảnh đã xử lí
import pickle   # dùng để lưu và tải database chứa các dữ liệu mà máy đã train
from sklearn.metrics.pairwise import cosine_similarity # so sánh độ tương đồng từ ảnh test và ảnh đc train
```
- **Chi Tiết:**
    - ***insightface:*** kết nối mô hình đã được train sẵn.
    - ***CV2:*** Đọc - xử lí ảnh, chuyển hóa ảnh thành `RGB` và vẽ `Box - Text` cho đầu ra.
    - ***OS:*** Quản Lí Hệ Thống File, tạo thư mục, kiểm tra, liệt kê file
    - ***Numpy:*** Tính Toán Khoa Học, mảng đa chiều.
    - ***File:*** Hiện Cửa Sổ Chọn File và Lấy File Tải Lên google Collab
    - ***Matplotlib:*** Vẽ Biểu Đồ - Hiển Thị Ảnh
    - ***Pickle***: Lưu Trữ Dữ Liệu Đã Mã Hóa
    - ***cosine_similarity:*** Ngưỡng so sánh các vector trong ảnh
---
### **Kết Nối Với Drive - Kho Dữ Liệu:**
```python
drive_path = "/content/drive/MyDrive" # Đường dẫn đến file Chính Của Drive
dataset_path = os.path.join(drive_path, "dataset") #Đường Dẫn Đến Kho Dữ Liệu
model_path = os.path.join(drive_path, "model.pkl")  #Đường Dẫn đến model đã huấn luyện
```
- **Chi Tiết:**
    - Kết Nối Với 3 Hệ Thống:
        + Drive
        + Kho Dữ Liệu
        + Model đã train
---
### **Tạo Cơ Sở Dữ Liệu Cho Từng Thành Viên Để Lưu Trữ Ảnh:**
```python
    os.makedirs(dataset_path, exist_ok=True) # Hàm tạo dữ liệu nếu chưa có
    for name in ["duy", "aan", "minh", "unknown"]:
    os.makedirs(os.path.join(dataset_path, name), exist_ok=True) # Tương tự
    print(f"Đã tạo thư mục: {name}") # Thông báo nếu đã tạo xong
```
- **Chi Tiết:**
    + Tự Động Tạo Thư Mục Với Hàm` os.makedirs`
    + Vòng lặp quét qua 3 cái tên: duy, aan, minh, unknown để tạo dữ liệu.
---
### Khởi Tạo Model:
```python
model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
model.prepare(ctx_id=0)
```
- **Chi Tiết:**
    - Khởi tạo model để tiến hành thu thập dữ liệu:
---
### **Hàm Upload Ảnh:**
```python
def upload_images_simple(folder_name):
    print(f"up ảnh lên cho {folder_name}...")
    uploaded = files.upload()
```
- Khởi Tạo Hàm upload ảnh
```py
    if not uploaded:
        print("ko có ảnh nào được tải lên")
        return 0
```
- Thông Báo Nếu Người Dùng Không Up Ảnh Lên
```py
count = 0
    for filename in uploaded.keys():
        new_name = f"{folder_name}_{count+1}_{filename}"
        file_content = uploaded[filename]
        new_path = os.path.join(dataset_pa
        th, folder_name, new_name)
        with open(new_path, 'wb') as f:
          f.write(file_content)
        count += 1
        if count % 10 == 0:
            print(f"đã upload {count} ảnh...")
```
- **Chi Tiết:**
    - Sửa tên file để tránh file bị trùng lặp, file được đổi tên sẽ theo cấu trúc `tên folder_số lượng_tên gốc của file` 
    - ví dụ như: `duy_1_anhcuaduy.jpg`
    - Thông báo cho người dùng biết họ đã up ảnh và up bao nhiêu ảnh.
---
### **Xây Dựng Database:**
```py
def build_face_database():
    print("tiến hành xây dựng database")
    face_database = []
```
- **Chi Tiết:**
    - Xây Dựng Database - Khởi Tạo 1 mảng rỗng để chứa database

```py
for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue
```
- **Chi Tiết:**
    - Đường dẫn vào file phía trong:
        + `prerson_name` tên folder của 1 người ví dụ như: duy, aan, minh, unknown
        + `person_dir` tên file cố định nằm trong `person_name`
    - Nếu không có bất kì file nào trong 1 thư mục sẽ bỏ qua và tiếp tục thực thi sang thư mục khác

```py
 print(f"đang xử lý: {person_name}")
        image_files = [f for f in os.listdir(person_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
```
- **Chi Tiết:**
    + In ra thông báo đang xử lí để thông báo cho người dùng biết hệ thống đang xử lí
    + Phân Tích Và tách các file có đuôi khác `jpg` `jpeg` `png` để tránh mã hóa cả các dự liệu không liên quan
    
```py
processed_count = 0
        for image_file in image_files[:100]:
            image_path = os.path.join(person_dir, image_file)
            try:
                img = cv2.imread(image_path)
                if img is None:
                    continue
```
- **Chi Tiết:**
    - Đặt Giới Hạn Cho Model mỗi khi lấy chỉ lấy tối đa 100 ảnh để tránh hệ thống quá tải
    - Đặt Đường dẫn đến file ảnh để hệ thống theo đường dẫn trích xuất file để mã hóa.
    - `try` để đọc các file ảnh, nếu các file ảnh bị hỏng, lỗi, sai định dạng,... thì sẽ bỏ qua
    - Đọc ảnh và trả về kết quả, nếu quá trình đọc ảnh có vấn đề nghiêm trọng sẽ trả về kết quả none

```py
# Phát hiện và mã hóa khuôn mặt
    faces = model.get(img)
        if len(faces) > 0:
            face = faces[0]  # Lấy khuôn mặt đầu tiên
            embedding = face.embedding
```
- **Chi Tiết:**
    - Phát hiện khuôn mặt trong ảnh, sau đó mã hóa thành embedding.
```py
face_database.append({
    'name': person_name, #Tên
    'embedding': embedding, #Dữ liệu đã mã hóa
    'bbox': face.bbox,  #tọa độ 
    'image_path': image_path #địa chỉ ảnh
         })
         processed_count += 1   #biến đếm

            except Exception as e:
                continue
```
- **Chi Tiết:**
    - Cấu Trúc Lưu Trữ File

```py
print(f" {person_name}: {processed_count} khuôn mặt")

    print(f"khởi tạo database thành công")
    return face_database
```
- **Chi tiết:**
    - Cấu Trúc Database.
    - thông báo đã nhận được bao nhiêu khuôn mặt và khởi tạo xong database.
---
## **Nhận Diện Khuôn Mặt**
```py
def recognize_faces_insightface():
    print("\nup ảnh lên...")
    uploaded = files.upload()
```
- **Chi Tiết:**
    - Khởi Tạo Hàm Nhận diện, yêu cầu người dùng up ảnh.
- *Trong trường hợp người dùng không up thì sẽ in kết quả:*
```py
if not uploaded:
        print("không có ảnh nào được tải lên")
        return
```
- Hệ Thống sẽ tự động gửi thông báo và dừng chương trình.
```py
image_filename = list(uploaded.keys())[0]
    print(f"Đang xử lý: {image_filename}")
```
- Load File vừa tải lên để hệ thống tiến hành xử lí.

```py
if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            face_database = pickle.load(f)
        print("Đã load database từ file")
    else:
        face_database = build_face_database()
        with open(model_path, 'wb') as f:
            pickle.dump(face_database, f)
        print("Đã xây dựng database mới")
```
- **Chi Tiết:**
    - Load Dữ Liệu Từ DataBase, nếu chưa có sẽ gọi hàm `build_face_database()` để tạo lại 1 database mới
- **Ưu Điểm:** Tiến Kiệm Tài Nguyên, Thời Gian Chạy, Không Cần Phải Xây Lại Database Ngay Từ Đầu Cho Mỗi Lần Quét
    ## **Bảng So Sánh:**

| Phương thức | Thời gian | Mức Hoạt Động CPU/GPU | Phù hợp |
|-------------|-----------|-----------------------|----------|
| **Xây Lại** | 3-5p |  Cao | Lần đầu, thay đổi dataset |
| **Load từ file** | 1 - 3s | Thấp | Các lần chạy sau |

```py
  # Đọc và xử lý ảnh
    img = cv2.imread(image_filename)
    if img is None:
        print("không thể đọc ảnh")
        return
```
- **Chi Tiết:** 
    - Đọc và Xử Lý Ảnh, Nếu Không xử lý được ảnh sẽ trả về kết quả `none` và in ra thông báo, dừng chương trình.

```py
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```
- Hàm Chuyển Từ BGR sang RGB

### **Kết nối với model**
```py
 # Phát hiện khuôn mặt
    print("đang tìm khuôn mặt...")
    faces = model.get(img)
```
- Hàm Này kết nối với model, để tìm khuôn mặt trong ảnh mới tải `(img)` và tìm khuôn mặt trong ảnh đó

### **Nhận Diện khuôn Mặt Trong Ảnh**

#### **1. Duyệt qua từng khuôn mặt phát hiện được:**
```python
for i, face in enumerate(faces):
```
- **`faces`**: Danh sách các khuôn mặt đã phát hiện trong ảnh
- **`enumerate(faces)`**: Vừa lấy khuôn mặt, vừa lấy index `i`
- **Mục đích:** Xử lý **từng khuôn mặt một** (có thể có nhiều người trong 1 ảnh)

#### **2. Trích xuất đặc trưng và tọa độ:**
```python
current_embedding = face.embedding
bbox = face.bbox.astype(int)
```
- **`face.embedding`**: Vector đặc trưng 512 chiều của khuôn mặt hiện tại
- **`face.bbox`**: Tọa độ bounding box (vị trí khuôn mặt trong ảnh)
- **Mục đích:** Chuẩn bị dữ liệu để so sánh và vẽ kết quả

#### **3. Khởi tạo biến so sánh:**
```python
best_match = "Unknown"
best_score = 0
```
- **`best_match`**: Tên người được nhận diện (mặc định "Unknown")
- **`best_score`**: Độ tương đồng cao nhất tìm được
- **Mục đích:** Theo dõi kết quả tốt nhất trong quá trình so sánh


#### **4. Duyệt qua database để so sánh:**
```python
for db_face in face_database:
```
- **`face_database`**: Danh sách tất cả khuôn mặt đã training
- **Mục đích:** So sánh khuôn mặt hiện tại với **từng khuôn mặt trong database**

#### **5. Tính độ tương đồng cosine:**
```python
similarity = cosine_similarity(
    [current_embedding],          # Vector khuôn mặt cần nhận diện
    [db_face['embedding']]        # Vector khuôn mặt trong database
)[0][0]                           # Lấy giá trị số từ ma trận 1x1
```

**Giải thích cosine similarity:**
- **Khoảng cách:** -1 (hoàn toàn khác) → 1 (giống hệt)
- **> 0.9:** Rất giống nhau
- **0.65-0.8:** Khá giống nhau  
- **< 0.5:** Khác nhau

#### **6. Logic tìm kết quả tốt nhất:**
```python
if similarity > best_score and similarity > 0.65:
    best_score = similarity
    best_match = db_face['name']
```

### **Hiển Thị Kết Quả**
```py
# Hiển thị kết quả
        if best_match == "Unknown":
            color = (0, 0, 255)  # Đỏ
            status = "X"
        else:
            color = (0, 255, 0)  # Xanh
            status = "+"

        print(f"{status} khuôn mặt {i+1}: {best_match} (độ tương đồng: {best_score:.3f})")
```
- **Giải Thích:**
    - Nếu kết quả sau khi nhận diện không thuộc diện nằm trong kho dữ liệu mà lại nằm trong mục `Unknown` thì dữ sẽ vẽ khung in ra màu đỏ - tượng trừng cho người lạ.
    - Ngược lại, in ra màu xanh.

### **Vẽ Khung Kết Quả**

```py
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
```
- **Chi Tiết:**
    - Vẽ Khung Xanh/Đỏ Và Hiển Thị Khung Ảnh Đã Xử Lí
---
### **Hàm check database**    

#### **1. Khai báo hàm:**
```python
def check_database():
```
**Mục đích:** Tạo hàm để kiểm tra trạng thái database

#### **2. Thông báo bắt đầu:**
```python
print("\nKIỂM TRA DATABASE:")
```
**Kết quả:** 
```
KIỂM TRA DATABASE:
```

#### **3. Kiểm tra file database tồn tại:**
```python
if os.path.exists(model_path):
```
- **`model_path`**: Đường dẫn đến file `model.pkl`
- **`os.path.exists()`**: Kiểm tra file có tồn tại không
- **Mục đích:** Xác định xem database đã được tạo chưa

---

### **Xử lí khi database đã tồn tại**

#### **4. Load database từ file:**
```python
with open(model_path, 'rb') as f:
    face_database = pickle.load(f)
```
- **`'rb'`**: Read binary - đọc file nhị phân
- **`pickle.load(f)`**: Giải nén dữ liệu từ file pickle
- **Mục đích:** Khôi phục database từ file đã lưu
---
## **Hướng Dẫn Sử Dụng Code**
### **Menu Chính Của Code**
```
MENU HỆ THỐNG NHẬN DIỆN KHUÔN MẶT
1. Upload ảnh training
2. Nhận diện ảnh
3. Kiểm tra database
```
#### **1. Upload ảnh training**
- Nhập tên người
- Upload ảnh
- Hệ thống tự động xử lý và lưu vào dataset  

#### **2. nhận diện ảnh:**
- Upload ảnh cần nhận diện
    - Hệ thống sẽ:
        - Phát hiện khuôn mặt
        - So sánh với database - nếu không có database hệ thống sẽ xử lí và tự động train lại database mới
        - Hiển thị kết quả với bounding box
        - Hiển thị độ tương đồng

#### **Sau Khi Làm Xong 2 Phần Trên, Ta Vào Drive check đã có dữ liệu chưa, nếu đã có thì code đã chạy thành công**

## Ví Dụ Ouput Khi Code Chạy Thành Công

![data_test.jpg](https://sv2.anhsieuviet.com/2025/11/15/data_test.jpg)
