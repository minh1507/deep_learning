python -m venv venv
source venv/bin/activate  # Hoặc venv\Scripts\activate trên Windows
pip install -r requirements.txt


# Cài thư viện
pip install -r requirements.txt

# Bước 1: Chia dữ liệu
python split_dataset.py

# Bước 2: Huấn luyện mô hình
python train.py

# Bước 3: Chạy Flask API
python app.py
