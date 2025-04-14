python -m venv venv
source venv/bin/activate  # Hoặc venv\Scripts\activate trên Windows
pip install -r requirements.txt


# Cài thư viện
pip install -r requirements.txt

# Bước 1: crawl data
python main.py

# Bước 2: Huấn luyện mô hình
Test nhỏ với 500 dòng đầu	python train/main.py --mode test
Train chunk bất kỳ (ví dụ 0)	python train/main.py --mode chunk --chunk 0
Train toàn bộ từ chunk 1 đến 9	python train/main.py --mode full

# Bước 3: Chạy Flask API
python source/main.py
