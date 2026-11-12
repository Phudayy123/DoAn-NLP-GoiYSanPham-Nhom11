# config.py - Cấu hình tham số toàn hệ thống và Siêu tham số (Hyperparameters)

DATA_PATH = "data.csv"
STOPWORDS_PATH = "vietnamese_stopwords.txt"
MODEL_SAVE_PATH = "svm_tfidf_pipeline.pkl"

# Tham số chia tập dữ liệu Huấn luyện / Kiểm thử
TEST_SIZE = 0.20          # 20% Test (300 mẫu), 80% Train (1.200 mẫu)
RANDOM_SEED = 42          # Cố định hạt giống ngẫu nhiên

# Siêu tham số cho TF-IDF Vectorizer
TFIDF_PARAMS = {
    "ngram_range": (1, 2), # Bắt cả từ đơn (unigram) và cụm 2 từ (bigram)
    "min_df": 2,           # Bỏ qua từ xuất hiện ít hơn 2 lần
    "max_df": 0.95,        # Bỏ qua từ xuất hiện ở hơn 95% văn bản
    "sublinear_tf": True   # Áp dụng thang đo logarit cho TF: 1 + log(tf)
}

# Siêu tham số cho mô hình Support Vector Machine (LinearSVC)
SVM_PARAMS = {
    "C": 1.0,              # Hệ số điều chuẩn (Regularization parameter)
    "dual": "auto",
    "max_iter": 2000,
    "random_state": RANDOM_SEED
}

# Số lượng sản phẩm gợi ý mặc định
DEFAULT_TOP_K = 5