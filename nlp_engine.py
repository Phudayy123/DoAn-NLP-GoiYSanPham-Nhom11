# nlp_engine.py - Core Engine kết nối Tiền xử lý, Phân loại SVM và Gợi ý Cosine
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from text_preprocessor import VietnameseTextPreprocessor
from model_evaluator import CategoryModelTrainer

class ProductRecommender:
    def __init__(self, data_path=config.DATA_PATH, stopwords_path=config.STOPWORDS_PATH):
        print("[1/4] Đang nạp dữ liệu từ:", data_path)
        self.df = pd.read_csv(data_path)

        print("[2/4] Đang khởi tạo bộ tiền xử lý tiếng Việt (PyVi + Stopwords)...")
        self.preprocessor = VietnameseTextPreprocessor(stopwords_path)
        self.df["clean_text"] = (
            self.df["ten_san_pham"] + " " + self.df["mo_ta_chi_tiet"]
        ).apply(self.preprocessor.preprocess)

        print("[3/4] Đang trích xuất đặc trưng TF-IDF...")
        self.vectorizer = TfidfVectorizer(**config.TFIDF_PARAMS)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["clean_text"])

        print("[4/4] Đang chia tập Train/Test và huấn luyện mô hình LinearSVC...")
        self.trainer = CategoryModelTrainer()
        self.eval_metrics = self.trainer.train_and_evaluate(
            self.tfidf_matrix, self.df["danh_muc"]
        )

        # Giữ các thuộc tính tương thích trực tiếp với app_gui.py của Đạt
        self.test_accuracy = self.eval_metrics["accuracy"]
        self.conf_matrix = self.eval_metrics["confusion_matrix"]
        self.class_report = self.eval_metrics["classification_report"]
        print(f"-> Hoàn tất! Độ chính xác kiểm thử (Test Accuracy): {self.test_accuracy*100:.2f}%")

    def preprocess_text(self, text: str) -> str:
        return self.preprocessor.preprocess(text)

    def get_recommendations(self, query: str, top_k: int = config.DEFAULT_TOP_K):
        clean_query = self.preprocess_text(query)
        if not clean_query.strip():
            return None, []

        # Biến câu truy vấn thành vector trong không gian TF-IDF
        query_vec = self.vectorizer.transform([clean_query])

        # Tầng 1: Dự đoán danh mục ngành hàng bằng mô hình SVM
        predicted_category = self.trainer.predict_category(query_vec)

        # Tầng 2: Xếp hạng sản phẩm bằng độ tương đồng góc Cosine
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score > 0.10:
                item = self.df.iloc[idx]
                results.append({
                    "id": int(item["id"]),
                    "ten_san_pham": item["ten_san_pham"],
                    "danh_muc": item["danh_muc"],
                    "gia": f"{int(item['gia']):,} VNĐ",
                    "score": f"{round(score * 100, 1)}%",
                    "mo_ta": item["mo_ta_chi_tiet"],
                    "link": item["link_san_pham"]
                })
        return predicted_category, results

# --- CHẠY TRỰC TIẾP FILE NÀY ĐỂ IN BÁO CÁO KỸ THUẬT CHI TIẾT ---
if __name__ == "__main__":
    engine = ProductRecommender()
    m = engine.eval_metrics

    print("\n" + "="*68)
    print("       BÁO CÁO THỰC NGHIỆM & ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST")
    print("="*68)
    print(f"- Tổng số mẫu dữ liệu         : {len(engine.df)} sản phẩm")
    print(f"- Kích thước tập Train (80%)  : {m['train_size']} mẫu")
    print(f"- Kích thước tập Test (20%)   : {m['test_size']} mẫu")
    print(f"- Kích thước từ điển (Vocab)  : {m['vocab_size']} đặc trưng N-grams")
    print(f"- File lưu trọng số mô hình   : {config.MODEL_SAVE_PATH}")
    print("-" * 68)
    print(f"- Accuracy trên tập Test      : {m['accuracy']*100:.2f}%")
    print(f"- Macro Precision             : {m['macro_precision']:.4f}")
    print(f"- Macro Recall                : {m['macro_recall']:.4f}")
    print(f"- Macro F1-Score              : {m['macro_f1']:.4f}")
    print("-" * 68)
    print("Ma trận nhầm lẫn (Confusion Matrix) [Laptop | Phụ kiện | Thời trang]:")
    print(m["confusion_matrix"])
    print("\nBáo cáo phân loại chi tiết (Classification Report):")
    print(m["classification_report"])