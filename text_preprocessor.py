# text_preprocessor.py - Module chuẩn hóa và tách từ tiếng Việt
import re
from pyvi import ViTokenizer
import config

class VietnameseTextPreprocessor:
    def __init__(self, stopwords_path=config.STOPWORDS_PATH):
        self.stopwords = self._load_stopwords(stopwords_path)

    def _load_stopwords(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            return set()

    def normalize(self, text: str) -> str:
        """Bước 1 & 2: Chuyển chữ thường và lọc ký tự đặc biệt bằng Regex"""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        # Giữ lại chữ cái tiếng Việt và chữ số cấu hình (12gb, i5, 144hz)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize_and_remove_stopwords(self, text: str) -> str:
        """Bước 3 & 4: Tách từ ghép PyVi và loại bỏ Stopwords"""
        tokenized_text = ViTokenizer.tokenize(text)
        valid_words = [
            word for word in tokenized_text.split() 
            if word not in self.stopwords and len(word) > 1
        ]
        return " ".join(valid_words)

    def preprocess(self, text: str) -> str:
        """Pipeline tiền xử lý hoàn chỉnh cho 1 văn bản"""
        norm_text = self.normalize(text)
        return self.tokenize_and_remove_stopwords(norm_text)