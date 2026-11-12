# model_evaluator.py - Module Huấn luyện SVM và Đánh giá Tập Test
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import config

class CategoryModelTrainer:
    def __init__(self):
        self.classifier = LinearSVC(**config.SVM_PARAMS)
        self.labels = ["Laptop", "Phụ kiện", "Thời trang"]
        self.metrics = {}

    def train_and_evaluate(self, X_tfidf, y_labels):
        # 1. Chia tập dữ liệu Stratified Train/Test (80% - 20%)
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y_labels,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_SEED,
            stratify=y_labels
        )

        # 2. Huấn luyện mô hình trên tập Train (1.200 mẫu)
        self.classifier.fit(X_train, y_train)

        # 3. Kiểm thử trên tập Test (300 mẫu)
        y_pred = self.classifier.predict(X_test)

        # 4. Tổng hợp các chỉ số đánh giá học thuật
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred, labels=self.labels)
        report = classification_report(y_test, y_pred, digits=4)

        self.metrics = {
            "train_size": X_train.shape[0],
            "test_size": X_test.shape[0],
            "vocab_size": X_tfidf.shape[1],
            "accuracy": acc,
            "macro_precision": prec,
            "macro_recall": rec,
            "macro_f1": f1,
            "confusion_matrix": cm,
            "classification_report": report
        }

        # Lưu mô hình đã huấn luyện xuống ổ cứng
        joblib.dump(self.classifier, config.MODEL_SAVE_PATH)
        return self.metrics

    def predict_category(self, query_vec) -> str:
        return self.classifier.predict(query_vec)[0]