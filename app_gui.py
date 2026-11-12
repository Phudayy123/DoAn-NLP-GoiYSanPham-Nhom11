import tkinter as tk
from tkinter import ttk, messagebox
import webbrowser
from nlp_engine import ProductRecommender

class RecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ Thống Phân Loại SVM & Gợi Ý Sản Phẩm - Đồ Án NLP")
        self.root.geometry("1020x640")
        self.root.configure(padx=15, pady=15)

        try:
            self.engine = ProductRecommender()
        except Exception as e:
            messagebox.showerror("Lỗi dữ liệu", f"Không tìm thấy hoặc lỗi cấu trúc file data.csv!\nChi tiết: {e}")
            self.root.destroy()
            return

        self.create_widgets()

    def create_widgets(self):
        # 1. Tiêu đề & Thông tin Huấn luyện Train/Test
        title_label = tk.Label(
            self.root, 
            text="HỆ THỐNG PHÂN LOẠI NGÀNH HÀNG (SVM) & GỢI Ý SẢN PHẨM (TF-IDF)", 
            font=("Arial", 14, "bold"),
            fg="#1a73e8"
        )
        title_label.pack(pady=(0, 5))

        acc_text = f"Mô hình: LinearSVC (Train: 1.200 mẫu | Test: 300 mẫu) — Độ chính xác kiểm thử (Test Accuracy): {self.engine.test_accuracy*100:.2f}%"
        acc_label = tk.Label(self.root, text=acc_text, font=("Arial", 10, "italic"), fg="#2e7d32")
        acc_label.pack(pady=(0, 12))

        # 2. Khung nhập nhu cầu tìm kiếm
        search_frame = tk.LabelFrame(self.root, text=" Nhập nhu cầu của bạn bằng ngôn ngữ tự nhiên ", font=("Arial", 10, "bold"), padx=10, pady=10)
        search_frame.pack(fill="x", pady=(0, 10))

        self.query_entry = tk.Entry(search_frame, font=("Arial", 12))
        self.query_entry.pack(side="left", fill="x", expand=True, padx=(0, 10), ipady=5)
        self.query_entry.bind("<Return>", lambda event: self.search_products())

        search_btn = tk.Button(
            search_frame, 
            text="Phân loại & Gợi ý", 
            font=("Arial", 11, "bold"), 
            bg="#1a73e8", 
            fg="white", 
            padx=15, 
            pady=5,
            cursor="hand2",
            command=self.search_products
        )
        search_btn.pack(side="right")

        # Nhãn hiển thị kết quả dự đoán ngành hàng của mô hình SVM
        self.pred_label = tk.Label(
            self.root, 
            text="Ngành hàng dự đoán bởi mô hình SVM: [Chưa có truy vấn]", 
            font=("Arial", 11, "bold"), 
            fg="#d93025",
            anchor="w"
        )
        self.pred_label.pack(fill="x", pady=(0, 8))

        # 3. Bảng hiển thị danh sách sản phẩm gợi ý (Treeview)
        table_frame = tk.LabelFrame(self.root, text=" Top 5 sản phẩm phù hợp nhất (Nhấp đúp vào dòng để mở Link mua hàng) ", font=("Arial", 10, "bold"), padx=10, pady=10)
        table_frame.pack(fill="both", expand=True, pady=(0, 12))

        columns = ("id", "danh_muc", "ten", "gia", "score", "link")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
        
        self.tree.heading("id", text="ID")
        self.tree.heading("danh_muc", text="Danh Mục")
        self.tree.heading("ten", text="Tên Sản Phẩm")
        self.tree.heading("gia", text="Giá Bán")
        self.tree.heading("score", text="Độ Tương Đồng")
        self.tree.heading("link", text="Link Sản Phẩm")

        self.tree.column("id", width=50, anchor="center")
        self.tree.column("danh_muc", width=95, anchor="center")
        self.tree.column("ten", width=400, anchor="w")
        self.tree.column("gia", width=115, anchor="e")
        self.tree.column("score", width=105, anchor="center")
        self.tree.column("link", width=190, anchor="w")

        self.tree.pack(fill="both", expand=True)
        
        self.tree.bind("<<TreeviewSelect>>", self.show_description)
        self.tree.bind("<Double-1>", self.open_product_link)

        # 4. Khung hiển thị chi tiết mô tả sản phẩm bên dưới
        desc_frame = tk.LabelFrame(self.root, text=" Mô tả chi tiết sản phẩm ", font=("Arial", 10, "bold"), padx=10, pady=10)
        desc_frame.pack(fill="x")

        self.desc_text = tk.Text(desc_frame, height=4, font=("Arial", 10), wrap="word", state="disabled", bg="#f8f9fa")
        self.desc_text.pack(fill="x")

        self.current_results = []

    def search_products(self):
        query = self.query_entry.get().strip()
        if not query:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập mô tả sản phẩm bạn đang tìm kiếm!")
            return

        for row in self.tree.get_children():
            self.tree.delete(row)

        pred_cat, self.current_results = self.engine.get_recommendations(query, top_k=5)

        if not self.current_results:
            self.pred_label.config(text="Ngành hàng dự đoán bởi mô hình SVM: Không xác định")
            messagebox.showinfo("Kết quả", "Không tìm thấy sản phẩm nào khớp với mô tả của bạn!")
            return

        self.pred_label.config(text=f"Ngành hàng dự đoán bởi mô hình SVM: {pred_cat.upper()}")

        for item in self.current_results:
            self.tree.insert("", "end", values=(
                item["id"], 
                item["danh_muc"],
                item["ten_san_pham"], 
                item["gia"], 
                item["score"], 
                item["link"]
            ))

    def show_description(self, event):
        selected_item = self.tree.selection()
        if not selected_item:
            return
        index = self.tree.index(selected_item[0])
        mo_ta = self.current_results[index]["mo_ta"]

        self.desc_text.config(state="normal")
        self.desc_text.delete("1.0", tk.END)
        self.desc_text.insert(tk.END, mo_ta)
        self.desc_text.config(state="disabled")

    def open_product_link(self, event):
        selected_item = self.tree.selection()
        if not selected_item:
            return
        index = self.tree.index(selected_item[0])
        link = self.current_results[index]["link"]
        webbrowser.open(link)

if __name__ == "__main__":
    root = tk.Tk()
    app = RecommendationApp(root)
    root.mainloop()