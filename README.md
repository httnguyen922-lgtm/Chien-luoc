# 📈 Quantitative Trading Lab: Chiến Lược Kết Hợp SMA & OBV Với Phân Bổ Danh Mục MPT & Equal Weight

Ứng dụng Web App tương tác xây dựng trên nền tảng **Streamlit** nhằm kiểm định (Backtesting), tối ưu hóa tham số (Hyperopt) và phân bổ danh mục đầu tư (Equal Weight vs Modern Portfolio Theory - MPT) dựa trên chiến lược kết hợp chỉ báo xu hướng **SMA (Simple Moving Average)** và chỉ báo động lượng dòng tiền **OBV (On-Balance Volume)** trên dữ liệu Sở Giao dịch Chứng khoán TP. Hồ Chí Minh (HOSE).

---

## 🌟 Tính Năng Nổi Bật Của Ứng Dụng

1. **Trực quan hóa Tín hiệu Kỹ thuật:**
   - Biểu đồ nến tương tác (Plotly Candlestick) kết hợp 2 đường SMA (Ngắn hạn & Dài hạn).
   - Tự động hiển thị các điểm kích hoạt Mua (▲ Green) và Bán (▼ Red).
   - Biểu đồ chỉ báo khối lượng OBV kèm đường trung bình động OBV MA.
   - Theo dõi trạng thái vị thế (Holding State: Long = 1, Tiền mặt = 0) theo thời gian thực.
2. **Kiểm định Đa Chiến lược (Single Stock Backtest):**
   - Đánh giá và so sánh đồng thời 5 chiến lược:
     - `SMA Only`
     - `OBV Only`
     - `SMA + OBV (AND)`: Vào/ra lệnh khi cả 2 chỉ báo cùng đồng thuận.
     - `SMA + OBV (OR)`: Vào/ra lệnh khi 1 trong 2 chỉ báo phát tín hiệu.
     - `Buy & Hold`: Chiến lược chuẩn đối sánh thị trường (Benchmark).
   - So sánh trực quan trên cả 2 giai đoạn: **In-Sample (Train 2020 - 2021)** và **Out-of-Sample (Test 2022 - Bear Market)**.
   - Thước đo định lượng toàn diện: Tổng lợi nhuận (Total Return), Lợi nhuận hàng năm (CAGR), Độ biến động (Annual Volatility), Tỷ số Sharpe (Sharpe Ratio), Mức sụt giảm tối đa (Max Drawdown), Tỷ lệ thắng (Win Rate).
3. **Tối ưu hóa Tham số Tự động (Bayesian Optimization with Hyperopt):**
   - Ứng dụng thuật toán **Tree-structured Parzen Estimator (TPE)** để tự động quét và tìm bộ tham số tối ưu (`ma_short`, `ma_long`, `obv_window`) nhằm cực đại hóa Sharpe Ratio trên tập Train.
   - Cơ chế áp dụng tức thì bộ tham số tối ưu vào toàn bộ hệ thống backtest.
4. **Phân bổ Danh mục Đầu tư (Portfolio Construction):**
   - So sánh giữa 2 trường phái:
     - **Equal Weight (1/N):** Phân bổ tỷ trọng đều cho các cổ phiếu.
     - **Markowitz Modern Portfolio Theory (MPT):** Tối ưu hóa trọng số trên tập Train giải bài toán Quadratic Programming (SLSQP) tìm Tangency Portfolio có Sharpe Ratio cao nhất.
   - Kiểm định tính bền vững ngoại mẫu trên tập **TEST (2022)** để kiểm chứng hiện tượng Overfitting của mô hình MPT.
   - Mô phỏng Monte Carlo **Đường biên Hiệu quả (Efficient Frontier)** với 1,500 danh mục ngẫu nhiên.
   - Ma trận tương quan (Correlation Matrix Heatmap) giữa các cổ phiếu được chọn.
5. **Xuất Báo Cáo:**
   - Cho phép tải bảng tổng kết kết quả kiểm định dạng file CSV chỉ với 1 cú click.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
├── app.py                             # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt                   # Danh sách các thư viện Python cần thiết
├── README.md                          # Tài liệu hướng dẫn sử dụng và triển khai
├── HOSE_2020_2023_in.csv              # Dữ liệu giá lịch sử HOSE (2020 - 2023)
└── HOSE_SMA_OBV_EqualWeight_MPT.ipynb # Jupyter Notebook nghiên cứu gốc
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Ứng Dụng Trên Máy Cục Bộ (Local)

### Bước 1: Chuẩn bị môi trường Python
Yêu cầu phiên bản **Python 3.9 - 3.11**. Mở Terminal (Command Prompt hoặc PowerShell trên Windows) và chuyển đến thư mục dự án:

```bash
cd "c:\Users\Lenovo\Desktop\Chien luoc"
```

### Bước 2: Tạo và kích hoạt môi trường ảo (Virtual Environment)
```bash
# Tạo virtual environment có tên venv
python -m venv venv

# Kích hoạt trên Windows:
.\venv\Scripts\activate

# Hoặc trên macOS / Linux:
# source venv/bin/activate
```

### Bước 3: Cài đặt các thư viện cần thiết
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 4: Khởi chạy Web App
```bash
streamlit run app.py
```
Sau khi chạy lệnh, trình duyệt web sẽ tự động mở địa chỉ: `http://localhost:8501`.

---

## ☁️ Hướng Dẫn Đưa Lên GitHub & Deploy Trực Tiếp Lên Streamlit Cloud

### Bước 1: Khởi tạo Git Repository và Đẩy Code Lên GitHub

1. Tạo một tài khoản trên [GitHub.com](https://github.com/) (nếu chưa có).
2. Tạo một Repository mới trên GitHub (ví dụ đặt tên: `sma-obv-mpt-streamlit`), chọn chế độ **Public**.
3. Tại terminal máy tính của bạn, chạy các lệnh sau:

```bash
# Khởi tạo git
git init

# Thêm tất cả các file vào git
git add app.py requirements.txt README.md HOSE_2020_2023_in.csv

# Tạo commit đầu tiên
git commit -m "Initial commit: SMA OBV Backtesting & MPT Portfolio Streamlit App"

# Đổi nhánh chính thành main
git branch -M main

# Liên kết với repo trên GitHub (thay URL bên dưới bằng URL repo GitHub của bạn)
git remote add origin https://github.com/<tai-khoan-github-cua-ban>/sma-obv-mpt-streamlit.git

# Đẩy code lên GitHub
git push -u origin main
```

> **Lưu ý về file dữ liệu:** File `HOSE_2020_2023_in.csv` có dung lượng ~5.7 MB (nằm trong giới hạn < 25 MB của GitHub) nên hoàn toàn có thể commit và push bình thường.

### Bước 2: Deploy Miễn Phí Trên Streamlit Community Cloud

1. Truy cập [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấn nút **"New app"** (hoặc **"Create app"**).
3. Điền thông tin ứng dụng:
   - **Repository:** Chọn repository bạn vừa tạo (ví dụ: `<tai-khoan-github-cua-ban>/sma-obv-mpt-streamlit`).
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Nhấn **"Deploy!"**.
5. Streamlit Cloud sẽ tự động đọc `requirements.txt`, cài đặt các thư viện và cung cấp cho bạn một đường link công khai dạng:
   `https://<ten-app>.streamlit.app` để truy cập và chia sẻ mọi lúc mọi nơi.

---

## 📐 Cơ Sở Lý Thuyết & Công Thức Tính Toán

### 1. Đường Trung Bình Động (Simple Moving Average - SMA)
$$SMA_t(k) = \frac{1}{k} \sum_{i=0}^{k-1} P_{t-i}$$
- **Tín hiệu Mua:** Khi $SMA_{short} > SMA_{long}$ và $SMA_{short, t-1} \le SMA_{long, t-1}$ (Golden Cross).
- **Tín hiệu Bán:** Khi $SMA_{short} < SMA_{long}$ và $SMA_{short, t-1} \ge SMA_{long, t-1}$ (Death Cross).

### 2. Chỉ Báo Khối Lượng Cân Bằng (On-Balance Volume - OBV)
$$OBV_t = OBV_{t-1} + \begin{cases} +Volume_t & \text{nếu } P_t > P_{t-1} \\ 0 & \text{nếu } P_t = P_{t-1} \\ -Volume_t & \text{nếu } P_t < P_{t-1} \end{cases}$$
- Sử dụng đường trung bình trượt của OBV ($OBV\_MA$) với chu kỳ `obv_window` để tạo tín hiệu giao cắt.

### 3. Logic Kết Hợp Tín Hiệu
- **Logic AND:** $Buy_{AND} = (Buy_{SMA} == 1) \land (Buy_{OBV} == 1)$
- **Logic OR:** $Buy_{OR} = (Buy_{SMA} == 1) \lor (Buy_{OBV} == 1)$ (nếu xuất hiện mâu thuẫn vừa Buy vừa Sell thì đưa vị thế về 0).

### 4. Cơ Chế Khớp Lệnh Thực Tế (Execution & Turnover)
- Để tránh thiên lệch nhìn trước (Look-ahead bias), lệnh được giả định khớp tại phiên kế tiếp:
  $$\text{Executed Holding}_t = \text{Holding}_{t-1}$$
- Lợi nhuận chiến lược sau chi phí giao dịch ($c$):
  $$R_{strat, t} = \text{Executed Holding}_t \cdot R_{asset, t} - |\text{Executed Holding}_t - \text{Executed Holding}_{t-1}| \cdot c$$

### 5. Tối Ưu Hóa Danh Mục Hiện Đại (Modern Portfolio Theory - MPT)
Cực đại hóa Tỷ số Sharpe của danh mục:
$$\max_{w} \frac{w^T \mu - R_f}{\sqrt{w^T \Sigma w}}$$
$$\text{Thỏa mãn: } \sum_{i=1}^N w_i = 1 \quad \text{và} \quad 0 \le w_i \le 1 \quad (\forall i)$$
Trong đó:
- $\mu$: Vector tỷ suất sinh lời kỳ vọng hàng năm của từng tài sản.
- $\Sigma$: Ma trận hiệp phương sai hàng năm giữa các tài sản.
- $R_f$: Lãi suất phi rủi ro.

---

## 📊 Khuyến Nghị Nghiên Cứu Định Lượng

- **Hiện tượng Overfitting trong MPT:** Trên tập Train (2020 - 2021, giai đoạn thị trường Uptrend), danh mục MPT thường có Sharpe Ratio cao hơn Equal Weight do được tối ưu trực tiếp trên mẫu. Tuy nhiên trên tập Test (2022, Downtrend), danh mục **Equal Weight (1/N)** thường thể hiện tính kiên định và giảm thiểu rủi ro tập trung tốt hơn.
- **Chi phí giao dịch:** Với các chiến lược sử dụng logic **OR**, số lượng lệnh sinh ra nhiều hơn logic **AND**, nhà đầu tư cần chú ý đến tỷ lệ bào mòn vốn bởi thuế và phí giao dịch thực tế.

---

## 📜 Giấy Phép & Tác Quyền
Dự án được xây dựng phục vụ mục đích nghiên cứu học thuật và kiểm định chiến lược định lượng. Mã nguồn mở theo giấy phép MIT License.
