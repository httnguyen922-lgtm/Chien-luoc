"""
========================================================================================
HỆ THỐNG KIỂM ĐỊNH CHIẾN LƯỢC GIAO DỊCH KỸ THUẬT: KẾT HỢP SMA VÀ OBV
VỚI PHÂN BỔ DANH MỤC EQUAL WEIGHT & MPT (MARKOWITZ)
========================================================================================
Tác giả: Chuyên gia Định lượng (Quantitative Trading App Developer)
Nền tảng: Streamlit Web App
Dữ liệu: Thị trường Chứng khoán Việt Nam (HOSE 2020 - 2023)
========================================================================================
"""

import os
import io
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.optimize import minimize

try:
    import ta
except ImportError:
    ta = None

try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

warnings.filterwarnings("ignore")

# ======================================================================================
# CẤU HÌNH TRANG STREAMLIT
# ======================================================================================
st.set_page_config(
    page_title="Kiểm định Chiến lược SMA + OBV & MPT Portfolio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện hiện đại, chuyên nghiệp
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 80%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.375rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 6px 6px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ======================================================================================
# CÁC HÀM XỬ LÝ DỮ LIỆU VÀ TÍNH TOÁN KỸ THUẬT (DATA & INDICATORS)
# ======================================================================================

@st.cache_data(show_spinner=False)
def load_csv_data(file_source):
    """Đọc và chuẩn hóa dữ liệu từ file CSV."""
    if isinstance(file_source, str):
        if not os.path.exists(file_source):
            return None
        df = pd.read_csv(file_source, encoding="utf-8-sig", low_memory=False)
    else:
        df = pd.read_csv(file_source, encoding="utf-8-sig", low_memory=False)

    df.columns = df.columns.str.strip().str.lower()
    required = ["date", "ticker", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "adj_close" in df.columns:
        df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")

    df = df.dropna(subset=["ticker", "close", "volume"])
    return df


def prepare_stock_series(df_full, ticker, use_adj_close=False):
    """Lọc và sắp xếp chuỗi thời gian của 1 mã cổ phiếu."""
    sub = df_full[df_full["ticker"] == ticker.upper()].copy()
    if sub.empty:
        return pd.DataFrame()

    sub = sub.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    close_col = "adj_close" if (use_adj_close and "adj_close" in sub.columns) else "close"

    df_stock = pd.DataFrame({
        "Open": sub["open"].values,
        "High": sub["high"].values,
        "Low": sub["low"].values,
        "Close": sub[close_col].values,
        "Volume": sub["volume"].values
    }, index=pd.DatetimeIndex(sub["date"]))

    df_stock = df_stock.dropna()
    return df_stock


def calc_sma_indicators(close_series, ma_short, ma_long):
    """Tính toán 2 đường SMA ngắn hạn và dài hạn."""
    if ta is not None:
        ma_s = ta.trend.SMAIndicator(close=close_series, window=int(ma_short)).sma_indicator()
        ma_l = ta.trend.SMAIndicator(close=close_series, window=int(ma_long)).sma_indicator()
    else:
        ma_s = close_series.rolling(window=int(ma_short)).mean()
        ma_l = close_series.rolling(window=int(ma_long)).mean()
    return ma_s, ma_l


def calc_obv_indicator(close_series, volume_series, obv_window):
    """Tính toán chỉ báo On-Balance Volume (OBV) và đường trung bình OBV MA."""
    if ta is not None:
        obv = ta.volume.OnBalanceVolumeIndicator(close=close_series, volume=volume_series).on_balance_volume()
    else:
        direction = np.sign(close_series.diff().fillna(0.0))
        obv = (direction * volume_series).cumsum()

    obv_ma = obv.rolling(window=int(obv_window)).mean()
    return obv, obv_ma


def find_position_sma(df, ma_short, ma_long):
    """Tín hiệu giao dịch từ chiến lược SMA Crossover."""
    position = pd.Series(0.0, index=df.index, name="position")
    if int(ma_short) >= int(ma_long):
        return position

    ma_s, ma_l = calc_sma_indicators(df["Close"], ma_short, ma_long)
    buy_signal = (ma_s > ma_l) & (ma_s.shift(1) <= ma_l.shift(1))
    sell_signal = (ma_s < ma_l) & (ma_s.shift(1) >= ma_l.shift(1))

    position.loc[buy_signal] = 1.0
    position.loc[sell_signal] = -1.0
    return position


def find_position_obv(df, obv_window):
    """Tín hiệu giao dịch từ chỉ báo OBV cắt đường trung bình."""
    position = pd.Series(0.0, index=df.index, name="position")
    obv, obv_ma = calc_obv_indicator(df["Close"], df["Volume"], obv_window)

    buy_signal = (obv > obv_ma) & (obv.shift(1) <= obv_ma.shift(1))
    sell_signal = (obv < obv_ma) & (obv.shift(1) >= obv_ma.shift(1))

    position.loc[buy_signal] = 1.0
    position.loc[sell_signal] = -1.0
    return position


def find_position_combined_and(df, ma_short, ma_long, obv_window):
    """Kết hợp SMA và OBV theo logic AND (Đồng thuận cả 2 mới vào/ra lệnh)."""
    sma = find_position_sma(df, ma_short, ma_long)
    obv = find_position_obv(df, obv_window)

    position = pd.Series(0.0, index=df.index, name="position")
    position.loc[(sma == 1.0) & (obv == 1.0)] = 1.0
    position.loc[(sma == -1.0) & (obv == -1.0)] = -1.0
    return position


def find_position_combined_or(df, ma_short, ma_long, obv_window):
    """Kết hợp SMA và OBV theo logic OR (Một trong hai phát tín hiệu)."""
    sma = find_position_sma(df, ma_short, ma_long)
    obv = find_position_obv(df, obv_window)

    position = pd.Series(0.0, index=df.index, name="position")
    buy = (sma == 1.0) | (obv == 1.0)
    sell = (sma == -1.0) | (obv == -1.0)

    conflict = buy & sell
    position.loc[buy & ~conflict] = 1.0
    position.loc[sell & ~conflict] = -1.0
    return position


def events_to_holding(events):
    """Chuyển đổi tín hiệu giao dịch (+1 Buy, -1 Sell) thành trạng thái nắm giữ (1: Giữ, 0: Tiền mặt)."""
    holding = pd.Series(0.0, index=events.index)
    current = 0.0
    for i in range(len(events)):
        signal = events.iloc[i]
        if signal == 1.0:
            current = 1.0
        elif signal == -1.0:
            current = 0.0
        holding.iloc[i] = current
    return holding


def strategy_returns(df, events, commission=0.0):
    """Tính chuỗi tỷ suất sinh lời thực tế (dịch 1 phiên thực thi T+1 và trừ phí giao dịch)."""
    asset_ret = df["Close"].pct_change().fillna(0.0)
    holding = events_to_holding(events)

    # Dịch 1 phiên: Tín hiệu hôm nay chỉ được thực thi tại phiên kế tiếp
    executed_holding = holding.shift(1).fillna(0.0)
    strat_ret = executed_holding * asset_ret

    # Chi phí giao dịch theo biến động vị thế (turnover)
    turnover = executed_holding.diff().abs().fillna(executed_holding.abs())
    strat_ret = strat_ret - turnover * commission

    return strat_ret, executed_holding


def performance_stats(returns, trading_days=252, risk_free_rate=0.0):
    """Tính toán các chỉ số đo lường hiệu quả danh mục và chiến lược."""
    r = returns.dropna()
    if len(r) == 0:
        return {
            "Total Return [%]": np.nan,
            "Annual Return [%]": np.nan,
            "Annual Volatility [%]": np.nan,
            "Sharpe Ratio": np.nan,
            "Max Drawdown [%]": np.nan,
            "Win Rate [%]": np.nan,
            "Profit Factor": np.nan
        }

    equity = (1.0 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)

    years = len(r) / trading_days
    annual_return = (
        float(equity.iloc[-1] ** (1.0 / years) - 1.0)
        if years > 0 and equity.iloc[-1] > 0
        else np.nan
    )

    annual_vol = float(r.std() * np.sqrt(trading_days))

    # Sharpe ratio với lãi suất phi rủi ro
    excess_ret = float(r.mean() * trading_days - risk_free_rate)
    sharpe = (
        excess_ret / annual_vol
        if (annual_vol > 0 and not np.isnan(annual_vol))
        else np.nan
    )

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = float(drawdown.min())

    pos_ret = r[r > 0]
    neg_ret = r[r < 0]
    win_rate = (len(pos_ret) / len(r[r != 0]) * 100.0) if len(r[r != 0]) > 0 else np.nan
    profit_factor = (pos_ret.sum() / abs(neg_ret.sum())) if (len(neg_ret) > 0 and abs(neg_ret.sum()) > 0) else np.nan

    return {
        "Total Return [%]": total_return * 100.0,
        "Annual Return [%]": annual_return * 100.0,
        "Annual Volatility [%]": annual_vol * 100.0,
        "Sharpe Ratio": sharpe,
        "Max Drawdown [%]": max_dd * 100.0,
        "Win Rate [%]": win_rate,
        "Profit Factor": profit_factor
    }


# ======================================================================================
# TỐI ƯU HÓA DANH MỤC MPT (MARKOWITZ) & TỐI ƯU THAM SỐ HYPEROPT
# ======================================================================================

def portfolio_returns(return_matrix, weights):
    """Tính chuỗi tỷ suất lợi nhuận của toàn danh mục dựa trên vector trọng số."""
    return return_matrix.mul(weights, axis=1).sum(axis=1)


def optimize_mpt(train_return_matrix, risk_free_rate=0.0, trading_days=252):
    """Tối ưu hóa danh mục MPT để tối đa hóa Sharpe Ratio (Markowitz Tangency Portfolio)."""
    n = train_return_matrix.shape[1]
    if n <= 1:
        return np.array([1.0])

    mean_daily = train_return_matrix.mean().values
    cov_annual = train_return_matrix.cov().values * trading_days

    def neg_sharpe(weights):
        p_return = float(weights @ mean_daily * trading_days)
        variance = float(weights.T @ cov_annual @ weights)
        p_vol = np.sqrt(max(variance, 1e-8))
        if p_vol <= 0:
            return 1e9
        return -(p_return - risk_free_rate) / p_vol

    x0 = np.repeat(1.0 / n, n)
    bounds = [(0.0, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    result = minimize(
        neg_sharpe,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if result.success:
        return result.x
    return x0


def run_hyperopt_tuning(df_train, max_evals=40, commission=0.0):
    """Tối ưu hóa tham số SMA & OBV bằng Hyperopt TPE trên tập Train."""
    if not HYPEROPT_AVAILABLE:
        return {"ma_short": 50, "ma_long": 200, "obv_window": 20}

    def score_sma(p):
        ma_s = int(p["ma_short"])
        ma_l = int(p["ma_long"])
        if ma_s >= ma_l:
            return 999999.0
        events = find_position_sma(df_train, ma_s, ma_l)
        ret, _ = strategy_returns(df_train, events, commission=commission)
        sharpe = performance_stats(ret)["Sharpe Ratio"]
        return -sharpe if (not pd.isna(sharpe)) else 999999.0

    def score_obv(p):
        obv_w = int(p["obv_window"])
        events = find_position_obv(df_train, obv_w)
        ret, _ = strategy_returns(df_train, events, commission=commission)
        sharpe = performance_stats(ret)["Sharpe Ratio"]
        return -sharpe if (not pd.isna(sharpe)) else 999999.0

    space_sma = {
        "ma_short": hp.quniform("ma_short", 20, 150, 5),
        "ma_long": hp.quniform("ma_long", 160, 400, 5)
    }
    best_sma_raw = fmin(
        fn=score_sma,
        space=space_sma,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=Trials(),
        verbose=False
    )

    space_obv = {
        "obv_window": hp.quniform("obv_window", 5, 100, 5)
    }
    best_obv_raw = fmin(
        fn=score_obv,
        space=space_obv,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=Trials(),
        verbose=False
    )

    return {
        "ma_short": int(best_sma_raw["ma_short"]),
        "ma_long": int(best_sma_raw["ma_long"]),
        "obv_window": int(best_obv_raw["obv_window"])
    }


# ======================================================================================
# GIAO DIỆN THANH BÊN (SIDEBAR)
# ======================================================================================

st.sidebar.markdown("## ⚙️ Cấu Hình Hệ Thống")

# 1. Nguồn Dữ Liệu
st.sidebar.markdown("### 📁 1. Nguồn Dữ Liệu")
data_option = st.sidebar.radio(
    "Chọn nguồn file:",
    ["File mẫu cục bộ (HOSE_2020_2023_in.csv)", "Tải lên file CSV mới"],
    index=0
)

df_raw = None
if data_option == "File mẫu cục bộ (HOSE_2020_2023_in.csv)":
    candidate_paths = [
        "HOSE_2020_2023_in.csv",
        "HOSE_2020_2023_in(1).csv",
        os.path.join(os.path.dirname(__file__), "HOSE_2020_2023_in.csv")
    ]
    found_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            found_path = p
            break
    if found_path:
        df_raw = load_csv_data(found_path)
    else:
        st.sidebar.warning("Không tìm thấy file mẫu `HOSE_2020_2023_in.csv` tại thư mục hiện tại. Vui lòng tải file lên.")
else:
    uploaded = st.sidebar.file_uploader("Tải lên file CSV định dạng OHLCV:", type=["csv"])
    if uploaded is not None:
        df_raw = load_csv_data(uploaded)

if df_raw is None:
    st.error("⚠️ **Chưa có dữ liệu hợp lệ.** Vui lòng kiểm tra lại file `HOSE_2020_2023_in.csv` trong thư mục hoặc tải lên file CSV có các cột: `date, ticker, open, high, low, close, volume`.")
    st.stop()

all_tickers = sorted(df_raw["ticker"].unique().tolist())
min_date = df_raw["date"].min().date()
max_date = df_raw["date"].max().date()

# 2. Chọn Cổ Phiếu
st.sidebar.markdown("### 🎯 2. Lựa Chọn Cổ Phiếu")
default_selected = [t for t in ["ACB", "FPT", "HPG"] if t in all_tickers]
if not default_selected and len(all_tickers) >= 3:
    default_selected = all_tickers[:3]

selected_tickers = st.sidebar.multiselect(
    "Chọn các mã cổ phiếu đưa vào danh mục:",
    options=all_tickers,
    default=default_selected
)

if len(selected_tickers) == 0:
    st.sidebar.error("Vui lòng chọn ít nhất 1 mã cổ phiếu!")
    st.stop()

# 3. Phân Tách Tập Dữ Liệu Train / Test
st.sidebar.markdown("### 📅 3. Phân Kỳ Dữ Liệu (In-Sample / Out-of-Sample)")

# Hàm an toàn để đưa mốc ngày mặc định về trong khoảng [min_date, max_date]
def safe_date_clamp(target_str, fallback):
    try:
        d = pd.to_datetime(target_str).date()
        if d < min_date:
            return min_date
        if d > max_date:
            return max_date
        return d
    except Exception:
        return fallback

default_train_start = safe_date_clamp("2020-01-01", min_date)
default_train_end = safe_date_clamp("2021-12-31", max_date)
default_test_start = safe_date_clamp("2022-01-01", min_date)
default_test_end = safe_date_clamp("2022-12-31", max_date)

col_t1, col_t2 = st.sidebar.columns(2)
with col_t1:
    train_start = st.date_input("Train Bắt đầu", value=default_train_start, min_value=min_date, max_value=max_date)
    test_start = st.date_input("Test Bắt đầu", value=default_test_start, min_value=min_date, max_value=max_date)
with col_t2:
    train_end = st.date_input("Train Kết thúc", value=default_train_end, min_value=min_date, max_value=max_date)
    test_end = st.date_input("Test Kết thúc", value=default_test_end, min_value=min_date, max_value=max_date)

# 4. Tham Số Chiến Lược
st.sidebar.markdown("### 🎛️ 4. Thiết Lập Tham Số Chỉ Báo")
param_mode = st.sidebar.selectbox(
    "Chế độ thiết lập tham số:",
    ["Đồng bộ toàn bộ mã", "Riêng từng mã cổ phiếu"]
)

# Quản lý tham số trong Session State
if "stock_params" not in st.session_state:
    st.session_state.stock_params = {}

for t in selected_tickers:
    if t not in st.session_state.stock_params:
        st.session_state.stock_params[t] = {"ma_short": 50, "ma_long": 200, "obv_window": 20}

if param_mode == "Đồng bộ toàn bộ mã":
    g_ma_short = st.sidebar.slider("SMA Ngắn (Phiên)", min_value=10, max_value=100, value=50, step=5)
    g_ma_long = st.sidebar.slider("SMA Dài (Phiên)", min_value=100, max_value=350, value=200, step=10)
    g_obv_window = st.sidebar.slider("OBV MA Window (Phiên)", min_value=5, max_value=100, value=20, step=5)
    for t in selected_tickers:
        st.session_state.stock_params[t] = {"ma_short": g_ma_short, "ma_long": g_ma_long, "obv_window": g_obv_window}
else:
    t_inspect = st.sidebar.selectbox("Chọn mã để chỉnh tham số:", selected_tickers)
    c_s = st.sidebar.slider(f"{t_inspect}: SMA Ngắn", 10, 100, st.session_state.stock_params[t_inspect]["ma_short"], 5)
    c_l = st.sidebar.slider(f"{t_inspect}: SMA Dài", 100, 350, st.session_state.stock_params[t_inspect]["ma_long"], 10)
    c_o = st.sidebar.slider(f"{t_inspect}: OBV Window", 5, 100, st.session_state.stock_params[t_inspect]["obv_window"], 5)
    st.session_state.stock_params[t_inspect] = {"ma_short": c_s, "ma_long": c_l, "obv_window": c_o}

# 5. Cấu Hình Danh Mục
st.sidebar.markdown("### 💼 5. Cấu Hình Danh Mục & Chi Phí")
portfolio_signal_mode = st.sidebar.selectbox(
    "Logic kết hợp tín hiệu cho Danh mục:",
    ["OR", "AND"],
    help="OR: Mua/bán khi 1 trong 2 chỉ báo kích hoạt. AND: Mua/bán khi cả 2 cùng kích hoạt."
)
commission_pct = st.sidebar.number_input(
    "Phí & Thuế giao dịch (% mỗi chiều):",
    min_value=0.0, max_value=2.0, value=0.15, step=0.05
) / 100.0
risk_free_pct = st.sidebar.number_input(
    "Lãi suất phi rủi ro (% / năm):",
    min_value=0.0, max_value=15.0, value=0.0, step=0.5
) / 100.0
initial_capital = st.sidebar.number_input(
    "Vốn khởi điểm (VNĐ):",
    min_value=100_000, value=1_000_000, step=100_000
)

# ======================================================================================
# CHUẨN BỊ VÀ PHÂN CHIA DỮ LIỆU
# ======================================================================================

stock_data = {}
train_data = {}
test_data = {}

for ticker in selected_tickers:
    df_s = prepare_stock_series(df_raw, ticker)
    if df_s.empty:
        st.warning(f"Mã {ticker} không có đủ dữ liệu. Bỏ qua.")
        continue

    train = df_s.loc[(df_s.index >= pd.to_datetime(train_start)) & (df_s.index <= pd.to_datetime(train_end))].copy()
    test = df_s.loc[(df_s.index >= pd.to_datetime(test_start)) & (df_s.index <= pd.to_datetime(test_end))].copy()

    if train.empty or test.empty:
        st.error(f"Mã {ticker} có tập Train ({len(train)} phiên) hoặc Test ({len(test)} phiên) rỗng. Vui lòng điều chỉnh lại khoảng ngày.")
        st.stop()

    stock_data[ticker] = df_s
    train_data[ticker] = train
    test_data[ticker] = test


# ======================================================================================
# GIAO DIỆN CHÍNH (MAIN DASHBOARD TABS)
# ======================================================================================

st.markdown('<div class="main-header">📈 Quantitative Trading Lab: SMA + OBV & MPT Portfolio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Kiểm định chiến lược giao dịch kết hợp chỉ báo Xu hướng (SMA) và Động lượng Khối lượng (OBV) trên TTCK Việt Nam</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Biểu Đồ & Tín Hiệu",
    "🧪 Backtest Từng Cổ Phiếu",
    "⚖️ Tối Ưu Tham Số (Hyperopt)",
    "💼 Danh Mục: Equal Weight vs MPT",
    "📑 Lý Thuyết & Báo Cáo"
])


# --------------------------------------------------------------------------------------
# TAB 1: BIỂU ĐỒ KỸ THUẬT & TÍN HIỆU GIAO DỊCH
# --------------------------------------------------------------------------------------
with tab1:
    st.subheader("📊 Trực Quan Hóa Tín Hiệu Giao Dịch & Trạng Thái Vị Thế")
    c_ticker, c_period = st.columns([2, 2])
    with c_ticker:
        target_ticker = st.selectbox("Chọn cổ phiếu xem chi tiết:", selected_tickers, key="tab1_ticker")
    with c_period:
        target_period = st.radio("Khung thời gian:", ["Toàn bộ dữ liệu", "Chỉ tập Train", "Chỉ tập Test"], horizontal=True)

    if target_period == "Toàn bộ dữ liệu":
        df_plot = stock_data[target_ticker]
    elif target_period == "Chỉ tập Train":
        df_plot = train_data[target_ticker]
    else:
        df_plot = test_data[target_ticker]

    p_short = st.session_state.stock_params[target_ticker]["ma_short"]
    p_long = st.session_state.stock_params[target_ticker]["ma_long"]
    p_obv_w = st.session_state.stock_params[target_ticker]["obv_window"]

    # Tính toán chỉ báo và tín hiệu
    ma_s, ma_l = calc_sma_indicators(df_plot["Close"], p_short, p_long)
    obv_val, obv_ma = calc_obv_indicator(df_plot["Close"], df_plot["Volume"], p_obv_w)

    evt_sma = find_position_sma(df_plot, p_short, p_long)
    evt_obv = find_position_obv(df_plot, p_obv_w)
    evt_comb_or = find_position_combined_or(df_plot, p_short, p_long, p_obv_w)
    evt_comb_and = find_position_combined_and(df_plot, p_short, p_long, p_obv_w)

    holding_or = events_to_holding(evt_comb_or)
    holding_and = events_to_holding(evt_comb_and)

    # Hiển thị metrics nhanh
    m1, m2, m3, m4, m5 = st.columns(5)
    last_close = df_plot["Close"].iloc[-1]
    prev_close = df_plot["Close"].iloc[-2] if len(df_plot) > 1 else last_close
    pct_day = ((last_close / prev_close) - 1.0) * 100.0

    m1.metric("Giá Hiện Tại", f"{last_close:,.2f}", f"{pct_day:+.2f}%")
    m2.metric("Số Phiên", f"{len(df_plot)} phiên")
    m3.metric("Số Lệnh SMA", f"{(evt_sma != 0).sum()} tín hiệu")
    m4.metric("Số Lệnh OBV", f"{(evt_obv != 0).sum()} tín hiệu")
    m5.metric("Vị Thế OR Hiện Tại", "ĐANG NẮM GIỮ (LONG)" if holding_or.iloc[-1] == 1 else "TIỀN MẶT (CASH)")

    # Vẽ biểu đồ nến + SMA + Tín hiệu + OBV Subplot
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=(
            f"Diễn biến Giá & Tín hiệu SMA ({p_short}/{p_long}) - {target_ticker}",
            f"Chỉ báo On-Balance Volume (OBV) & Đường MA({p_obv_w})",
            "Trạng thái Vị thế Nắm giữ (Holding State: 1 = Cổ phiếu, 0 = Tiền mặt)"
        )
    )

    # Row 1: Candlestick & SMAs
    fig.add_trace(
        go.Candlestick(
            x=df_plot.index,
            open=df_plot["Open"],
            high=df_plot["High"],
            low=df_plot["Low"],
            close=df_plot["Close"],
            name="Nến Giá",
            increasing_line_color="#22C55E",
            decreasing_line_color="#EF4444"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_plot.index, y=ma_s, name=f"SMA {p_short}", line=dict(color="#3B82F6", width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_plot.index, y=ma_l, name=f"SMA {p_long}", line=dict(color="#F97316", width=2)),
        row=1, col=1
    )

    # Đánh dấu tín hiệu Buy/Sell của chiến lược Combined OR
    buy_dates = df_plot.index[evt_comb_or == 1.0]
    sell_dates = df_plot.index[evt_comb_or == -1.0]

    if len(buy_dates) > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_dates,
                y=df_plot.loc[buy_dates, "Low"] * 0.97,
                mode="markers",
                name="Tín hiệu Mua (OR)",
                marker=dict(symbol="triangle-up", size=11, color="#10B981")
            ),
            row=1, col=1
        )
    if len(sell_dates) > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_dates,
                y=df_plot.loc[sell_dates, "High"] * 1.03,
                mode="markers",
                name="Tín hiệu Bán (OR)",
                marker=dict(symbol="triangle-down", size=11, color="#DC2626")
            ),
            row=1, col=1
        )

    # Row 2: OBV & OBV MA
    fig.add_trace(
        go.Scatter(x=df_plot.index, y=obv_val, name="OBV", line=dict(color="#8B5CF6", width=1.5)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_plot.index, y=obv_ma, name=f"OBV MA({p_obv_w})", line=dict(color="#EC4899", width=1.5, dash="dot")),
        row=2, col=1
    )

    # Row 3: Holding State
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=holding_or,
            name="Holding (OR)",
            line=dict(color="#059669", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.15)"
        ),
        row=3, col=1
    )

    fig.update_layout(
        height=820,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------------------
# TAB 2: KIỂM ĐỊNH TỪNG CỔ PHIẾU (SINGLE STOCK BACKTEST)
# --------------------------------------------------------------------------------------
with tab2:
    st.subheader("🧪 Đánh Giá Hiệu Quả Chiến Lược Trên Từng Cổ Phiếu")
    sel_stock = st.selectbox("Chọn mã cổ phiếu kiểm tra:", selected_tickers, key="tab2_stock")

    params = st.session_state.stock_params[sel_stock]
    st.info(f"Đang sử dụng tham số cho **{sel_stock}**: `ma_short={params['ma_short']}`, `ma_long={params['ma_long']}`, `obv_window={params['obv_window']}`, `phí giao dịch={commission_pct*100:.2f}%`")

    def run_all_strats_for_stock(df):
        ms, ml, ow = params["ma_short"], params["ma_long"], params["obv_window"]
        ev_sma = find_position_sma(df, ms, ml)
        ev_obv = find_position_obv(df, ow)
        ev_and = find_position_combined_and(df, ms, ml, ow)
        ev_or = find_position_combined_or(df, ms, ml, ow)

        r_sma, h_sma = strategy_returns(df, ev_sma, commission_pct)
        r_obv, h_obv = strategy_returns(df, ev_obv, commission_pct)
        r_and, h_and = strategy_returns(df, ev_and, commission_pct)
        r_or, h_or = strategy_returns(df, ev_or, commission_pct)

        # Benchmark: Buy & Hold
        r_bnh = df["Close"].pct_change().fillna(0.0)

        returns_dict = {
            "SMA Only": r_sma,
            "OBV Only": r_obv,
            "SMA + OBV (AND)": r_and,
            "SMA + OBV (OR)": r_or,
            "Buy & Hold (Benchmark)": r_bnh
        }
        stats_df = pd.DataFrame({
            k: performance_stats(v, risk_free_rate=risk_free_pct)
            for k, v in returns_dict.items()
        }).T
        return returns_dict, stats_df

    train_rets, train_stats = run_all_strats_for_stock(train_data[sel_stock])
    test_rets, test_stats = run_all_strats_for_stock(test_data[sel_stock])

    col_box1, col_box2 = st.columns(2)
    with col_box1:
        st.markdown(f"#### 📗 Kết quả In-Sample (TRAIN: {train_start} đến {train_end})")
        st.dataframe(train_stats.style.format("{:.2f}").background_gradient(cmap="Blues", subset=["Sharpe Ratio"]), use_container_width=True)
    with col_box2:
        st.markdown(f"#### 📙 Kết quả Out-of-Sample (TEST: {test_start} đến {test_end})")
        st.dataframe(test_stats.style.format("{:.2f}").background_gradient(cmap="Oranges", subset=["Sharpe Ratio"]), use_container_width=True)

    # Đồ thị so sánh Equity Curve (NAV)
    st.markdown("#### 📈 Đồ Thị Đường Tích Lũy Vốn (Equity Curve)")
    eval_choice = st.radio("Chọn tập dữ liệu để vẽ biểu đồ NAV:", ["Tập TEST (Out-of-Sample 2022)", "Tập TRAIN (In-Sample 2020-2021)"], horizontal=True)

    active_rets = test_rets if "TEST" in eval_choice else train_rets

    fig_equity = go.Figure()
    colors = {
        "SMA Only": "#3B82F6",
        "OBV Only": "#8B5CF6",
        "SMA + OBV (AND)": "#EC4899",
        "SMA + OBV (OR)": "#10B981",
        "Buy & Hold (Benchmark)": "#6B7280"
    }

    for strat_name, r_series in active_rets.items():
        nav = initial_capital * (1.0 + r_series).cumprod()
        fig_equity.add_trace(go.Scatter(
            x=nav.index,
            y=nav.values,
            mode="lines",
            name=strat_name,
            line=dict(color=colors.get(strat_name, "#000000"), width=2.5 if "OR" in strat_name or "AND" in strat_name else 1.5)
        ))

    fig_equity.update_layout(
        title=f"Đường Tích Lũy Tài Sản (NAV) - {sel_stock} ({eval_choice})",
        xaxis_title="Thời gian",
        yaxis_title="Giá trị Danh mục (VNĐ)",
        height=480,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_equity, use_container_width=True)

    # Đồ thị Mức sụt giảm cực đại (Underwater Drawdown)
    st.markdown("#### 📉 Đồ Thị Sụt Giảm Giá Trị Danh Mục (Underwater Drawdown %)")
    fig_dd = go.Figure()
    for strat_name, r_series in active_rets.items():
        eq = (1.0 + r_series).cumprod()
        dd = (eq - eq.cummax()) / eq.cummax() * 100.0
        fig_dd.add_trace(go.Scatter(
            x=dd.index,
            y=dd.values,
            mode="lines",
            name=strat_name,
            line=dict(color=colors.get(strat_name, "#000000"), width=1.5)
        ))

    fig_dd.update_layout(
        xaxis_title="Thời gian",
        yaxis_title="Drawdown (%)",
        height=320,
        margin=dict(l=40, r=40, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_dd, use_container_width=True)


# --------------------------------------------------------------------------------------
# TAB 3: TỐI ƯU HÓA THAM SỐ (HYPEROPT)
# --------------------------------------------------------------------------------------
with tab3:
    st.subheader("⚖️ Tối Ưu Hóa Tham Số SMA & OBV Bằng Hyperopt (TPE Algorithm)")
    st.write("""
    Module này sử dụng thuật toán **Tree-structured Parzen Estimator (TPE)** của thư viện `hyperopt` để tìm bộ tham số
    `ma_short`, `ma_long`, và `obv_window` tối ưu hóa **Sharpe Ratio** trên tập **TRAIN** (In-Sample),
    sau đó kiểm tra tính vững (robustness) trên tập **TEST** (Out-of-Sample).
    """)

    c_opt1, c_opt2 = st.columns([1, 2])
    with c_opt1:
        evals_count = st.slider("Số vòng thử nghiệm (Max Evaluations):", 10, 100, 40, 10)
        btn_run_opt = st.button("🚀 Bắt đầu Tối Ưu Hóa Từng Mã", type="primary")

    if btn_run_opt:
        if not HYPEROPT_AVAILABLE:
            st.error("Thư viện `hyperopt` chưa được cài đặt. Vui lòng cài `pip install hyperopt`.")
        else:
            opt_progress = st.progress(0)
            status_text = st.empty()
            new_params = {}

            for idx, ticker in enumerate(selected_tickers):
                status_text.text(f"Đang tối ưu mã {ticker} ({idx+1}/{len(selected_tickers)})...")
                tuned = run_hyperopt_tuning(train_data[ticker], max_evals=evals_count, commission=commission_pct)
                new_params[ticker] = tuned
                opt_progress.progress((idx + 1) / len(selected_tickers))

            status_text.success("🎉 Hoàn tất quá trình tối ưu hóa!")
            st.session_state.optimized_params = new_params

    if "optimized_params" in st.session_state:
        st.markdown("#### 📋 Bảng Tham Số Tối Ưu Hóa Thu Được Từ Tập Train:")
        opt_df = pd.DataFrame(st.session_state.optimized_params).T
        st.dataframe(opt_df, use_container_width=True)

        if st.button("✅ Áp dụng ngay bộ tham số tối ưu này vào toàn bộ ứng dụng"):
            for t, p in st.session_state.optimized_params.items():
                st.session_state.stock_params[t] = p
            st.success("Đã cập nhật tham số thành công! Các tab Backtest và Portfolio sẽ tự động cập nhật.")
            st.rerun()


# --------------------------------------------------------------------------------------
# TAB 4: DANH MỤC ĐẦU TƯ: EQUAL WEIGHT VS MPT (MARKOWITZ)
# --------------------------------------------------------------------------------------
with tab4:
    st.subheader("💼 So Sánh Danh Mục Đầu Tư: Equal Weight vs Modern Portfolio Theory (MPT)")
    st.markdown(f"""
    Xây dựng danh mục gồm **{len(selected_tickers)} cổ phiếu** ({', '.join(selected_tickers)}) theo chiến lược **`SMA + OBV ({portfolio_signal_mode})`**.
    - **Equal Weight (EW):** Phân bổ đều trọng số $w_i = 1 / N$.
    - **MPT (Markowitz Tangency):** Tối ưu hóa trọng số trên tập **TRAIN** để tối đa hóa Sharpe Ratio, sau đó giữ nguyên trọng số kiểm thử trên tập **TEST (2022)**.
    """)

    # 1. Tạo ma trận tỷ suất sinh lời của từng cổ phiếu trong tập Train và Test
    strat_key = f"SMA + OBV {portfolio_signal_mode}"

    train_ret_dict = {}
    test_ret_dict = {}

    for ticker in selected_tickers:
        p = st.session_state.stock_params[ticker]
        if portfolio_signal_mode == "OR":
            ev_tr = find_position_combined_or(train_data[ticker], p["ma_short"], p["ma_long"], p["obv_window"])
            ev_te = find_position_combined_or(test_data[ticker], p["ma_short"], p["ma_long"], p["obv_window"])
        else:
            ev_tr = find_position_combined_and(train_data[ticker], p["ma_short"], p["ma_long"], p["obv_window"])
            ev_te = find_position_combined_and(test_data[ticker], p["ma_short"], p["ma_long"], p["obv_window"])

        r_tr, _ = strategy_returns(train_data[ticker], ev_tr, commission_pct)
        r_te, _ = strategy_returns(test_data[ticker], ev_te, commission_pct)

        train_ret_dict[ticker] = r_tr
        test_ret_dict[ticker] = r_te

    train_ret_matrix = pd.DataFrame(train_ret_dict).dropna()
    test_ret_matrix = pd.DataFrame(test_ret_dict).dropna()

    # 2. Tính trọng số Equal Weight & MPT Weights (từ Train)
    n_assets = len(selected_tickers)
    ew_weights = np.repeat(1.0 / n_assets, n_assets)
    mpt_weights = optimize_mpt(train_ret_matrix, risk_free_rate=risk_free_pct)

    col_w1, col_w2 = st.columns([1, 2])
    with col_w1:
        st.markdown("#### ⚖️ Trọng Số Phân Bổ Danh Mục")
        weight_comparison = pd.DataFrame({
            "Equal Weight": ew_weights,
            "MPT (Markowitz)": mpt_weights
        }, index=selected_tickers)
        st.dataframe(weight_comparison.style.format("{:.2%}"), use_container_width=True)

    with col_w2:
        fig_weight = go.Figure(data=[
            go.Bar(name="Equal Weight (1/N)", x=selected_tickers, y=ew_weights, marker_color="#3B82F6"),
            go.Bar(name="MPT (Max Sharpe)", x=selected_tickers, y=mpt_weights, marker_color="#10B981")
        ])
        fig_weight.update_layout(
            title="So Sánh Tỷ Trọng Phân Bổ Danh Mục",
            yaxis=dict(title="Trọng số", tickformat=".0%"),
            barmode="group",
            height=300,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_weight, use_container_width=True)

    # 3. Tính toán Returns của Danh mục
    ew_train_p_ret = portfolio_returns(train_ret_matrix, ew_weights)
    mpt_train_p_ret = portfolio_returns(train_ret_matrix, mpt_weights)

    ew_test_p_ret = portfolio_returns(test_ret_matrix, ew_weights)
    mpt_test_p_ret = portfolio_returns(test_ret_matrix, mpt_weights)

    port_stats_train = pd.DataFrame({
        "Equal Weight (Train)": performance_stats(ew_train_p_ret, risk_free_rate=risk_free_pct),
        "MPT (Train)": performance_stats(mpt_train_p_ret, risk_free_rate=risk_free_pct)
    }).T

    port_stats_test = pd.DataFrame({
        "Equal Weight (Test)": performance_stats(ew_test_p_ret, risk_free_rate=risk_free_pct),
        "MPT (Test)": performance_stats(mpt_test_p_ret, risk_free_rate=risk_free_pct)
    }).T

    st.markdown("#### 📊 Bảng So Sánh Hiệu Suất Đầu Tư Giữa 2 Phương Pháp Phân Bổ")
    c_tbl1, c_tbl2 = st.columns(2)
    with c_tbl1:
        st.markdown("##### 📗 Tập In-Sample (TRAIN 2020 - 2021)")
        st.dataframe(port_stats_train.style.format("{:.2f}").background_gradient(cmap="Blues", subset=["Sharpe Ratio"]), use_container_width=True)
    with c_tbl2:
        st.markdown("##### 📙 Tập Out-of-Sample (TEST 2022 - QUAN TRỌNG NHẤT)")
        st.dataframe(port_stats_test.style.format("{:.2f}").background_gradient(cmap="Oranges", subset=["Sharpe Ratio"]), use_container_width=True)

    # 4. Đồ thị Đường Tích Lũy Vốn Của Danh Mục Trên Tập TEST
    st.markdown("#### 📈 Diễn Biến NAV Của Danh Mục Trên Tập Ngoại Mẫu TEST (2022)")
    ew_equity_test = initial_capital * (1.0 + ew_test_p_ret).cumprod()
    mpt_equity_test = initial_capital * (1.0 + mpt_test_p_ret).cumprod()

    fig_port_nav = go.Figure()
    fig_port_nav.add_trace(go.Scatter(
        x=ew_equity_test.index,
        y=ew_equity_test.values,
        mode="lines",
        name="Equal Weight (1/N)",
        line=dict(color="#3B82F6", width=2.5)
    ))
    fig_port_nav.add_trace(go.Scatter(
        x=mpt_equity_test.index,
        y=mpt_equity_test.values,
        mode="lines",
        name="MPT (Markowitz Tangency)",
        line=dict(color="#10B981", width=2.5)
    ))

    fig_port_nav.update_layout(
        title=f"Đường Tích Lũy NAV Danh Mục (Tập TEST 2022 - {strat_key})",
        xaxis_title="Thời gian",
        yaxis_title="Giá trị Danh mục (VNĐ)",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_port_nav, use_container_width=True)

    # 5. Phân Tích Đường Biên Hiệu Quả (Markowitz Efficient Frontier Simulation)
    st.markdown("#### 🌐 Mô Phỏng Đường Biên Hiệu Quả Markowitz (Efficient Frontier)")
    with st.expander("Bấm để xem mô phỏng Monte Carlo 1,500 danh mục ngẫu nhiên trên tập Train"):
        np.random.seed(42)
        n_sim = 1500
        sim_vols = np.zeros(n_sim)
        sim_rets = np.zeros(n_sim)
        sim_sharpes = np.zeros(n_sim)

        cov_annual = train_ret_matrix.cov().values * 252
        mean_ret_annual = train_ret_matrix.mean().values * 252

        for i in range(n_sim):
            w = np.random.random(n_assets)
            w = w / np.sum(w)
            p_ret = float(np.sum(w * mean_ret_annual))
            p_vol = float(np.sqrt(w.T @ cov_annual @ w))
            sim_rets[i] = p_ret * 100.0
            sim_vols[i] = p_vol * 100.0
            sim_sharpes[i] = (p_ret - risk_free_pct) / p_vol if p_vol > 0 else 0

        # Điểm MPT và EW
        mpt_p_ret = float(np.sum(mpt_weights * mean_ret_annual)) * 100.0
        mpt_p_vol = float(np.sqrt(mpt_weights.T @ cov_annual @ mpt_weights)) * 100.0

        ew_p_ret = float(np.sum(ew_weights * mean_ret_annual)) * 100.0
        ew_p_vol = float(np.sqrt(ew_weights.T @ cov_annual @ ew_weights)) * 100.0

        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=sim_vols,
            y=sim_rets,
            mode="markers",
            marker=dict(
                size=5,
                color=sim_sharpes,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Sharpe Ratio")
            ),
            name="Danh mục Ngẫu nhiên"
        ))
        fig_ef.add_trace(go.Scatter(
            x=[mpt_p_vol],
            y=[mpt_p_ret],
            mode="markers",
            marker=dict(symbol="star", size=16, color="#EF4444"),
            name="MPT Tangency (Tối ưu Sharpe)"
        ))
        fig_ef.add_trace(go.Scatter(
            x=[ew_p_vol],
            y=[ew_p_ret],
            mode="markers",
            marker=dict(symbol="diamond", size=14, color="#3B82F6"),
            name="Equal Weight (1/N)"
        ))

        fig_ef.update_layout(
            title="Mô phỏng Đường biên Hiệu quả Markowitz (Train In-Sample)",
            xaxis_title="Độ biến động Hàng năm (Volatility %)",
            yaxis_title="Lợi nhuận Kỳ vọng Hàng năm (Annual Return %)",
            height=450,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_ef, use_container_width=True)

    # 6. Ma Trận Tương Quan Lợi Nhuận
    st.markdown("#### 🔗 Ma Trận Tương Quan Chiến Lược Giữa Các Cổ Phiếu (Correlation Matrix)")
    corr_df = train_ret_matrix.corr()
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale="RdBu",
        zmin=-1.0, zmax=1.0,
        text=np.round(corr_df.values, 2),
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    fig_corr.update_layout(height=350, margin=dict(l=40, r=40, t=20, b=40))
    st.plotly_chart(fig_corr, use_container_width=True)


# --------------------------------------------------------------------------------------
# TAB 5: LÝ THUYẾT & KẾT LUẬN NGHIÊN CỨU
# --------------------------------------------------------------------------------------
with tab5:
    st.subheader("📑 Tổng Kết Chiến Lược & Phương Pháp Luận Học Thuật")
    st.markdown("""
    ### 1. Cơ Sở Lý Thuyết Chiến Lược SMA + OBV
    - **Simple Moving Average (SMA):**
      - Tín hiệu Crossover giữa đường trung bình ngắn hạn và dài hạn giúp phát hiện sự chuyển dịch của xu hướng chính.
      - *Ưu điểm:* Lọc bớt nhiễu thị trường ngắn hạn.
      - *Nhược điểm:* Độ trễ (Lagging indicator), dễ gặp tín hiệu giả (Whipsaw) trong thị trường đi ngang (Sideway).
    - **On-Balance Volume (OBV):**
      - Đo lường dòng tiền tích lũy và áp lực khối lượng giao dịch. Giá tăng đi kèm khối lượng lớn xác nhận xu hướng lành mạnh từ dòng tiền tổ chức.
      - *Ưu điểm:* Dẫn dắt giá (Leading indicator), phát hiện phân kỳ dòng tiền.
    - **Sự Kết Hợp (AND vs OR):**
      - **Logic AND:** Yêu cầu cả SMA và OBV cùng đồng thuận phát tín hiệu Mua/Bán. Số lượng giao dịch ít hơn, độ chính xác cao hơn, nhưng có thể bỏ lỡ nhịp chạy sớm.
      - **Logic OR:** Linh hoạt hơn, phản ứng nhanh hơn khi 1 trong 2 chỉ báo kích hoạt.

    ### 2. Phương Pháp Luận Kiểm Định In-Sample & Out-of-Sample
    - **Tập TRAIN (2020 - 2021):** Được dùng để tối ưu hóa tham số chỉ báo (bằng Hyperopt) và ước lượng ma trận hiệp phương sai nhằm giải bài toán phân bổ vốn MPT.
    - **Tập TEST (2022):** Dữ liệu hoàn toàn độc lập ("Out-of-Sample"). Năm 2022 là giai đoạn thị trường giảm mạnh (Bear Market). Đây là bài test khắc nghiệt nhất để đánh giá khả năng quản trị rủi ro và bảo toàn vốn của chiến lược.

    ### 3. Đánh Giá Phân Bổ Danh Mục: Equal Weight vs MPT
    - **Lý thuyết Danh mục Hiện đại (MPT):** Tối đa hóa Sharpe Ratio dựa trên ma trận hiệp phương sai quá khứ. MPT có xu hướng dồn tỷ trọng lớn vào các tài sản có hiệu suất vượt trội trong quá khứ.
    - **Hiệu ứng "Overfitting" của MPT:** Trong thực tế kiểm nghiệm Out-of-Sample (Test 2022), nếu cấu trúc tương quan và lợi nhuận tài sản đảo chiều, MPT có thể sụt giảm mạnh hơn danh mục **Equal Weight (1/N)**. Đây là hiện tượng nổi tiếng trong lý thuyết tài chính định lượng: *Chiến lược phân bổ đều 1/N thường rất khó bị đánh bại ngoài mẫu*.
    """)

    st.markdown("### 📥 Tải Xuất Báo Cáo Kết Quả Kiểm Định")
    # Tạo CSV download
    summary_report = pd.concat([port_stats_train, port_stats_test])
    csv_buffer = io.StringIO()
    summary_report.to_csv(csv_buffer)
    st.download_button(
        label="⬇️ Tải file Báo Cáo Hiệu Suất Danh Mục (.CSV)",
        data=csv_buffer.getvalue(),
        file_name="Portfolio_Performance_Report.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown("<p style='text-align: center; color: #9CA3AF;'>Ứng dụng được thiết kế phục vụ nghiên cứu Định lượng & Quản trị Danh mục Đầu tư | Streamlit App</p>", unsafe_allow_html=True)
