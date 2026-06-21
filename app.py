import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import random

# =====================================================================
# 1. 頁面設定與精美樣式 (Page Config & Custom CSS)
# =====================================================================
st.set_page_config(
    page_title="0050 實時交易模擬挑戰 | OptiTime-0050",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入 CSS 提升 UI 視覺質感 (玻璃擬態、漸層與現代字型)
st.markdown("""
<style>
    /* 引入 Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Noto+Sans+TC:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', 'Noto Sans TC', sans-serif;
    }
    
    /* 漸層標題與副標題 */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #10B981, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    
    .subtitle {
        font-size: 1.25rem;
        color: #4B5563;
        margin-bottom: 25px;
        font-weight: 600;
    }
    
    /* 績效指標卡片 */
    .metric-card {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(229, 231, 235, 1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .dark .metric-card {
        background: rgba(31, 41, 55, 0.7);
        border: 1px solid rgba(75, 85, 99, 0.4);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* 對比文案大卡片 */
    .comparison-hero {
        background: linear-gradient(135deg, #f0fdf4, #eff6ff);
        border: 1px solid #dcfce7;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 25px;
        text-align: center;
    }
    .dark .comparison-hero {
        background: linear-gradient(135deg, #064e3b, #1e3a8a);
        border: 1px solid #065f46;
    }
    
    .comparison-text {
        font-size: 1.3rem;
        font-weight: 600;
        color: #111827;
    }
    .dark .comparison-text {
        color: #f3f4f6;
    }
    
    .comparison-highlight {
        font-size: 1.7rem;
        font-weight: 700;
        color: #10B981;
    }
    
    /* 調整大按鈕的樣式以方便點擊 */
    .stButton>button {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        padding: 15px 10px !important;
        border-radius: 10px !important;
        min-height: 80px !important;
        white-space: pre-line !important;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# 2. 數據獲取模組 (Data Acquisition with Cache & Exception Handling)
# =====================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_data(ticker="0050.TW", start="2015-01-01", end="2026-12-31"):
    """
    從 yfinance 下載個股資料，並處理除權息 (強制使用 Adj Close)。
    """
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            return None
        
        # 處理 MultiIndex 欄位
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
            
        # 檢查 Adj Close 是否存在
        if 'Adj Close' not in df.columns:
            if 'Close' in df.columns:
                df['Adj Close'] = df['Close']
            else:
                return None
                
        # 保留需要的欄位
        df = df[['Adj Close']].dropna()
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        print(f"Error downloading data from yfinance: {e}")
        return None


# =====================================================================
# 3. 核心演算法模組 (God-View Backtester Core Engine)
# =====================================================================
def run_god_view_backtest(df, n_days):
    """
    執行上帝視角回測 (僅限現股做多)。
    """
    df = df.copy()
    L = len(df)
    
    # 初始化資產與持倉狀態記錄
    god_view_asset = np.ones(L)
    
    # 記錄買賣點的座標 (Date, Price) 用於繪圖
    buy_signals = []   # 格式: (Date, Price, Type)
    sell_signals = []  # 格式: (Date, Price, Type)
    
    current_asset_value = 1.0
    
    # 將資料切割為非重疊的區間
    num_blocks = int(np.ceil(L / n_days))
    
    for k in range(num_blocks):
        start_idx = k * n_days
        end_idx = min((k + 1) * n_days, L)
        
        if start_idx >= L:
            break
            
        sub_df = df.iloc[start_idx:end_idx]
        
        # 尋找區間內極值之相對索引
        min_idx_rel = sub_df['Adj Close'].argmin()
        max_idx_rel = sub_df['Adj Close'].argmax()
        
        # 轉換為全域絕對索引
        min_idx = sub_df.index[min_idx_rel]
        max_idx = sub_df.index[max_idx_rel]
        
        p_min = df.loc[min_idx, 'Adj Close']
        p_max = df.loc[max_idx, 'Adj Close']
        
        # 區間的起始資產價值
        asset_start = current_asset_value
        
        if min_idx == max_idx:
            # 極大極小在同一天 (通常是區間只有 1 天，或價格完全沒變)
            for i in range(start_idx, end_idx):
                god_view_asset[i] = asset_start
            
        elif min_idx < max_idx:
            # 做多交易 (買低賣高): 買點在賣點之前
            # 1. 買點之前: 持有現金
            for i in range(start_idx, min_idx):
                god_view_asset[i] = asset_start
            # 2. 買點到賣點期間: 持有部位，資產隨價格變動
            for i in range(min_idx, max_idx + 1):
                god_view_asset[i] = asset_start * (df.loc[i, 'Adj Close'] / p_min)
            # 3. 賣點之後到區間結束: 持有現金
            for i in range(max_idx + 1, end_idx):
                god_view_asset[i] = asset_start * (p_max / p_min)
                
            # 更新當前資產價值
            current_asset_value = asset_start * (p_max / p_min)
            
            # 記錄訊號
            buy_signals.append((df.loc[min_idx, 'Date'], p_min, 'Long Buy'))
            sell_signals.append((df.loc[max_idx, 'Date'], p_max, 'Long Sell'))
            
        else:
            # max_idx < min_idx: 賣點在買點之前
            # 僅限做多模式下，若無法先買後賣，則此區間空手 (全程持有現金以避開下跌段)
            for i in range(start_idx, end_idx):
                god_view_asset[i] = asset_start
                
    df['God_View_Asset'] = god_view_asset
    
    # 計算 Buy & Hold (長期持有) 資產走勢
    p0 = df.loc[0, 'Adj Close']
    df['BH_Asset'] = df['Adj Close'] / p0
    
    # 建立訊號 DataFrame
    df_buy = pd.DataFrame(buy_signals, columns=['Date', 'Price', 'Type']) if buy_signals else pd.DataFrame(columns=['Date', 'Price', 'Type'])
    df_sell = pd.DataFrame(sell_signals, columns=['Date', 'Price', 'Type']) if sell_signals else pd.DataFrame(columns=['Date', 'Price', 'Type'])
    
    return df, df_buy, df_sell


def find_optimal_trade(df):
    """
    在給定 DataFrame 中尋找單筆最完美的做多交易（買在最低、賣在最高，且買在賣之前）。
    """
    prices = df['Adj Close'].values
    dates = df['Date'].values
    
    if len(prices) == 0:
        return None, None, None, None, 0.0
        
    min_price_so_far = prices[0]
    min_date_so_far = dates[0]
    best_ret = 1.0
    
    best_buy_date = dates[0]
    best_buy_price = prices[0]
    best_sell_date = dates[0]
    best_sell_price = prices[0]
    
    for i in range(len(prices)):
        if prices[i] < min_price_so_far:
            min_price_so_far = prices[i]
            min_date_so_far = dates[i]
            
        ret = prices[i] / min_price_so_far
        if ret > best_ret:
            best_ret = ret
            best_buy_date = min_date_so_far
            best_buy_price = min_price_so_far
            best_sell_date = dates[i]
            best_sell_price = prices[i]
            
    return best_buy_date, best_buy_price, best_sell_date, best_sell_price, (best_ret - 1.0) * 100


def calculate_metrics(df):
    """
    計算回測指標。
    """
    # 總報酬率
    gv_total_ret = (df['God_View_Asset'].iloc[-1] - 1.0) * 100
    bh_total_ret = (df['BH_Asset'].iloc[-1] - 1.0) * 100
    alpha = gv_total_ret - bh_total_ret
    
    # 計算年化報酬率 (CAGR)
    days = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
    years = max(days / 365.25, 0.01)
    
    gv_cagr = (df['God_View_Asset'].iloc[-1]) ** (1 / years) - 1
    bh_cagr = (df['BH_Asset'].iloc[-1]) ** (1 / years) - 1
    
    # 最大回撤 (MDD)
    def compute_mdd(series):
        cum_max = series.cummax()
        drawdown = (series - cum_max) / cum_max
        return drawdown.min() * 100
        
    gv_mdd = compute_mdd(df['God_View_Asset'])
    bh_mdd = compute_mdd(df['BH_Asset'])
    
    return {
        'gv_total_ret': gv_total_ret,
        'bh_total_ret': bh_total_ret,
        'alpha': alpha,
        'gv_cagr': gv_cagr * 100,
        'bh_cagr': bh_cagr * 100,
        'gv_mdd': gv_mdd,
        'bh_mdd': bh_mdd,
        'years': years
    }


def select_game_periods(df, n_setups=5, length=60, n_days=20):
    """
    在整個回測範圍內隨機挑選 n 個包含「最佳買賣點」（由 n_days 上帝視角區間決定）且互不重疊的交易區間。
    """
    L = len(df)
    
    # 1. 尋找所有上帝視角區間中的獲利交易（最低點在最高點之前，即可以買低賣高）
    num_blocks = int(np.ceil(L / n_days))
    valid_trades = []
    
    for k in range(num_blocks):
        start_idx = k * n_days
        end_idx = min((k + 1) * n_days, L)
        if start_idx >= L:
            break
        sub_df = df.iloc[start_idx:end_idx]
        if len(sub_df) < 2:
            continue
            
        min_idx_rel = sub_df['Adj Close'].argmin()
        max_idx_rel = sub_df['Adj Close'].argmax()
        min_idx = sub_df.index[min_idx_rel]
        max_idx = sub_df.index[max_idx_rel]
        
        # 只有在最低點在最高點之前（即可以買低賣高）時才是有效交易
        if min_idx < max_idx:
            valid_trades.append((min_idx, max_idx))
            
    # 2. 隨機打亂這些交易，嘗試為其建立互不重疊的 60 天區間
    random.shuffle(valid_trades)
    
    indices = []
    for buy_idx, sell_idx in valid_trades:
        if len(indices) == n_setups:
            break
            
        # 計算此交易能完全容納在 60 天區間內的起始索引範圍
        min_start = max(0, sell_idx - (length - 1))
        max_start = min(buy_idx, L - length)
        
        if min_start <= max_start:
            # 隨機決定買點出現在第 12 到 25 天之間，確保在挑戰初始進度天數（10天）之後
            offset = random.randint(12, 25)
            start = max(min_start, min(buy_idx - offset, max_start))
            
            # 檢查是否與已選區間重疊
            overlap = False
            for existing_start in indices:
                if abs(start - existing_start) < length:
                    overlap = True
                    break
            if not overlap:
                indices.append(start)
                
    # 3. 如果符合條件的交易區間不足，則用純隨機且不重疊的方式補齊
    if len(indices) < n_setups:
        # 嘗試 1000 次尋找不重疊的起點
        for _ in range(1000):
            if len(indices) == n_setups:
                break
            start = random.randint(0, L - length)
            overlap = False
            for existing_start in indices:
                if abs(start - existing_start) < length:
                    overlap = True
                    break
            if not overlap:
                indices.append(start)
                
    # 4. 如果依然不足，則以均勻間隔填充
    if len(indices) < n_setups:
        indices = [i * (L - length) // n_setups for i in range(n_setups)]
        
    indices.sort()
    
    setups = []
    for start in indices:
        setups.append({
            'start_idx': start,
            'end_idx': start + length - 1,
            'length': length
        })
    return setups


def plot_active_setup_chart(df_full, setup, current_day_offset, trades=[], holding_position=False, buy_day_offset=None, buy_price=None):
    """
    繪製當前關卡已揭示的股價線圖（只畫到當前進度天數，並標示已完成的交易與進行中的交易，支援跨回合交易）。
    """
    prices = df_full['Adj Close'].values
    dates = df_full['Date'].values
    
    start_idx = setup['start_idx']
    curr_idx = start_idx + current_day_offset
    
    # 擷取可見的數據
    visible_df = df_full.iloc[start_idx : curr_idx + 1]
    
    fig = go.Figure()
    
    # 可見的股價折線
    fig.add_trace(go.Scatter(
        x=visible_df['Date'],
        y=visible_df['Adj Close'],
        mode='lines+markers' if len(visible_df) < 20 else 'lines',
        name='0050.TW 股價',
        line=dict(color='#3B82F6', width=3),
        hovertemplate='日期: %{x}<br>價格: NT$ %{y:.2f}<extra></extra>'
    ))
    
    # 1. 繪製已完成的交易 (past trades in this round)
    for i, t in enumerate(trades):
        t_buy_offset = t['buy_day_offset']
        t_sell_idx = start_idx + t['sell_day_offset']
        
        if t_buy_offset >= 0:
            t_buy_idx = start_idx + t_buy_offset
            # 買入標記
            fig.add_trace(go.Scatter(
                x=[dates[t_buy_idx]],
                y=[prices[t_buy_idx]],
                mode='markers',
                marker=dict(color='#10B981', size=10, symbol='triangle-up'),
                hovertemplate=f'第 {i+1} 次買入價: NT$ %{{y:.2f}}<extra></extra>',
                showlegend=False
            ))
            trade_dates = dates[t_buy_idx : t_sell_idx + 1]
            trade_prices = prices[t_buy_idx : t_sell_idx + 1]
        else:
            # 跨回合交易：本關卡開局即持有
            trade_dates = dates[start_idx : t_sell_idx + 1]
            trade_prices = prices[start_idx : t_sell_idx + 1]

        # 賣出標記
        fig.add_trace(go.Scatter(
            x=[dates[t_sell_idx]],
            y=[prices[t_sell_idx]],
            mode='markers',
            marker=dict(color='#EF4444', size=10, symbol='triangle-down'),
            hovertemplate=f'第 {i+1} 次賣出價: NT$ %{{y:.2f}}<extra></extra>',
            showlegend=False
        ))
        # 持有期間連線
        fig.add_trace(go.Scatter(
            x=trade_dates,
            y=trade_prices,
            mode='lines',
            line=dict(color='#10B981', width=2, dash='dot'),
            hoverinfo='skip',
            showlegend=False
        ))
        
    # 2. 如果當前有持倉，繪製進行中的買入點與持股成本線
    if holding_position and buy_price is not None and buy_price > 0:
        if buy_day_offset is not None and buy_day_offset >= 0:
            buy_global_idx = start_idx + buy_day_offset
            fig.add_trace(go.Scatter(
                x=[dates[buy_global_idx]],
                y=[prices[buy_global_idx]],
                mode='markers',
                name='您的買入點',
                marker=dict(color='#10B981', size=14, symbol='triangle-up-dot'),
                hovertemplate='買入價: NT$ %{y:.2f}<extra></extra>'
            ))
            
            # 繪製橫向的成本虛線，從買入日延伸到當前進度日
            visible_dates_since_buy = visible_df['Date'].iloc[buy_day_offset:]
            fig.add_trace(go.Scatter(
                x=visible_dates_since_buy,
                y=[buy_price] * len(visible_dates_since_buy),
                mode='lines',
                name='買入成本線',
                line=dict(color='#10B981', width=1.5, dash='dash'),
                hoverinfo='skip'
            ))
        else:
            # 跨回合持倉：開局即持有，繪製整段成本虛線
            fig.add_trace(go.Scatter(
                x=visible_df['Date'],
                y=[buy_price] * len(visible_df),
                mode='lines',
                name='跨回合買入成本線',
                line=dict(color='#10B981', width=1.5, dash='dash'),
                hoverinfo='skip'
            ))
        
    fig.update_layout(
        height=380,
        xaxis_title="時間進度 (天)",
        yaxis_title="股價 (元)",
        margin=dict(l=40, r=40, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_settled_setup_chart(df_full, setup, trades=[]):
    """
    關卡結束後，展示完整的關卡區間股價與玩家的進出場位置（支援多次交易與跨回合交易）。
    """
    df_setup = df_full.iloc[setup['start_idx'] : setup['end_idx'] + 1]
    prices = df_full['Adj Close'].values
    dates = df_full['Date'].values
    start_idx = setup['start_idx']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_setup['Date'],
        y=df_setup['Adj Close'],
        mode='lines',
        name='0050.TW 完整股價',
        line=dict(color='#9CA3AF', width=2),
        hovertemplate='日期: %{x}<br>價格: NT$ %{y:.2f}<extra></extra>'
    ))
    
    # 繪製每一次交易
    for i, t in enumerate(trades):
        buy_offset = t['buy_day_offset']
        sell_offset = t['sell_day_offset']
        
        sell_global_idx = start_idx + sell_offset
        
        if buy_offset >= 0:
            buy_global_idx = start_idx + buy_offset
            # 買入標記
            fig.add_trace(go.Scatter(
                x=[dates[buy_global_idx]],
                y=[prices[buy_global_idx]],
                mode='markers',
                name=f'買入點 #{i+1}',
                marker=dict(color='#10B981', size=12, symbol='triangle-up-dot'),
                hovertemplate=f'買入價 #{i+1}: NT$ %{{y:.2f}}<extra></extra>',
                showlegend=False
            ))
            # 持有期間高亮
            trade_dates = dates[buy_global_idx : sell_global_idx + 1]
            trade_prices = prices[buy_global_idx : sell_global_idx + 1]
        else:
            # 跨回合交易：本關卡開局即持有
            trade_dates = dates[start_idx : sell_global_idx + 1]
            trade_prices = prices[start_idx : sell_global_idx + 1]
            
        # 賣出標記
        fig.add_trace(go.Scatter(
            x=[dates[sell_global_idx]],
            y=[prices[sell_global_idx]],
            mode='markers',
            name=f'賣出點 #{i+1}',
            marker=dict(color='#EF4444', size=12, symbol='triangle-down-dot'),
            hovertemplate=f'賣出價 #{i+1}: NT$ %{{y:.2f}}<extra></extra>',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=trade_dates,
            y=trade_prices,
            mode='lines',
            name='持有期間' if i == 0 else None,
            line=dict(color='#3B82F6', width=3.5),
            hoverinfo='skip',
            showlegend=True if i == 0 else False
        ))
        
    fig.update_layout(
        height=380,
        xaxis_title="日期",
        yaxis_title="股價 (元)",
        margin=dict(l=40, r=40, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


# =====================================================================
# 5. Streamlit 介面渲染 (UI Render)
# =====================================================================
def main():
    # 標題
    st.markdown('<div class="main-title">0050 實時交易模擬挑戰</div>', unsafe_allow_html=True)
    
    # Sidebar 控制面板
    st.sidebar.header("模擬參數設定")
    
    # 時間範圍設定
    today = datetime.date.today()
    default_start = datetime.date(2015, 1, 1)
    
    start_date = st.sidebar.date_input(
        "回測起始日期",
        value=default_start,
        min_value=datetime.date(2003, 6, 25), # 0050 掛牌日
        max_value=today
    )
    
    end_date = st.sidebar.date_input(
        "回測結束日期",
        value=today,
        min_value=datetime.date(2003, 6, 25),
        max_value=today
    )
    
    if start_date >= end_date:
        st.sidebar.error("錯誤：起始日期必須早於結束日期。")
        return
        
    # N 天滑動視窗設定 (回測與分析用)
    n_days = st.sidebar.slider(
        "上帝視角觀察區間 (N 天)",
        min_value=1,
        max_value=90,
        value=20,
        step=1,
        help="上帝視角演算法會將歷史劃分為每 N 天一個區間，並在區間內尋找最低價（買點）與最高價（賣點）。"
    )
    
    # 讓玩家設定挑戰回合數 (預設 5，遊戲進行中禁用)
    n_rounds = st.sidebar.number_input(
        "挑戰回合數",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        disabled=st.session_state.get('game_active', False),
        help="設定您想要挑戰的獨立交易時段（回合）數量。遊戲進行中無法修改此設定。"
    )
    
    # 讀取數據
    with st.spinner("正在從 Yahoo Finance 獲取 0050.TW 歷史數據..."):
        df_raw = load_stock_data("0050.TW", start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        
    if df_raw is None or len(df_raw) < 10:
        st.error("無法獲取股價數據。請檢查網路連線，或嘗試調整日期範圍。")
        if st.button("重新載入數據"):
            st.rerun()
        return

    # 執行回測
    df_result, df_buy, df_sell = run_god_view_backtest(df_raw, n_days)
    metrics = calculate_metrics(df_result)
    
    # 決定當前要顯示/執行的總回合數
    if st.session_state.get('game_active', False):
        total_rounds = len(st.session_state.setups)
    else:
        total_rounds = n_rounds

    # =====================================================================
    # 1. 核心區塊：實時買賣操作挑戰
    # =====================================================================
    st.markdown(f"## {total_rounds} 回合實時交易決策挑戰")
    
    # 初始化遊戲狀態
    if 'game_active' not in st.session_state:
        st.session_state.game_active = False
        st.session_state.setups = []
        st.session_state.current_setup_idx = 0
        st.session_state.current_day_offset = 10  # 預設前 10 天為歷史對比
        st.session_state.holding_position = False
        st.session_state.buy_price = 0.0
        st.session_state.buy_day_offset = -1
        st.session_state.buy_global_idx = -1
        st.session_state.buy_round_idx = -1
        st.session_state.cash = 1000000.0
        st.session_state.shares = 0.0
        st.session_state.total_fees = 0.0
        st.session_state.setup_reports = []
        st.session_state.setup_state = "playing"  # "playing" 或 "settled"
        st.session_state.round_start_cash = 1000000.0
        st.session_state.current_round_trades = []

    L = len(df_result)
    
    if not st.session_state.game_active:
        st.markdown("<h3 style='color: #2563EB;'>挑戰模式規則說明</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        為了考驗您是否能做到「低買高賣」，本挑戰將在選定的歷史區間中，隨機選取 **{total_rounds} 個互相獨立的時段（每個時段 60 天）**。
        
        * 您將攜帶 **100 萬元的初始資金** 依序進入每個時段。
        * **買賣決策由您全權決定**：在每個時段的 60 天中，您可以決定在何時點選 **【買入】**，並在買入後決定何時選擇 **【賣出】**。
        * **支援多次交易**：只要您手中有足夠的現金或股票，您可以在 60 天內自由交易多次。
        * **必須猜對兩次**：不論交易幾次，每一筆交易您都必須同時精準猜對「買點」與「賣點」才能成功獲利。
        * **時段結束強制結算**：如果您在第 60 天結束時仍持有股票，系統會將股票以最後一天的股價自動賣出結算。
        * **交易摩擦成本**：買入需支付 **0.1425%** 手續費；賣出需支付 **0.1425%** 手續費 ＋ **0.3%** 證券交易稅（共 **0.4425%**）。
        """)
        
        if st.button(f"開始 {total_rounds} 回合擇時交易挑戰", type="primary", use_container_width=True):
            setups = select_game_periods(df_result, n_setups=total_rounds, length=60, n_days=n_days)
            st.session_state.game_active = True
            st.session_state.setups = setups
            st.session_state.current_setup_idx = 0
            st.session_state.current_day_offset = 10
            st.session_state.holding_position = False
            st.session_state.buy_price = 0.0
            st.session_state.buy_day_offset = -1
            st.session_state.buy_global_idx = -1
            st.session_state.buy_round_idx = -1
            st.session_state.cash = 1000000.0
            st.session_state.shares = 0.0
            st.session_state.total_fees = 0.0
            st.session_state.setup_reports = []
            st.session_state.setup_state = "playing"
            st.session_state.round_start_cash = 1000000.0
            st.session_state.current_round_trades = []
            st.rerun()
    else:
        setup_idx = st.session_state.current_setup_idx
        total_setups = len(st.session_state.setups)
        
        if setup_idx < total_setups:
            setup = st.session_state.setups[setup_idx]
            prices = df_result['Adj Close'].values
            dates = df_result['Date'].values
            
            # 當前關卡的全局索引
            curr_global_idx = setup['start_idx'] + st.session_state.current_day_offset
            p_now = prices[curr_global_idx]
            date_now_str = pd.to_datetime(dates[curr_global_idx]).strftime('%Y-%m-%d')
            
            # 計算當前資產總值 (現金 + 股票市值)
            portfolio_value = st.session_state.cash + st.session_state.shares * p_now
            if st.session_state.shares > 0.0001:
                unrealized_return = (p_now / st.session_state.buy_price - 1) * 100
                pos_label = f"持股中 (均價: NT$ {st.session_state.buy_price:.2f}, 目前損益: {unrealized_return:+.2f}%)"
            else:
                pos_label = "未持倉 (持有現金)"
                
            st.markdown(f"### 交易挑戰：第 {setup_idx + 1} / {total_setups} 回合")
            st.progress((setup_idx) / total_setups)
            
            # 當前回合即時資產狀況
            st.markdown(f"""
            <div style="background-color: rgba(59, 130, 246, 0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(59, 130, 246, 0.2); margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-around; text-align: center; flex-wrap: wrap;">
                    <div style="margin: 5px;">
                        <div style="font-size: 0.95rem; color: #4B5563;">模擬日期進度</div>
                        <div style="font-size: 1.6rem; font-weight: 750; color: #1E3A8A;">第 {st.session_state.current_day_offset + 1} / {setup['length']} 天 ({date_now_str})</div>
                    </div>
                    <div style="margin: 5px;">
                        <div style="font-size: 0.95rem; color: #4B5563;">當日收盤價</div>
                        <div style="font-size: 1.6rem; font-weight: 750; color: #3B82F6;">NT$ {p_now:.2f} 元</div>
                    </div>
                    <div style="margin: 5px;">
                        <div style="font-size: 0.95rem; color: #4B5563;">持有部位狀態</div>
                        <div style="font-size: 1.4rem; font-weight: 750; color: {'#10B981' if st.session_state.shares > 0.0001 else '#EF4444'};">{pos_label}</div>
                    </div>
                    <div style="margin: 5px;">
                        <div style="font-size: 0.95rem; color: #2563EB; font-weight: 600;">我的總資產價值</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #2563EB;">TWD {portfolio_value:,.0f} 元</div>
                        <div style="font-size: 0.85rem; color: #4B5563; font-weight: 500;">(現金: {st.session_state.cash:,.0f} | 股票: {st.session_state.shares * p_now:,.0f})</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.setup_state == "playing":
                # 1. 挑戰中：畫出歷史至今的價格走勢
                fig_game = plot_active_setup_chart(
                    df_result, 
                    setup, 
                    st.session_state.current_day_offset,
                    trades=st.session_state.current_round_trades,
                    holding_position=st.session_state.holding_position,
                    buy_day_offset=st.session_state.buy_day_offset if st.session_state.buy_day_offset >= 0 else None,
                    buy_price=st.session_state.buy_price if st.session_state.buy_price > 0 else None
                )
                st.plotly_chart(fig_game, width='stretch')
                
                # 操作按鈕區
                st.markdown("""
                <div style="background-color: #F3F4F6; padding: 12px; border-radius: 8px; margin-bottom: 15px; font-weight: 600; font-size: 1.1rem; color: #374151;">
                    點擊按鈕進行買賣，或點選右側的前進天數以觀察後續股價：
                </div>
                """, unsafe_allow_html=True)
                
                col_buy_panel, col_sell_panel, col_step_panel = st.columns([3, 3, 2])
                
                is_buy_disabled = st.session_state.cash < 100.0 or st.session_state.current_day_offset >= setup['length'] - 1
                is_sell_disabled = st.session_state.shares <= 0.0001
                
                with col_buy_panel:
                    st.markdown("<div style='text-align: center; font-weight: 700; color: #10B981; margin-bottom: 5px; font-size: 0.95rem;'>買入操作 (剩餘現金)</div>", unsafe_allow_html=True)
                    b_col1, b_col2, b_col3 = st.columns(3)
                    with b_col1:
                        if st.button("買 1/4", use_container_width=True, key=f"play_buy_25_{setup_idx}", disabled=is_buy_disabled):
                            fraction = 0.25
                            cash_to_use = st.session_state.cash * fraction
                            fee = cash_to_use * 0.001425
                            net_buy_cash = cash_to_use - fee
                            shares_bought = net_buy_cash / p_now
                            
                            old_shares = st.session_state.shares
                            old_price = st.session_state.buy_price
                            new_shares = old_shares + shares_bought
                            
                            st.session_state.buy_price = (old_shares * old_price + shares_bought * p_now) / new_shares
                            
                            if old_shares == 0:
                                st.session_state.buy_day_offset = st.session_state.current_day_offset
                                st.session_state.buy_global_idx = curr_global_idx
                                st.session_state.buy_round_idx = setup_idx
                                
                            st.session_state.shares = new_shares
                            st.session_state.cash -= cash_to_use
                            st.session_state.total_fees += fee
                            st.session_state.holding_position = True
                            st.rerun()
                            
                    with b_col2:
                        if st.button("買 1/2", use_container_width=True, key=f"play_buy_50_{setup_idx}", disabled=is_buy_disabled):
                            fraction = 0.50
                            cash_to_use = st.session_state.cash * fraction
                            fee = cash_to_use * 0.001425
                            net_buy_cash = cash_to_use - fee
                            shares_bought = net_buy_cash / p_now
                            
                            old_shares = st.session_state.shares
                            old_price = st.session_state.buy_price
                            new_shares = old_shares + shares_bought
                            
                            st.session_state.buy_price = (old_shares * old_price + shares_bought * p_now) / new_shares
                            
                            if old_shares == 0:
                                st.session_state.buy_day_offset = st.session_state.current_day_offset
                                st.session_state.buy_global_idx = curr_global_idx
                                st.session_state.buy_round_idx = setup_idx
                                
                            st.session_state.shares = new_shares
                            st.session_state.cash -= cash_to_use
                            st.session_state.total_fees += fee
                            st.session_state.holding_position = True
                            st.rerun()
                            
                    with b_col3:
                        if st.button("買全部", use_container_width=True, key=f"play_buy_100_{setup_idx}", disabled=is_buy_disabled):
                            fraction = 1.0
                            cash_to_use = st.session_state.cash * fraction
                            fee = cash_to_use * 0.001425
                            net_buy_cash = cash_to_use - fee
                            shares_bought = net_buy_cash / p_now
                            
                            old_shares = st.session_state.shares
                            old_price = st.session_state.buy_price
                            new_shares = old_shares + shares_bought
                            
                            st.session_state.buy_price = (old_shares * old_price + shares_bought * p_now) / new_shares
                            
                            if old_shares == 0:
                                st.session_state.buy_day_offset = st.session_state.current_day_offset
                                st.session_state.buy_global_idx = curr_global_idx
                                st.session_state.buy_round_idx = setup_idx
                                
                            st.session_state.shares = new_shares
                            st.session_state.cash = 0.0
                            st.session_state.total_fees += fee
                            st.session_state.holding_position = True
                            st.rerun()
                            
                with col_sell_panel:
                    st.markdown("<div style='text-align: center; font-weight: 700; color: #EF4444; margin-bottom: 5px; font-size: 0.95rem;'>賣出操作 (持股部位)</div>", unsafe_allow_html=True)
                    s_col1, s_col2, s_col3 = st.columns(3)
                    with s_col1:
                        if st.button("賣 1/4", use_container_width=True, key=f"play_sell_25_{setup_idx}", disabled=is_sell_disabled):
                            fraction = 0.25
                            shares_to_sell = st.session_state.shares * fraction
                            raw_val = shares_to_sell * p_now
                            fee = raw_val * 0.001425
                            tax = raw_val * 0.003
                            net_cash = raw_val - (fee + tax)
                            
                            st.session_state.cash += net_cash
                            st.session_state.shares -= shares_to_sell
                            st.session_state.total_fees += (fee + tax)
                            
                            buy_cost_total = st.session_state.buy_price * 1.001425
                            sell_net_total = p_now * (1 - 0.004425)
                            trade_return = (sell_net_total / buy_cost_total - 1) * 100
                            
                            is_cross_round = (st.session_state.buy_round_idx < setup_idx)
                            st.session_state.current_round_trades.append({
                                'buy_day_offset': st.session_state.buy_day_offset if not is_cross_round else -1,
                                'buy_date': dates[st.session_state.buy_global_idx],
                                'buy_price': st.session_state.buy_price,
                                'sell_day_offset': st.session_state.current_day_offset,
                                'sell_date': dates[curr_global_idx],
                                'sell_price': p_now,
                                'trade_return': trade_return,
                                'action': '賣出 25% (跨回合)' if is_cross_round else '賣出 25%'
                            })
                            
                            if st.session_state.shares <= 0.0001:
                                st.session_state.shares = 0.0
                                st.session_state.holding_position = False
                                st.session_state.buy_price = 0.0
                                st.session_state.buy_day_offset = -1
                                st.session_state.buy_global_idx = -1
                                st.session_state.buy_round_idx = -1
                            st.rerun()
                            
                    with s_col2:
                        if st.button("賣 1/2", use_container_width=True, key=f"play_sell_50_{setup_idx}", disabled=is_sell_disabled):
                            fraction = 0.50
                            shares_to_sell = st.session_state.shares * fraction
                            raw_val = shares_to_sell * p_now
                            fee = raw_val * 0.001425
                            tax = raw_val * 0.003
                            net_cash = raw_val - (fee + tax)
                            
                            st.session_state.cash += net_cash
                            st.session_state.shares -= shares_to_sell
                            st.session_state.total_fees += (fee + tax)
                            
                            buy_cost_total = st.session_state.buy_price * 1.001425
                            sell_net_total = p_now * (1 - 0.004425)
                            trade_return = (sell_net_total / buy_cost_total - 1) * 100
                            
                            is_cross_round = (st.session_state.buy_round_idx < setup_idx)
                            st.session_state.current_round_trades.append({
                                'buy_day_offset': st.session_state.buy_day_offset if not is_cross_round else -1,
                                'buy_date': dates[st.session_state.buy_global_idx],
                                'buy_price': st.session_state.buy_price,
                                'sell_day_offset': st.session_state.current_day_offset,
                                'sell_date': dates[curr_global_idx],
                                'sell_price': p_now,
                                'trade_return': trade_return,
                                'action': '賣出 50% (跨回合)' if is_cross_round else '賣出 50%'
                            })
                            
                            if st.session_state.shares <= 0.0001:
                                st.session_state.shares = 0.0
                                st.session_state.holding_position = False
                                st.session_state.buy_price = 0.0
                                st.session_state.buy_day_offset = -1
                                st.session_state.buy_global_idx = -1
                                st.session_state.buy_round_idx = -1
                            st.rerun()
                            
                    with s_col3:
                        if st.button("賣全部", use_container_width=True, key=f"play_sell_100_{setup_idx}", disabled=is_sell_disabled):
                            shares_to_sell = st.session_state.shares
                            raw_val = shares_to_sell * p_now
                            fee = raw_val * 0.001425
                            tax = raw_val * 0.003
                            net_cash = raw_val - (fee + tax)
                            
                            st.session_state.cash += net_cash
                            st.session_state.shares = 0.0
                            st.session_state.total_fees += (fee + tax)
                            
                            buy_cost_total = st.session_state.buy_price * 1.001425
                            sell_net_total = p_now * (1 - 0.004425)
                            trade_return = (sell_net_total / buy_cost_total - 1) * 100
                            
                            is_cross_round = (st.session_state.buy_round_idx < setup_idx)
                            st.session_state.current_round_trades.append({
                                'buy_day_offset': st.session_state.buy_day_offset if not is_cross_round else -1,
                                'buy_date': dates[st.session_state.buy_global_idx],
                                'buy_price': st.session_state.buy_price,
                                'sell_day_offset': st.session_state.current_day_offset,
                                'sell_date': dates[curr_global_idx],
                                'sell_price': p_now,
                                'trade_return': trade_return,
                                'action': '賣出全部 (跨回合)' if is_cross_round else '賣出全部'
                            })
                            
                            st.session_state.shares = 0.0
                            st.session_state.holding_position = False
                            st.session_state.buy_price = 0.0
                            st.session_state.buy_day_offset = -1
                            st.session_state.buy_global_idx = -1
                            st.session_state.buy_round_idx = -1
                            st.rerun()
                            
                with col_step_panel:
                    st.markdown("<div style='text-align: center; font-weight: 700; color: #3B82F6; margin-bottom: 5px; font-size: 0.95rem;'>時間控制</div>", unsafe_allow_html=True)
                    t_row1_col1, t_row1_col2 = st.columns(2)
                    t_row2_col1, t_row2_col2 = st.columns(2)
                    
                    is_forward_disabled = st.session_state.current_day_offset >= setup['length'] - 1
                    
                    with t_row1_col1:
                        if not is_forward_disabled:
                            if st.button("前進1天", use_container_width=True, key=f"play_step1_{setup_idx}"):
                                st.session_state.current_day_offset += 1
                                st.rerun()
                        else:
                            if setup_idx < total_setups - 1:
                                if st.button("結束回合", use_container_width=True, key=f"play_settle_{setup_idx}"):
                                    # 計算當前資產總值 (不強迫賣出，跨回合攜帶)
                                    portfolio_val = st.session_state.cash
                                    if st.session_state.holding_position:
                                        portfolio_val += st.session_state.shares * p_now
                                    
                                    # 計算此回合累積實際報酬率
                                    round_return = (portfolio_val / st.session_state.round_start_cash - 1) * 100
                                    
                                    setup_prices = prices[setup['start_idx'] : setup['end_idx'] + 1]
                                    bh_buy_c = setup_prices[0] * 1.001425
                                    bh_sell_n = setup_prices[-1] * (1 - 0.004425)
                                    bh_ret = (bh_sell_n / bh_buy_c - 1) * 100
                                    
                                    # 上帝視角完美單次交易報酬率
                                    _, gv_buy_p, _, gv_sell_p, _ = find_optimal_trade(df_result.iloc[setup['start_idx'] : setup['end_idx'] + 1])
                                    gv_buy_c = gv_buy_p * 1.001425
                                    gv_sell_n = gv_sell_p * (1 - 0.004425)
                                    gv_ret = max(0.0, (gv_sell_n / gv_buy_c - 1) * 100)
                                    
                                    st.session_state.setup_reports.append({
                                        'setup_idx': setup_idx,
                                        'round_start_cash': st.session_state.round_start_cash,
                                        'round_end_cash': portfolio_val,
                                        'trade_return': round_return,
                                        'bh_return': bh_ret,
                                        'gv_return': gv_ret,
                                        'trades': list(st.session_state.current_round_trades),
                                        'action': 'Carried Over' if st.session_state.holding_position else 'Cash Only'
                                    })
                                    st.session_state.setup_state = "settled"
                                    st.rerun()
                            else:
                                # 最後一關強制結算賣出
                                if st.button("結算挑戰", use_container_width=True, key=f"play_settle_{setup_idx}"):
                                    if st.session_state.holding_position:
                                        st.session_state.holding_position = False
                                        raw_val = st.session_state.shares * p_now
                                        fee = raw_val * 0.001425
                                        tax = raw_val * 0.003
                                        st.session_state.cash = raw_val - (fee + tax)
                                        st.session_state.shares = 0.0
                                        st.session_state.total_fees += (fee + tax)
                                        
                                        buy_cost_total = st.session_state.buy_price * 1.001425
                                        sell_net_total = p_now * (1 - 0.004425)
                                        trade_return = (sell_net_total / buy_cost_total - 1) * 100
                                        
                                        is_cross_round = (st.session_state.buy_round_idx < setup_idx)
                                        st.session_state.current_round_trades.append({
                                            'buy_day_offset': st.session_state.buy_day_offset if not is_cross_round else -1,
                                            'buy_date': dates[st.session_state.buy_global_idx],
                                            'buy_price': st.session_state.buy_price,
                                            'sell_day_offset': st.session_state.current_day_offset,
                                            'sell_date': dates[curr_global_idx],
                                            'sell_price': p_now,
                                            'trade_return': trade_return,
                                            'action': 'Forced Settle (Cross-Round)' if is_cross_round else 'Forced Settle'
                                        })
                                        st.session_state.buy_price = 0.0
                                        st.session_state.buy_day_offset = -1
                                        st.session_state.buy_global_idx = -1
                                        st.session_state.buy_round_idx = -1
                                        
                                    # 計算此回合累積實際報酬率
                                    round_return = (st.session_state.cash / st.session_state.round_start_cash - 1) * 100
                                    
                                    setup_prices = prices[setup['start_idx'] : setup['end_idx'] + 1]
                                    bh_buy_c = setup_prices[0] * 1.001425
                                    bh_sell_n = setup_prices[-1] * (1 - 0.004425)
                                    bh_ret = (bh_sell_n / bh_buy_c - 1) * 100
                                    
                                    # 上帝視角完美單次交易報酬率
                                    _, gv_buy_p, _, gv_sell_p, _ = find_optimal_trade(df_result.iloc[setup['start_idx'] : setup['end_idx'] + 1])
                                    gv_buy_c = gv_buy_p * 1.001425
                                    gv_sell_n = gv_sell_p * (1 - 0.004425)
                                    gv_ret = max(0.0, (gv_sell_n / gv_buy_c - 1) * 100)
                                    
                                    st.session_state.setup_reports.append({
                                        'setup_idx': setup_idx,
                                        'round_start_cash': st.session_state.round_start_cash,
                                        'round_end_cash': st.session_state.cash,
                                        'trade_return': round_return,
                                        'bh_return': bh_ret,
                                        'gv_return': gv_ret,
                                        'trades': list(st.session_state.current_round_trades),
                                        'action': 'Final Settle'
                                    })
                                    st.session_state.setup_state = "settled"
                                    st.rerun()
                                    
                    with t_row1_col2:
                        if st.button("前進5天", use_container_width=True, key=f"play_step5_{setup_idx}", disabled=is_forward_disabled):
                            st.session_state.current_day_offset = min(setup['length'] - 1, st.session_state.current_day_offset + 5)
                            st.rerun()
                            
                    with t_row2_col1:
                        if st.button("前進10天", use_container_width=True, key=f"play_step10_{setup_idx}", disabled=is_forward_disabled):
                            st.session_state.current_day_offset = min(setup['length'] - 1, st.session_state.current_day_offset + 10)
                            st.rerun()
                            
                    with t_row2_col2:
                        if st.button("前進20天", use_container_width=True, key=f"play_step20_{setup_idx}", disabled=is_forward_disabled):
                            st.session_state.current_day_offset = min(setup['length'] - 1, st.session_state.current_day_offset + 20)
                            st.rerun()
                        
                        
                        
            elif st.session_state.setup_state == "settled":
                # 2. 結算中：展示完整時段的線圖與成果
                report = st.session_state.setup_reports[-1]
                bh_ret = report['bh_return']
                gv_ret = report['gv_return']
                round_ret = report['trade_return']
                
                # 畫完整歷史圖
                fig_game = plot_settled_setup_chart(
                    df_result, 
                    setup, 
                    trades=report['trades']
                )
                st.plotly_chart(fig_game, width='stretch')
                
                st.markdown("### 回合結算結果")
                
                # 顯示資訊
                if len(report['trades']) == 0:
                    st.warning("您在此區間未進行任何交易，回合累積報酬率為 0.00%。")
                else:
                    st.markdown("#### 回合交易明細")
                    for i, t in enumerate(report['trades']):
                        action_label = "正常交易" if t['action'] == 'Traded' else "到期強制結算"
                        color_style = "#10B981" if t['trade_return'] >= 0 else "#EF4444"
                        st.markdown(f"""
                        <div style="border: 1px solid {color_style}; padding: 12px; border-radius: 8px; background-color: {color_style}05; margin-bottom: 12px;">
                            <span style="font-weight: 700; color: {color_style};">第 {i+1} 筆交易 ({action_label})</span><br>
                            買入日期：<strong>{pd.to_datetime(t['buy_date']).strftime('%Y-%m-%d')}</strong> (價格: NT$ {t['buy_price']:.2f})<br>
                            賣出日期：<strong>{pd.to_datetime(t['sell_date']).strftime('%Y-%m-%d')}</strong> (價格: NT$ {t['sell_price']:.2f})<br>
                            淨報酬率（扣除摩擦成本）：<strong style="color:{color_style};">{t['trade_return']:+.2f}%</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="background-color: rgba(59, 130, 246, 0.05); padding: 15px; border-radius: 8px; border: 1px solid rgba(59, 130, 246, 0.2); margin-top: 15px;">
                        本回合累積實際報酬率（複利計）：<strong style="font-size: 1.25rem; color: {'#10B981' if round_ret >= 0 else '#EF4444'}">{round_ret:+.2f}%</strong>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 策略對比面板
                st.write("")
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">長期持有 (Buy & Hold) 報酬</div>
                        <div style="font-size: 1.8rem; font-weight:700; color:#4B5563;">{bh_ret:+.2f}%</div>
                        <div style="font-size: 0.85rem; color:#6B7280;">第一天買入，持股至最後一天</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_c2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">上帝視角 (God-View) 完美單次交易報酬</div>
                        <div style="font-size: 1.8rem; font-weight:700; color:#10B981;">{gv_ret:+.2f}%</div>
                        <div style="font-size: 0.85rem; color:#6B7280;">最低點買入，最高點賣出</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.write("")
                next_btn_label = "進入下一回合" if setup_idx < total_setups - 1 else "查看挑戰總結報告"
                if st.button(next_btn_label, type="primary", use_container_width=True, key=f"btn_next_{setup_idx}"):
                    # 前進到下一個區間
                    st.session_state.current_setup_idx += 1
                    if st.session_state.current_setup_idx < total_setups:
                        next_setup = st.session_state.setups[st.session_state.current_setup_idx]
                        next_start_price = prices[next_setup['start_idx'] + 10]
                        # 進入新回合的起始總資產價值（現金 ＋ 股票按新回合第一天價格估值）
                        portfolio_val_start = st.session_state.cash
                        if st.session_state.holding_position:
                            portfolio_val_start += st.session_state.shares * next_start_price
                            # 買入日不在當前關卡，設為 -1 避開買入點圖示，但保留成本線與 buy_price / buy_global_idx / buy_round_idx
                            st.session_state.buy_day_offset = -1
                        st.session_state.round_start_cash = portfolio_val_start
                    
                    st.session_state.current_day_offset = 10
                    st.session_state.current_round_trades = []
                    st.session_state.setup_state = "playing"
                    st.rerun()
                    
        else:
            # 挑戰全部回合完成，顯示最終統計報告
            st.markdown("### 挑戰結束！總結報告")
            
            user_final_asset = st.session_state.cash
            user_overall_return = (user_final_asset / 1000000.0 - 1.0) * 100
            
            # 計算累積 Buy & Hold 報酬
            bh_asset = 1000000.0
            for rep in st.session_state.setup_reports:
                bh_asset *= (1 + rep['bh_return'] / 100)
            bh_overall_return = (bh_asset / 1000000.0 - 1.0) * 100
            
            # 計算累積 God-View 報酬
            gv_asset = 1000000.0
            for rep in st.session_state.setup_reports:
                gv_asset *= (1 + rep['gv_return'] / 100)
            gv_overall_return = (gv_asset / 1000000.0 - 1.0) * 100
            
            st.markdown(f"""
            <div style="background-color: rgba(59, 130, 246, 0.05); padding: 25px; border-radius: 12px; margin-bottom: 25px; border: 2px solid rgba(59, 130, 246, 0.2);">
                <h4 style="color: #2563EB; text-align: center; margin-bottom: 15px; font-size:1.5rem;">{total_rounds} 回合累積表現對比 (初始資金 100 萬)</h4>
                <div style="display: flex; justify-content: space-around; text-align: center; flex-wrap: wrap;">
                    <div style="margin: 10px; min-width: 220px;">
                        <div style="font-size: 1.15rem; color: #4B5563; font-weight: 600; margin-bottom:5px;">您的挑戰最終資產</div>
                        <div style="font-size: 2.3rem; font-weight: 850; color: #2563EB;">TWD {user_final_asset:,.0f} 元</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: {'#10B981' if user_overall_return >= 0 else '#EF4444'};">總報酬率: {user_overall_return:+.2f}%</div>
                    </div>
                    <div style="margin: 10px; min-width: 220px; border-left: 2px solid #E5E7EB; padding-left: 15px;">
                        <div style="font-size: 1.15rem; color: #4B5563; font-weight: 600; margin-bottom:5px;">長期持有 ({total_rounds}回合累積)</div>
                        <div style="font-size: 2.3rem; font-weight: 850; color: #4B5563;">TWD {bh_asset:,.0f} 元</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: {'#10B981' if bh_overall_return >= 0 else '#EF4444'};">總報酬率: {bh_overall_return:+.2f}%</div>
                    </div>
                    <div style="margin: 10px; min-width: 220px; border-left: 2px solid #E5E7EB; padding-left: 15px;">
                        <div style="font-size: 1.15rem; color: #4B5563; font-weight: 600; margin-bottom:5px;">上帝視角 ({total_rounds}回合完美擇時)</div>
                        <div style="font-size: 2.3rem; font-weight: 850; color: #10B981;">TWD {gv_asset:,.0f} 元</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: {'#10B981' if gv_overall_return >= 0 else '#EF4444'};">總報酬率: {gv_overall_return:+.2f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 圖表比較
            fig_bar_game = go.Figure()
            fig_bar_game.add_trace(go.Bar(
                x=["您的累積交易", "長期持有累積", "上帝視角累積"],
                y=[user_overall_return, bh_overall_return, gv_overall_return],
                marker_color=['#2563EB', '#4B5563', '#10B981'],
                text=[f"{user_overall_return:+.2f}%", f"{bh_overall_return:+.2f}%", f"{gv_overall_return:+.2f}%"],
                textposition='auto',
                width=0.4
            ))
            fig_bar_game.update_layout(
                yaxis_title="累積報酬率 (%)",
                height=350,
                margin=dict(l=40, r=40, t=10, b=40)
            )
            st.plotly_chart(fig_bar_game, width='stretch')
            
            # 明細表
            st.subheader(f" {total_rounds} 回合交易細節摘要")
            rows = []
            for idx, rep in enumerate(st.session_state.setup_reports):
                dec_str = f"交易 {len(rep['trades'])} 次" if len(rep['trades']) > 0 else "未交易"
                rows.append({
                    "回合": f"第 {idx + 1} 回合",
                    "您的決策": dec_str,
                    "您的報酬率": f"{rep['trade_return']:+.2f}%",
                    "長期持有報酬": f"{rep['bh_return']:+.2f}%",
                    "完美擇時報酬": f"{rep['gv_return']:+.2f}%"
                })
            st.table(pd.DataFrame(rows).set_index("回合"))
            
            st.markdown("""
            #### 交易挑戰的重要啟示：
            * **「猜對兩次」的致命難度**：要透過頻繁買賣賺錢，您必須同時精準預測「什麼時候是最低點（買點）」和「什麼時候是最高點（賣點）」。只要其中一次猜錯，或者行情不如預期，您的回報會大幅折損。
            * **交易手續費與稅金**：您每次進行買賣，都需要付給政府與券商約 0.58% 的摩擦成本。隨着交易次數增加，這些隱形成本會迅速蠶食您的本金（您本次累計支付的交易摩擦成本即顯著影響了最終損益）。
            * **為什麼長期持有（存股）是贏家？**：如上面的累積績效對比所示，即使市場有波動，**什麼都不做、單純持有的「存股」策略**，由於沒有頻繁進出產生的手續費，且完整參與了市場的上漲，最終往往能擊敗大多數試圖預測高低點的交易者。
            """)
            
            st.write("")
            if st.button("重新開始新挑戰", use_container_width=True):
                st.session_state.game_active = False
                st.rerun()

    # =====================================================================
    # 2. 輔助分析與上帝視角回測 (以折疊面板呈現，避免干擾主要挑戰)
    # =====================================================================
    st.write("---")
    with st.expander("查看整個時間範圍的「上帝視角」對比與歷史統計資料"):
        # 強調對比感文案
        if metrics['alpha'] >= 0:
            comparison_html = f"""比存股多賺 <span class="comparison-highlight" style="color: #10B981;">{metrics['alpha']:,.2f}%</span>"""
        else:
            comparison_html = f"""比存股少賺 <span class="comparison-highlight" style="color: #EF4444;">{abs(metrics['alpha']):,.2f}%</span>"""
            
        compare_copy = f"""
        <div class="comparison-hero">
            <span class="comparison-text">
                如果過去 <span style="color:#3B82F6; font-weight:700;">{n_days}</span> 天你能精準掌握最高與最低點，
                你將 {comparison_html} 的超額報酬！
            </span>
        </div>
        """
        st.markdown(compare_copy, unsafe_allow_html=True)
        
        # 指標面板
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">上帝視角總報酬</div>
                <div class="metric-value" style="color: #10B981;">{metrics['gv_total_ret']:,.2f}%</div>
                <div class="metric-delta" style="color: #10B981;">年化: {metrics['gv_cagr']:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">長期持有總報酬</div>
                <div class="metric-value" style="color: #6B7280;">{metrics['bh_total_ret']:,.2f}%</div>
                <div class="metric-delta" style="color: #6B7280;">年化: {metrics['bh_cagr']:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
            <div class="metric-card" style="border: 2px solid rgba(59, 130, 246, 0.4);">
                <div class="metric-label">超額報酬 (Alpha)</div>
                <div class="metric-value" style="color: #3B82F6;">{metrics['alpha']:,.2f}%</div>
                <div class="metric-delta" style="color: #3B82F6;">領先存股</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">上帝視角最大回撤</div>
                <div class="metric-value" style="color: #EF4444;">{metrics['gv_mdd']:.2f}%</div>
                <div class="metric-delta" style="color: #6B7280;">極值交易風險</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">長期持有最大回撤</div>
                <div class="metric-value" style="color: #EF4444;">{metrics['bh_mdd']:.2f}%</div>
                <div class="metric-delta" style="color: #6B7280;">存股最大跌幅</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        
        # 標籤頁
        sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs(["上帝視角對比圖表", "績效統計與數據報表", "錯過最佳交易日分析", "演算法運作邏輯解析"])
        
        with sub_tab1:
            st.subheader("價格走勢與資產淨值對比")
            scale_type = st.radio("資產走勢座標軸刻度", ["線性刻度 (Linear)", "對數刻度 (Log)"], horizontal=True, key="sub_scale_type")
            is_log = scale_type == "對數刻度 (Log)"
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=("0050 歷史股價與上帝視角交易點位", "策略累積淨值走勢對比 (初始資產為 1.0)"),
                row_heights=[0.5, 0.5]
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df_result['Date'],
                    y=df_result['Adj Close'],
                    mode='lines',
                    name='0050.TW 股價 (Adj Close)',
                    line=dict(color='#9CA3AF', width=2),
                    hovertemplate='日期: %{x}<br>股價: NT$ %{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            if not df_buy.empty:
                long_buys = df_buy[df_buy['Type'] == 'Long Buy']
                if not long_buys.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=long_buys['Date'],
                            y=long_buys['Price'],
                            mode='markers',
                            name='做多買點 (Long Buy)',
                            marker=dict(color='#10B981', size=8, symbol='triangle-up'),
                            hovertemplate='買入日期: %{x}<br>價格: NT$ %{y:.2f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    
            if not df_sell.empty:
                long_sells = df_sell[df_sell['Type'] == 'Long Sell']
                if not long_sells.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=long_sells['Date'],
                            y=long_sells['Price'],
                            mode='markers',
                            name='做多賣點 (Long Sell)',
                            marker=dict(color='#EF4444', size=8, symbol='triangle-down'),
                            hovertemplate='賣出日期: %{x}<br>價格: NT$ %{y:.2f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    
            fig.add_trace(
                go.Scatter(
                    x=df_result['Date'],
                    y=df_result['God_View_Asset'],
                    mode='lines',
                    name='上帝視角 (God-View)',
                    line=dict(color='#10B981', width=3),
                    hovertemplate='日期: %{x}<br>淨值: %{y:,.2f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df_result['Date'],
                    y=df_result['BH_Asset'],
                    mode='lines',
                    name='長期持有 (Buy & Hold)',
                    line=dict(color='#3B82F6', width=2),
                    hovertemplate='日期: %{x}<br>淨值: %{y:,.2f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=750,
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            if is_log:
                fig.update_yaxes(type="log", row=2, col=1)
                
            fig.update_yaxes(title_text="股價 (元)", row=1, col=1)
            fig.update_yaxes(title_text="資產淨值", row=2, col=1)
            fig.update_xaxes(title_text="日期", row=2, col=1)
            
            st.plotly_chart(fig, width='stretch')
            
        with sub_tab2:
            st.subheader("績效統計與數據報表")
            
            summary_data = {
                "績效指標": [
                    "累積報酬率 (Cumulative Return)",
                    "年化報酬率 (CAGR)",
                    "最大回撤 (Max Drawdown)",
                    "模擬期間總交易次數 (Total Trades)"
                ],
                "長期持有 (Buy & Hold)": [
                    f"{metrics['bh_total_ret']:.2f}%",
                    f"{metrics['bh_cagr']:.2f}%",
                    f"{metrics['bh_mdd']:.2f}%",
                    "1"
                ],
                "上帝視角策略 (God-View)": [
                    f"{metrics['gv_total_ret']:.2f}%",
                    f"{metrics['gv_cagr']:.2f}%",
                    f"{metrics['gv_mdd']:.2f}%",
                    f"{len(df_buy)}"
                ],
                "超額表現 (Alpha)": [
                    f"+{metrics['alpha']:.2f}%",
                    f"+{metrics['gv_cagr'] - metrics['bh_cagr']:.2f}%",
                    f"{(metrics['gv_mdd'] - metrics['bh_mdd']):+.2f}%",
                    "-"
                ]
            }
            df_summary = pd.DataFrame(summary_data)
            st.table(df_summary.set_index("績效指標"))
            
            st.subheader("原始回測數據下載")
            df_download = df_result[['Date', 'Adj Close', 'God_View_Asset', 'BH_Asset']].copy()
            df_download.columns = ['日期', '0050還原收盤價', '上帝視角資產淨值', '長期持有資產淨值']
            csv = df_download.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="下載完整回測日資料 (CSV)",
                data=csv,
                file_name="god_view_backtest_results.csv",
                mime="text/csv",
                key="dl_btn_sub"
            )
            
        with sub_tab3:
            st.subheader("待在市場的價值：錯過 0050 關鍵上漲日分析")
            st.markdown("""
            這個實驗驗證了投資學中極起著名的理論：**「上漲只發生在少數幾天，因此你必須隨時待在市場中。」**
            許多投資人試圖透過擇時（Timing）低買高賣，或者在市場下跌時離場避險。然而，這會面臨一個致命的風險：**錯過市場報復性反彈的最佳交易日**。
            """)
            
            df_result['Daily_Return'] = df_result['Adj Close'].pct_change().fillna(0)
            sorted_returns = df_result.sort_values(by='Daily_Return', ascending=False)
            
            miss_scenarios = [0, 5, 10, 20, 50]
            scenario_names = [
                "完全持有 (Buy & Hold)",
                "錯過前 5 大上漲日",
                "錯過前 10 大上漲日",
                "錯過前 20 大上漲日",
                "錯過前 50 大上漲日"
            ]
            scenario_returns = []
            
            for k in miss_scenarios:
                if k == 0:
                    final_val = df_result['BH_Asset'].iloc[-1]
                else:
                    top_k_indices = sorted_returns.head(k).index
                    mod_returns = df_result['Daily_Return'].copy()
                    mod_returns.loc[top_k_indices] = 0.0
                    final_val = (1 + mod_returns).prod()
                scenario_returns.append((final_val - 1.0) * 100)
                
            fig_bar = go.Figure()
            colors = ['#3B82F6', '#F59E0B', '#EF4444', '#DC2626', '#991B1B']
            
            fig_bar.add_trace(go.Bar(
                x=scenario_names,
                y=scenario_returns,
                marker_color=colors,
                text=[f"{r:,.2f}%" for r in scenario_returns],
                textposition='auto',
                hovertemplate='持有情境: %{x}<br>累積報酬率: %{y:.2f}%<extra></extra>'
            ))
            
            fig_bar.update_layout(
                yaxis_title="累積報酬率 (%)",
                xaxis_title="持有情境",
                height=450,
                margin=dict(l=50, r=50, t=20, b=50)
            )
            st.plotly_chart(fig_bar, width='stretch')
            
            col_left, col_right = st.columns([1, 1])
            with col_left:
                st.markdown(f"""
                #### 數據重點回顧：
                * **完全持有 (Buy & Hold)**：累積報酬率為 **{scenario_returns[0]:.2f}%**。
                * **錯過前 5 大上漲日**：累積報酬率降至 **{scenario_returns[1]:.2f}%**。
                * **錯過前 10 大上漲日**：累積報酬率降至 **{scenario_returns[2]:.2f}%**。
                * **錯過前 20 大上漲日**：累積報酬率降至 **{scenario_returns[3]:.2f}%**。
                * **錯過前 50 大上漲日**：累積報酬率降至 **{scenario_returns[4]:.2f}%**。
                
                > 關鍵啟示：上圖清楚顯示，只要您錯過了短短的 **10 天**，累積收益便會面臨腰斬以上的巨幅縮水；錯過 20 天以上甚至會由盈轉虧。
                > 這說明了**「待在市場的時間 (Time in the market)」遠比「預測市場的時機 (Timing the market)」重要**。對於一般散戶，抱緊優質大盤指數 ETF（如 0050），才是最穩健的財富累積之路。
                """)
            with col_right:
                st.markdown("#### 本回測區間 0050 單日漲幅前 10 大交易日")
                top_10_days = sorted_returns.head(10)[['Date', 'Daily_Return']].copy()
                top_10_days['Daily_Return'] = top_10_days['Daily_Return'] * 100
                top_10_days.columns = ['日期', '單日上漲幅 (%)']
                top_10_days['日期'] = top_10_days['日期'].dt.strftime('%Y-%m-%d')
                st.table(top_10_days.reset_index(drop=True))
                
        with sub_tab4:
            st.subheader("演算法運作邏輯解析")
            st.markdown(f"""
            ### 1. 上帝視角策略是如何運作的？
            該演算法將您所選的時間範圍，依據您在側邊欄設定的「回測觀察區間 $N$ 天」進行**非重疊切割**。
            在每一個區間中，策略扮演一個**「擁有未來視角且僅限做多」**的交易者：
            
            * **交易區間 ($t_{{min}} < t_{{max}}$)**:
              當區間內的最低價（買點）在最高價（賣點）之前發生時，策略會在 $t_{{min}}$ 價格最低時滿倉買入，並在 $t_{{max}}$ 價格最高時全數獲利了結。
              其餘時間（$t < t_{{min}}$ 與 $t > t_{{max}}$）皆**空手持有現金**，資產不隨股價回撤而受損。
            
            * **價格下行區間 ($t_{{max}} < t_{{min}}$)**:
              若最高價先於最低價發生，為了遵循僅做多的現股交易規則（無法在買入前先賣出），策略在此區間將**全程空手持有現金**，從而完美避開價格下行段（此區間報酬率為 0%）。
            
            ### 2. 為什麼這個模擬被稱為「上帝視角」？
            在實際的金融市場中，**我們永遠不可能在當前知道未來 $N$ 天的最低點和最高點在哪裡**。
            此模擬器透過歷史數據的後見之明 (Hindsight Bias)，為您展現如果能夠「精準預測極值」的極限獲利能力。
            
            ### 3. 此模擬給我們的啟示
            當您拉大觀察區間 $N$（例如 $N=60$ 或 $N=90$）時，您會發現上帝視角的總報酬會呈現**爆炸性成長**，動輒達到數百萬甚至數億百分比。
            然而，這正是「擇時交易」的矛盾之處：
            * **預測未來極為困難**：只要猜錯一次高低點，實際績效就會大幅折損。
            * **交易成本並未計入**：本模擬未計入頻繁交易產生的證交稅與手續費，真實交易中這些成本會進一步蠶食報酬。
            * **對比長期持有 (Buy & Hold)**：長期持有（存股）雖然在中間會面臨較大的回撤（如 2020 年新冠疫情、2022 年升息循環），但完全不需要花費精力預測高低點，便能穩健享受台灣前 50 大企業隨經濟成長帶來的股利與資本利得。
            """)


if __name__ == '__main__':
    main()
