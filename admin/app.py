import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time

st.set_page_config(page_title="Coin Predict MVP", layout="wide")

BASE_URL = os.getenv("API_URL", "http://nginx")


@st.cache_data(ttl=60)
def get_history_data(symbol):
    """Nginx에서 과거 데이터 정적 파일(SSG) 조회"""
    try:
        # 파일명 규칙 적용 (BTC/USDT -> BTC_USDT)
        safe_symbol = symbol.replace("/", "_")
        url = f"{BASE_URL}/static/history_{safe_symbol}.json"

        response = requests.get(url, timeout=5)
        response.raise_for_status()

        data = response.json()
        df = pd.DataFrame(data["data"])  # SSG 구조에 맞게 수정

        # 날짜 변환 (ISO 8601 -> datetime)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df, data.get("updated_at")  # 생성 시점 반환
    except Exception as e:
        st.error(f"Failed to fetch history file: {e}")
        return pd.DataFrame(), None


@st.cache_data(ttl=60)
def get_forecast_data(symbol):
    """Nginx에서 예측 데이터 정적 파일(SSG) 조회"""
    try:
        safe_symbol = symbol.replace("/", "_")
        url = f"{BASE_URL}/static/prediction_{safe_symbol}.json"

        response = requests.get(url, timeout=5)
        response.raise_for_status()

        data = response.json()
        df = pd.DataFrame(data["forecast"])  # SSG 구조에 맞게 수정

        # 날짜 변환
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df, data.get("updated_at")
    except Exception as e:
        st.error(f"Failed to fetch prediction file: {e}")
        return pd.DataFrame(), None


# 차트 그리기 함수
def plot_chart(symbol, history_df, forecast_df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 과거 데이터 (Candlestick)
    fig.add_trace(
        go.Candlestick(
            x=history_df["timestamp"],
            open=history_df["open"],
            high=history_df["high"],
            low=history_df["low"],
            close=history_df["close"],
            name="History",
        ),
        secondary_y=False,
    )

    # 예측 데이터 (Line + Confidence Interval)
    if not forecast_df.empty:
        # 예측선 (yhat)
        fig.add_trace(
            go.Scatter(
                x=forecast_df["timestamp"],
                y=forecast_df["yhat"],
                mode="lines",
                name="Prediction",
                line=dict(color="#ff00ff", width=2, dash="dot"),
            ),
            secondary_y=False,
        )

        # 신뢰구간 (Upper & Lower)
        fig.add_trace(
            go.Scatter(
                x=forecast_df["timestamp"],
                y=forecast_df["yhat_upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_df["timestamp"],
                y=forecast_df["yhat_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",  # Upper와 Lower 사이를 채움
                fillcolor="rgba(255, 0, 255, 0.1)",
                showlegend=False,
            ),
            secondary_y=False,
        )

    fig.update_layout(
        title=f"{symbol} Price Analysis (30 Days + 24h Forecast)",
        xaxis_title="Time (UTC)",
        yaxis_title="Price (USDT)",
        height=600,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
    )

    return fig


# 메인 UI 로직
st.title("Coin Predict Admin Dashboard")
st.markdown("코인 예측 모니터링 시스템")

# 사이드바
st.sidebar.header("Control Panel")
symbol = st.sidebar.selectbox(
    "Target Asset", ["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "DOGE/USDT"]
)

if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()  # 캐시 비우기 (새로고침)

# 메인 화면
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"📊 {symbol} Market Status")

    # API 호출
    with st.spinner("Calling API Server..."):
        history_df = get_history_data(symbol)
        forecast_df, exec_time = get_forecast_data(symbol)

    if not history_df.empty:
        # KPI 계산
        last_close = history_df.iloc[-1]["close"]
        prev_close = history_df.iloc[-2]["close"]
        change = last_close - prev_close
        change_pct = (change / prev_close) * 100

        # 차트 그리기
        fig = plot_chart(symbol, history_df, forecast_df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No historical data found. Please check Ingest Worker.")

with col2:
    st.subheader("System Metrics")
    if not history_df.empty:
        st.metric("Current Price", f"${last_close:,.2f}", f"{change_pct:.2f}%")
        st.metric("DB Records", f"{len(history_df)} rows", "Last 30 Days")

    st.divider()

    st.subheader("Model Inference")
    if not forecast_df.empty:
        st.metric("Inference Time", f"{exec_time:.4f} sec", "CPU Bound")

        # 예측 요약
        last_pred = forecast_df.iloc[-1]["yhat"]
        start_pred = forecast_df.iloc[0]["yhat"]
        pred_change = last_pred - start_pred

        st.write("Next 24h Trend:")
        if pred_change > 0:
            st.success(f"📈 Bullish (+${pred_change:,.2f})")
        else:
            st.error(f"📉 Bearish (-${abs(pred_change):,.2f})")
    else:
        st.error("Model Server Error")

# 하단: 원본 데이터 확인 (디버깅용)
with st.expander("View Raw JSON Response"):
    st.json(
        {
            "history_tail": (
                history_df.tail(3).to_dict(orient="records")
                if not history_df.empty
                else {}
            ),
            "forecast_head": (
                forecast_df.head(3).to_dict(orient="records")
                if not forecast_df.empty
                else {}
            ),
        }
    )
