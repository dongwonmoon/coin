import ccxt
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def train_and_save(symbol="BTC/USDT"):
    print(f"[{symbol}] 모델 학습 시작...")

    # 데이터 수집 (최근 500시간)
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, "1h", limit=500)
    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Prophet 데이터 포맷 준비
    train_df = df[["timestamp", "close"]].rename(
        columns={"timestamp": "ds", "close": "y"}
    )

    # 학습
    model = Prophet(daily_seasonality=True)
    model.fit(train_df)

    # 저장 (JSON 형식으로 시리얼라이즈)
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    file_name = BASE_DIR / "models" / f"model_{symbol.replace('/', '_')}.json"
    with open(file_name, "w") as fout:
        fout.write(model_to_json(model))

    print(f"[{file_name}] 모델 저장 완료!")


if __name__ == "__main__":
    for symbol in ["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "DOGE/USDT"]:
        train_and_save(symbol)
