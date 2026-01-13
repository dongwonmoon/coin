import ccxt
import pandas as pd
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
import os
from dotenv import load_dotenv
import time

load_dotenv()  # .env Load

# 환경 변수 가져오기
INFLUXDB_URL = os.getenv("INFLUXDB_URL")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET")


def fetch_binance_data(symbol="BTC/USDT", timeframe="1h", limit=1000):
    """
    바이낸스에서 OHLCV 데이터를 가져와 DataFrame으로 반환
    """
    print(f"[Binance] Fetching {symbol} ({timeframe})...")
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    # DataFrame 변환
    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    # Timestamp를 datetime 객체로 변환 (UTC 기준)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # InfluxDB는 인덱스가 시간이어야 함
    df.set_index("timestamp", inplace=True)

    print(f"Downloaded {len(df)} rows.")
    return df


def save_to_influxdb(write_api, df, symbol):
    """
    DataFrame을 InfluxDB에 저장
    """
    print(f"[InfluxDB] Saving {symbol} to bucket '{INFLUXDB_BUCKET}'...")

    df["symbol"] = symbol  # Tag 용 (인덱싱할 컬럼)

    # DataFrame을 InfluxDB 포맷으로 변환하여 쓰기
    write_api.write(
        bucket=INFLUXDB_BUCKET,
        org=INFLUXDB_ORG,
        record=df,
        data_frame_measurement_name="ohlcv",  # 테이블 이름
        data_frame_tag_columns=["symbol"],  # 인덱싱할 컬럼
    )


if __name__ == "__main__":
    # 수집할 코인 리스트
    target_coins = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]

    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    write_api = client.write_api(write_options=SYNCHRONOUS)

    try:
        for symbol in target_coins:
            # 데이터 가져오기
            df = fetch_binance_data(symbol)

            save_to_influxdb(write_api=write_api, df=df, symbol=symbol)
            print(f"   -> Saved {symbol}")

        print("All data saved successfully!")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()
