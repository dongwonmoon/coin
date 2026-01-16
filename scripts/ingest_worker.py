import ccxt
import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
import os
import time
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# 1. 환경 설정 및 연결
load_dotenv()

INFLUXDB_URL = os.getenv("INFLUXDB_URL")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET")

# 수집 대상 및 설정
TARGET_COINS = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "DOGE/USDT"]
TIMEFRAME = "1h"
LOOKBACK_DAYS = 30  # 과거 30일치 데이터 유지


def get_last_timestamp(query_api, symbol):
    """
    InfluxDB에서 해당 코인의 가장 마지막 데이터 시간(Timestamp)을 조회
    """
    query = f"""
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: -{LOOKBACK_DAYS}d)
      |> filter(fn: (r) => r["_measurement"] == "ohlcv")
      |> filter(fn: (r) => r["symbol"] == "{symbol}")
      |> last(column: "_time")
    """
    try:
        result = query_api.query(query=query)
        if len(result) > 0 and len(result[0].records) > 0:
            last_time = result[0].records[0].get_time()
            # InfluxDB 시간은 UTC timezone이 포함됨.
            return last_time
    except Exception as e:
        print(f"[{symbol}] DB 조회 중 에러 (아마 데이터 없음): {e}")

    return None


def fetch_and_save(write_api, symbol, since_ts):
    """
    ccxt로 데이터 가져와서 InfluxDB에 저장
    """
    exchange = ccxt.binance()

    # since_ts가 datetime 객체라면 밀리초(int)로 변환 필요
    if isinstance(since_ts, datetime):
        since_ms = int(since_ts.timestamp() * 1000)
    else:
        since_ms = int(since_ts)  # 이미 int면 그대로

    try:
        # 데이터 가져오기
        ohlcv = exchange.fetch_ohlcv(
            symbol, TIMEFRAME, since=since_ms, limit=1000
        )  # 약 41일 치

        if not ohlcv:
            print(f"[{symbol}] 새로운 데이터 없음.")
            return

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize(
            "UTC"
        )
        df.set_index("timestamp", inplace=True)

        # 태그 추가
        df["symbol"] = symbol

        # 저장
        write_api.write(
            bucket=INFLUXDB_BUCKET,
            org=INFLUXDB_ORG,
            record=df,
            data_frame_measurement_name="ohlcv",
            data_frame_tag_columns=["symbol"],
        )
        print(f"[{symbol}] {len(df)}개 봉 저장 완료 (Last: {df.index[-1]})")

    except Exception as e:
        print(f"[{symbol}] 수집 실패: {e}")


def run_worker():
    print(f"[Ingest Worker] Started. Target: {TARGET_COINS}, Timeframe: {TIMEFRAME}")

    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    write_api = client.write_api(write_options=SYNCHRONOUS)
    query_api = client.query_api()

    while True:
        print(f"\n[Cycle] 작업 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        for symbol in TARGET_COINS:
            # DB에서 마지막 데이터 시간 확인
            last_time = get_last_timestamp(query_api, symbol)

            # 시작 시간 결정 (Since)
            if last_time:
                # 마지막 데이터가 있으면, 그 시간부터 다시 가져옴 (덮어쓰기 업데이트)
                since = last_time
            else:
                # 데이터가 아예 없으면 30일 전부터
                since = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
                print(f"[{symbol}] 초기 데이터 수집 시작 (30일 전부터)")

            # 수집 및 저장
            fetch_and_save(write_api, symbol, since)

        print("1분 대기 중...")
        time.sleep(60)  # 1분마다 반복


if __name__ == "__main__":
    run_worker()
