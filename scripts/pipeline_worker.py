import ccxt
import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
from prophet.serialize import model_from_json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

INFLUXDB_URL = os.getenv("INFLUXDB_URL")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

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


def save_history_to_json(df, symbol):
    """
    과거 데이터(1h) 정적 파일 생성
    TODO: 지금은 단순하게 1시간 봉에 대해서만 정적 파일을 생성하고 있는데, (predict도 마찬가지)
    추후에 확장할 것.
    """
    try:
        export_df = df.copy()
        export_df["timestamp"] = export_df.index.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )  # UTC Aware 가정

        json_output = {
            "symbol": symbol,
            "data": export_df[
                ["timestamp", "open", "high", "low", "close", "volume"]
            ].to_dict(orient="records"),
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "type": "history_1h",
        }

        safe_symbol = symbol.replace("/", "_")
        file_path = STATIC_DIR / f"history_{safe_symbol}.json"

        with open(file_path, "w") as f:
            json.dump(json_output, f)  # indet=None로 용량 절약

        print(f"[{symbol}] 정적 파일 생성 완료: {file_path}")
    except Exception as e:
        print(f"[{symbol}] 정적 파일 생성 실패: {e}")


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

        # TODO: SSG 파일 생성을 DB 조회 후 덮어쓰기로 구현?

    except Exception as e:
        print(f"[{symbol}] 수집 실패: {e}")


def run_prediction_and_save(write_api, symbol):
    """모델 로드 -> 예측 -> 저장"""
    # 모델 로드
    model_file = MODELS_DIR / f"model_{symbol.replace('/', '_')}.json"
    if not model_file.exists():
        print(f"[{symbol}] 모델 없음")
        return

    try:
        with open(model_file, "r") as fin:
            model = model_from_json(fin.read())

        # 예측 (현재 시점부터 24시간)
        now = datetime.now(timezone.utc)
        # .replace(minute=0, second=0, microsecond=0) => UTC 통일
        future = pd.DataFrame({"ds": pd.date_range(start=now, periods=24, freq="H")})
        future["ds"] = future["ds"].dt.tz_localize(None)  # prophet은 tz-naive

        # As-Is: 여기서는 과거 데이터 없이 모델이 기억하는 패턴으로만 예측
        # To-Do: Training Worker 구축
        forecast = model.predict(future)

        # 필요한 데이터만 추출
        next_24h = forecast[forecast["ds"] > now.replace(tzinfo=None)].head(24).copy()

        if next_24h.empty:
            print(f"[{symbol}] 예측 범위 생성 실패.")
            return

        # 저장 (SSG)
        export_data = next_24h[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        export_data["ds"] = pd.to_datetime(export_data["ds"]).dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        export_data.rename(
            columns={
                "ds": "timestamp",
                "yhat": "price",
                "yhat_lower": "lower_bound",
                "yhat_upper": "upper_bound",
            },
            inplace=True,
        )

        json_output = {
            "symbol": symbol,
            "updated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),  # 생성 시점 기록
            "forecast": export_data.to_dict(orient="records"),
        }

        # 파일 저장 (덮어쓰기)
        safe_symbol = symbol.replace("/", "_")
        file_path = STATIC_DIR / f"prediction_{safe_symbol}.json"

        with open(file_path, "w") as f:
            json.dump(json_output, f, indent=2)

        print(f"[{symbol}] SSG 파일 생성 완료: {file_path}")

        # 저장(DB)
        next_24h["ds"] = pd.to_datetime(next_24h["ds"]).dt.tz_localize(
            "UTC"
        )  # 불필요한 연산 같기는 한데,,, 방어용???
        next_24h = next_24h[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        next_24h.rename(columns={"ds": "timestamp"}, inplace=True)
        next_24h.set_index("timestamp", inplace=True)  # InfluxDB는 index가 timestamp
        next_24h["symbol"] = symbol

        write_api.write(
            bucket=INFLUXDB_BUCKET,
            org=INFLUXDB_ORG,
            record=next_24h,
            data_frame_measurement_name="prediction",
            data_frame_tag_columns=["symbol"],
        )
        print(f"[{symbol}] {len(next_24h)}개 예측 저장 완료")

    except Exception as e:
        print(f"[{symbol}] 예측 에러: {e}")


def update_full_history_file(query_api, symbol):
    """DB에서 최근 30일치 데이터를 긁어와서 history json 파일 갱신"""
    query = f"""
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: -30d)
      |> filter(fn: (r) => r["_measurement"] == "ohlcv")
      |> filter(fn: (r) => r["symbol"] == "{symbol}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"], desc: false)
    """
    try:
        df = query_api.query_data_frame(query)
        if not df.empty:
            df.rename(columns={"_time": "timestamp"}, inplace=True)  # UTC Aware
            df.set_index("timestamp", inplace=True)
            save_history_to_json(df, symbol)
    except Exception as e:
        print(f"[{symbol}] History 갱신 중 에러: {e}")


def run_worker():
    print(f"[Pipeline Worker] Started. Target: {TARGET_COINS}, Timeframe: {TIMEFRAME}")

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

            # 수집
            fetch_and_save(write_api, symbol, since)

            # History 파일 갱신
            update_full_history_file(query_api, symbol)

            # 예측
            run_prediction_and_save(write_api, symbol)

        print("1분 대기 중...")
        time.sleep(60)  # 1분마다 반복


if __name__ == "__main__":
    run_worker()
