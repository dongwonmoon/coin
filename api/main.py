from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from influxdb_client import InfluxDBClient
import pandas as pd
from prophet.serialize import model_from_json
import os
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="ProphetOps API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MVP에선 편의상 모두 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경 변수
INFLUXDB_URL = os.getenv("INFLUXDB_URL")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET")

# 모델 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 전역 모델 저장소
loaded_models = {}


# 서버 시작 시 모델 미리 로드
@app.on_event("startup")
async def load_all_models():
    print(f"Loading models from {MODELS_DIR}...")
    if not os.path.exists(MODELS_DIR):
        print("Models directory not found. Skipping model load.")
        return

    for filename in os.listdir(MODELS_DIR):
        if filename.endswith(".json") and filename.startswith("model_"):
            # 파일명에서 심볼 추출
            symbol_part = filename.replace("model_", "").replace(".json", "")
            # 파일명에 _가 있으면 /로 복원 (BTC_USDT -> BTC/USDT)
            symbol = symbol_part.replace("_", "/")

            file_path = os.path.join(MODELS_DIR, filename)
            try:
                with open(file_path, "r") as fin:
                    loaded_models[symbol] = model_from_json(fin.read())
                print(f"Model loaded: {symbol}")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")


# InfluxDB 쿼리 헬퍼 함수
def query_influx(symbol: str, days: int = 30):
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    query_api = client.query_api()

    # 최근 N일 데이터 조회 + Pivot으로 테이블 형태 변환
    query = f"""
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: -{days}d)
      |> filter(fn: (r) => r["_measurement"] == "ohlcv")
      |> filter(fn: (r) => r["symbol"] == "{symbol}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    """

    df = query_api.query_data_frame(query)
    client.close()

    if df.empty:
        return None

    # InfluxDB 리턴값 정리 ('_time' -> 'timestamp')
    df.rename(columns={"_time": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(
        None
    )  # UTC 정보 제거 (JSON 호환)
    return df


@app.get("/history/{symbol:path}")
async def get_history(symbol: str):
    """
    과거 30일치 차트 데이터 반환
    """
    start_time = time.time()
    df = query_influx(symbol, days=30)

    if df is None:
        raise HTTPException(status_code=404, detail=f"No history data for {symbol}")

    # 필요한 컬럼만 추출
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    available_cols = [c for c in cols if c in df.columns]

    return {
        "symbol": symbol,
        "count": len(df),
        "execution_time": round(time.time() - start_time, 4),
        "data": df[available_cols].to_dict(orient="records"),
    }


@app.get("/predict/{symbol:path}")
async def predict_price(symbol: str):
    """
    향후 24시간 예측 데이터 반환
    """
    start_time = time.time()

    # 모델 확인
    if symbol not in loaded_models:
        raise HTTPException(
            status_code=404, detail=f"Model not found for {symbol}. Train it first."
        )

    model = loaded_models[symbol]

    now = datetime.now(timezone.utc)

    # prophet 모델은 학습했을 때를 기준으로 추후 데이터를 예측하기 때문에,
    # InfluxDB를 조회해서 새로운 데이터가 감지되면 prophet을 재학습하는 등의 로직이 필요함
    future_dates = pd.date_range(start=now, periods=25, freq="H", tz="UTC")

    future = pd.DateFrame({"ds": future_dates})
    future["ds"] = future["ds"].dt.tz_localize(None)  # UTC 정보 제거 (JSON 호환)
    forecast = model.predict(future)

    next_24h = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    return {
        "symbol": symbol,
        "execution_time": round(time.time() - start_time, 4),
        "forecast": next_24h.rename(columns={"ds": "timestamp"}).to_dict(
            orient="records"
        ),
    }


@app.get("/")
def health_check():
    return {"status": "ok", "models_loaded": list(loaded_models.keys())}
