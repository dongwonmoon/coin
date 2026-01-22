from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from influxdb_client import InfluxDBClient
from contextlib import asynccontextmanager
import pandas as pd
import os
import time
from datetime import datetime

# load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    print("Connecting to InfluxDB...")
    client = InfluxDBClient(
        url=INFLUXDB_URL,
        token=INFLUXDB_TOKEN,
        org=INFLUXDB_ORG,
        timeout=10000,  # 타임아웃 설정
        retries=3,  # 연결 끊김 대비 재시도
    )
    yield

    print("Closing InfluxDB connection...")
    client.close()


app = FastAPI(title="Coin Predict API", version="1.0.0", lifespan=lifespan)

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

# Memo: 모델 로드 코드는 임시로 남겨둠.
# 1. Worker에서 모델이 로드되는데, Worker이 1분마다 실행됨.
# 2. 매 실행마다 모델이 로드되는데, 이는 상당히 무거움.
# 최적화가 필요할 것 같음.

# # 서버 시작 시 모델 미리 로드
# @app.on_event("startup")
# async def load_all_models():
#     print(f"Loading models from {MODELS_DIR}...")
#     if not os.path.exists(MODELS_DIR):
#         print("Models directory not found. Skipping model load.")
#         return

#     for filename in os.listdir(MODELS_DIR):
#         if filename.endswith(".json") and filename.startswith("model_"):
#             # 파일명에서 심볼 추출
#             symbol_part = filename.replace("model_", "").replace(".json", "")
#             # 파일명에 _가 있으면 /로 복원 (BTC_USDT -> BTC/USDT)
#             symbol = symbol_part.replace("_", "/")

#             file_path = os.path.join(MODELS_DIR, filename)
#             try:
#                 with open(file_path, "r") as fin:
#                     loaded_models[symbol] = model_from_json(fin.read())
#                 print(f"Model loaded: {symbol}")
#             except Exception as e:
#                 print(f"Failed to load {filename}: {e}")


# InfluxDB 쿼리 헬퍼 함수
def query_influx(symbol: str, measurement: str, days: int = 30):
    query_api = client.query_api()

    # 최근 N일 데이터 조회 + Pivot으로 테이블 형태 변환
    # range stop: 2d -> 미래 데이터도 조회하기 위해 미래 시간까지 범위를 엶.
    query = f"""
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: -{days}d, stop: 2d)
      |> filter(fn: (r) => r["_measurement"] == "{measurement}")
      |> filter(fn: (r) => r["symbol"] == "{symbol}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"], desc: false)
    """

    try:
        df = query_api.query_data_frame(query)
    except Exception as e:
        print(f"DB Query Error: {e}")
        return None
    # finally:
    #     client.close()

    if isinstance(df, list) or df.empty:
        return None

    # InfluxDB 리턴값 정리 ('_time' -> 'timestamp')
    df.rename(columns={"_time": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(
        None
    )  # UTC 정보 제거 (JSON 호환)
    return df


@app.get("/history/{symbol:path}")
def get_history(symbol: str):
    """
    과거 30일치 차트 데이터 반환
    """
    start_time = time.time()
    df = query_influx(symbol, "ohlcv", days=30)

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
def predict_price(symbol: str):
    """
    'prediction' 테이블에서 미리 계산된 데이터를 가져옴
    """
    start_time = time.time()

    # DB에서 예측 결과 조회 (최근 24시간 내 생성된 데이터 중 미래값)
    df = query_influx(symbol, "prediction", days=2)

    if df is None:
        # DB에 아직 예측값이 없을 경우 (Worker가 안 돌았거나 모델이 없을 때)
        raise HTTPException(
            status_code=404, detail="Prediction not ready. Check Pipeline Worker."
        )

    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    df = df[df["timestamp"] > now]

    cols = ["timestamp", "yhat", "yhat_lower", "yhat_upper"]
    available_cols = [c for c in cols if c in df.columns]

    return {
        "symbol": symbol,
        "source": "InfluxDB (Pre-computed)",
        "execution_time": round(time.time() - start_time, 4),
        "forecast": df[available_cols].to_dict(orient="records"),
    }


@app.get("/")
def health_check():
    return {"status": "ok", "models_loaded": list(loaded_models.keys())}


@app.on_event("shutdown")
def shutdown_event():
    client.close()
