from fastapi import FastAPI, HTTPException
from influxdb_client import InfluxDBClient
import pandas as pd
from prophet.serialize import model_from_json
import os
from dotenv import load_dotenv
import time
from pathlib import Path

load_dotenv()

app = FastAPI(title="Coin Predict API")

# 환경 변수
INFLUXDB_URL = os.getenv("INFLUXDB_URL")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "models" / "model_BTC_USDT.json"
loaded_models = {}

# 서버 시작 시 모델 로드
# 추후 lifespan + app.state로 이관
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "r") as fin:
        loaded_models["BTC/USDT"] = model_from_json(fin.read())
        print("Pre-trained model loaded.")
else:
    print("Warning: Model file not found. Run train_model.py first.")


@app.get("/predict/{symbol:path}")
async def predict_price(symbol: str):
    start_time = time.time()

    # 모델이 없으면 에러
    if symbol not in loaded_models:
        raise HTTPException(status_code=404, detail="Model not trained for this symbol")

    model = loaded_models[symbol]

    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    query_api = client.query_api()

    # 최근 7일치 데이터를 긁어옴
    query = f"""
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: -7d)
      |> filter(fn: (r) => r["_measurement"] == "ohlcv")
      |> filter(fn: (r) => r["symbol"] == "{symbol}")
      |> filter(fn: (r) => r["_field"] == "close")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    """

    result = query_api.query_data_frame(query)
    client.close()

    if result.empty:
        raise HTTPException(status_code=404, detail="No data in DB")

    # 데이터 가공
    df = result[["_time", "close"]].rename(columns={"_time": "ds", "close": "y"})
    df["ds"] = df["ds"].dt.tz_localize(None)

    # predict
    future = model.make_future_dataframe(periods=24, freq="H")
    forecast = model.predict(future)

    next_24h = forecast.tail(24)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    return {
        "symbol": symbol,
        "execution_time": round(time.time() - start_time, 4),
        "forecast": next_24h.to_dict(orient="records"),
    }


@app.get("/history/{symbol:path}")
async def get_history(symbol: str):
    """
    InfluxDB에 저장된 과거 데이터 조회 (프론트엔드 차트용)
    """
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    query_api = client.query_api()

    # 최근 30일치 데이터 조회 (추후 range 조정)
    # pivot을 써서 _time, open, high, low, close, volume 형태로 정렬
    query = f"""
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: -30d)
      |> filter(fn: (r) => r["_measurement"] == "ohlcv")
      |> filter(fn: (r) => r["symbol"] == "{symbol}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    """

    result = query_api.query_data_frame(query)
    client.close()

    if result.empty:
        raise HTTPException(status_code=404, detail="No history data found")

    # DataFrame -> Dict 변환
    df = result.rename(columns={"_time": "timestamp"})
    # 필요한 컬럼만 선택
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    existing_cols = [c for c in columns if c in df.columns]

    return {"symbol": symbol, "data": df[existing_cols].to_dict(orient="records")}
