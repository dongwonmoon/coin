from locust import HttpUser, task, between
import json


class CoinUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict_btc(self):
        with self.client.get("/predict/BTC/USDT", catch_response=True) as response:

            if response.status_code == 0:
                print(f"\nStatus 0 Detected!")
                print(f"Reason: {response.error}")
                response.failure(f"Network Fail: {response.error}")
                return

            if response.status_code != 200:
                print(
                    f"[ERROR] Status: {response.status_code} | Body: {response.text[:50]}..."
                )
                response.failure(f"Status {response.status_code}")
                return

            try:
                response.json()
            except json.JSONDecodeError:
                response.failure("JSON Decode Error")
