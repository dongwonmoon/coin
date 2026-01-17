from locust import HttpUser, task, between


class CoinUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict_btc(self):
        with self.client.get("/predict/BTC/USDT") as response:
            if response.status_code != 200:
                response.failure(f"Status code: {response.status_code}")
