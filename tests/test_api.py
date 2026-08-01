import unittest
from fastapi.testclient import TestClient
from api import app, MODELS

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "running")
        self.assertIn("version", data)

    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("LPG Catering", response.text)

    def test_metrics_endpoint(self):
        response = self.client.get("/api/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("model_performance", data)
        self.assertIn("dataset", data)
        self.assertIn("cylinder_price_inr", data)

    def test_caterers_endpoint(self):
        response = self.client.get("/api/caterers")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertTrue(len(data["caterers"]) > 0)

    def test_predict_endpoint_fallback(self):
        # Even if models aren't present/trained, predict must fallback gracefully to rules
        payload = {
            "caterer_id": "CAT001",
            "caterer_name": "Test Murugan",
            "experience_yrs": 8,
            "num_burners": 8,
            "business_size": "medium",
            "event_date": "2026-11-15",
            "event_type": "wedding",
            "headcount": 500,
            "num_dishes": 7,
            "duration_hrs": 6.0,
            "menu_profile": "mixed_standard",
            "is_festival_season": True
        }
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("predicted_consumption_kg", data["data"])
        self.assertIn("cylinders_to_order", data["data"])

    def test_batch_optimize(self):
        payload = {
            "events": [
                {
                    "caterer_id": "CAT001",
                    "caterer_name": "Test Murugan",
                    "experience_yrs": 8,
                    "num_burners": 8,
                    "business_size": "medium",
                    "event_date": "2026-11-15",
                    "event_type": "wedding",
                    "headcount": 300,
                    "num_dishes": 6,
                    "duration_hrs": 5.0,
                    "menu_profile": "mixed_standard",
                    "is_festival_season": False
                }
            ],
            "budget_inr": 20000.0,
            "use_lp": True
        }
        response = self.client.post("/api/batch-optimize", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("lp_optimization", data)
        self.assertIn("summary", data)

if __name__ == "__main__":
    unittest.main()
