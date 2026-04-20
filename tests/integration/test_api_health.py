from fastapi.testclient import TestClient

from api.main import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_model_info_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/model-info")
    assert response.status_code == 200
    payload = response.json()
    assert "model_version" in payload
    assert "threshold" in payload


def test_predict_endpoint_returns_model_metadata() -> None:
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "age": 53,
            "sex": 1,
            "resting_bp": 132.0,
            "cholesterol": 246.0,
            "restecg": 1,
            "max_heart_rate": 151.0,
            "slope": 1,
            "ca": 0,
            "thal": 2,
            "fasting_blood_sugar": 0,
            "exercise_angina": 1,
            "chest_pain_type": 2,
            "oldpeak": 1.3,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert 0.0 <= payload["risk_score"] <= 1.0
    assert "model_version" in payload
    assert "threshold" in payload
