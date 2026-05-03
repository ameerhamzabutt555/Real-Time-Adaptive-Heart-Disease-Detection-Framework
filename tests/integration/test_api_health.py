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


def test_adaptive_predict_and_learn_cycle() -> None:
    client = TestClient(app)
    patient_payload = {
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
    }
    status_before = client.get("/adaptive/status")
    assert status_before.status_code == 200
    seen_before = status_before.json()["seen_samples"]

    predict_response = client.post("/adaptive/predict", json=patient_payload)
    assert predict_response.status_code == 200
    predict_payload = predict_response.json()
    assert 0.0 <= predict_payload["risk_score"] <= 1.0
    assert predict_payload["model_version"] == "adaptive-online-v1"

    learn_response = client.post("/adaptive/learn", json={"patient": patient_payload, "label": 1})
    assert learn_response.status_code == 200
    learn_payload = learn_response.json()
    assert learn_payload["true_label"] == 1
    assert learn_payload["detector"] in {"adwin", "ddm", "page_hinkley"}
    assert "drift_detected" in learn_payload
    assert learn_payload["seen_samples"] == seen_before + 1

    status_after = client.get("/adaptive/status")
    assert status_after.status_code == 200
    assert status_after.json()["seen_samples"] == seen_before + 1
