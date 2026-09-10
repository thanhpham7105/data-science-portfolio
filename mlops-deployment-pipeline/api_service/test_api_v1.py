from fastapi.testclient import TestClient
from Api_v1 import app

client = TestClient(app)

def test_root_health_ok():
    r = client.get("/")
    assert r.status_code == 200
    js = r.json()
    assert js.get("status") == "ok"
    assert "functional" in js.get("message", "").lower()

def test_predict_good_request_lax():
    r = client.get("/predict/delays", params={
        "dest": "JFK",
        "dep_time_local": "09:15",
        "arr_time_local": "12:45"
    })
    assert r.status_code == 200
    js = r.json()
    assert js["destination"] == "JFK"
    assert isinstance(js["average_departure_delay_minutes"], (int, float))

def test_predict_bad_time_still_responds_baseline():

    r = client.get("/predict/delays", params={
        "dest": "JFK",
        "dep_time_local": "9am",
        "arr_time_local": "12:45"
    })
    assert r.status_code in (200, 422)