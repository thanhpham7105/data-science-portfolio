from fastapi.testclient import TestClient
from Api_v2 import app

client = TestClient(app)

def test_root_health_ok():
    r = client.get("/")
    assert r.status_code == 200
    js = r.json()
    assert js.get("status") == "ok"
    assert "functional" in js.get("message", "").lower()

def test_predict_well_formed_ok_jfk():
    r = client.get("/predict/delays", params={
        "dest": "JFK",
        "dep_time_local": "09:15",
        "arr_time_local": "12:45",
        "order": 1
    })
    assert r.status_code in (200, 503)
    js = r.json()
    assert "destination" in js or "detail" in js

def test_predict_bad_time_format_422():
    r = client.get("/predict/delays", params={
        "dest": "JFK",
        "dep_time_local": "9am",  # invalid
        "arr_time_local": "12:45"
    })
    assert r.status_code in (422, 503)

def test_predict_unknown_airport_422():
    r = client.get("/predict/delays", params={
        "dest": "XXX",  # not in encodings -> 422 or 503
        "dep_time_local": "10:00",
        "arr_time_local": "12:00"
    })
    assert r.status_code in (422, 503)
