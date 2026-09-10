from fastapi import FastAPI
app = FastAPI(title="Airport Departure Delay API", version="1")
@app.get("/")
def root():
    return {"status": "ok", "message": "Delay prediction API is functional."}
@app.get("/predict/delays")
def predict_delays(dest: str, dep_time_local: str, arr_time_local: str):
    avg_delay_minutes = 8.5
    return {
        "destination": dest.upper(),
        "dep_time_local": dep_time_local,
        "arr_time_local": arr_time_local,
        "average_departure_delay_minutes": avg_delay_minutes,
        "source": "baseline"
    }
