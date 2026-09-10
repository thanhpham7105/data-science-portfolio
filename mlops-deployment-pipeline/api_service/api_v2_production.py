#!/usr/bin/env python
# coding: utf-8

# import statements
from fastapi import FastAPI, HTTPException
import json
import numpy as np
import pickle
import datetime
import os

app = FastAPI(title="Airport Departure Delay API", version="1.0.0")

# Import the airport encodings file
f = open('airport_encodings.json')
# returns JSON object as a dictionary
airports = json.load(f)
f.close()

def create_airport_encoding(airport: str, airports: dict) -> np.array:
    """
    create_airport_encoding is a function that creates an array the length of all arrival airports from the chosen
    departure aiport.  The array consists of all zeros except for the specified arrival airport, which is a 1.

    Parameters
    ----------
    airport : str
        The specified arrival airport code as a string
    airports: dict
        A dictionary containing all of the arrival airport codes served from the chosen departure airport

    Returns
    -------
    np.array
        A NumPy array the length of the number of arrival airports.  All zeros except for a single 1
        denoting the arrival airport.  Returns None if arrival airport is not found in the input list.
        This is a one-hot encoded airport array.

    """
    # NOTE: airports mapping is expected like {"JFK": 0, "LAX": 1, ...}
    if not isinstance(airports, dict) or len(airports) == 0:
        return None
    size = max(airports.values()) + 1 if airports else 0
    if size <= 0:
        return None
    temp = np.zeros(size, dtype=float)
    idx = airports.get(airport.upper())
    if idx is None or idx < 0 or idx >= size:
        return None
    temp[idx] = 1.0
    return temp

# -----------------------------------------------------------------------------
# TODO:  write the back-end logic to provide a prediction given the inputs
# requires finalized_model.pkl to be loaded
# the model must be passed a NumPy array consisting of the following:
# (polynomial order, encoded airport array, departure time as seconds since midnight, arrival time as seconds since midnight)
# the polynomial order is 1 unless you changed it during model training in Task 2
# YOUR CODE GOES HERE
# -----------------------------------------------------------------------------

# Load the trained model (finalized_model.pkl)
MODEL = None
MODEL_PATH = os.getenv("MODEL_PATH", "finalized_model.pkl")
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as mf:
            MODEL = pickle.load(mf)
    except Exception:
        MODEL = None  # keep API alive even if model can't load

def hhmm_to_seconds(hhmm: str) -> int:
    """
    Convert HH:MM (24-hour) to seconds since midnight.
    Raises ValueError on invalid format.
    """
    try:
        hh, mm = hhmm.split(":")
        h = int(hh);
        m = int(mm)
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError
        return h * 3600 + m * 60
    except Exception:
        raise ValueError("Time must be in HH:MM (24-hour) format")

def predict_average_delay_minutes(dest: str, dep_time_local: str, arr_time_local: str, order: int = 1) -> float:
    """
    Builds the feature vector and returns the model's predicted average departure delay (minutes).
    Feature vector layout (per template):
      [ order, one-hot(dest), dep_seconds_since_midnight, arr_seconds_since_midnight ]
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded; cannot compute prediction.")

    # Convert times to seconds (validates HH:MM format)
    dep_sec = hhmm_to_seconds(dep_time_local)
    arr_sec = hhmm_to_seconds(arr_time_local)

    # One-hot encode destination
    one_hot = create_airport_encoding(dest, airports)
    if one_hot is None:
        raise HTTPException(status_code=422, detail=f"Unknown destination airport: {dest}")

    # Assemble the exact feature vector the model expects
    try:
        features = np.hstack([
            np.array([float(order)], dtype=float),
            one_hot.astype(float),
            np.array([float(dep_sec), float(arr_sec)], dtype=float)
        ]).reshape(1, -1)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to construct feature vector.")

    # Inference
    try:
        y = MODEL.predict(features)
        pred = float(y[0])
        # Defensive clamp to keep outputs in a reasonable window
        pred = max(min(pred, 300.0), -120.0)
        return pred
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

# -----------------------------------------------------------------------------
# TODO:  write the API endpoints.
# YOUR CODE GOES HERE
# -----------------------------------------------------------------------------

@app.get("/")
def root():
    """Return a JSON message indicating that the API is functional."""
    return {"status": "ok", "message": "Delay prediction API is functional."}

@app.get("/predict/delays")
def predict_delays(dest: str, dep_time_local: str, arr_time_local: str, order: int = 1):
    """
    Accepts:
      - dest: arrival airport (IATA)
      - dep_time_local: HH:MM (24h)
      - arr_time_local: HH:MM (24h)
      - order: polynomial order (default 1)
    Returns:
      JSON with the average departure delay in minutes.
    """
    # Basic input check (keeping the template’s simple style)
    if not isinstance(dest, str) or len(dest.strip()) == 0:
        raise HTTPException(status_code=422, detail="Destination airport code is required.")

    # Delegate to the back-end prediction logic
    avg_delay = predict_average_delay_minutes(
        dest=dest,
        dep_time_local=dep_time_local,
        arr_time_local=arr_time_local,
        order=order
    )

    return {
        "destination": dest.upper(),
        "dep_time_local": dep_time_local,
        "arr_time_local": arr_time_local,
        "average_departure_delay_minutes": round(avg_delay, 2),
        "source": "model"
    }
