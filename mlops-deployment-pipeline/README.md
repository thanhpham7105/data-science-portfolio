# MLOps Deployment Pipeline

## Overview
An end-to-end MLOps project: a versioned data pipeline and a containerized prediction API, built around public U.S. flight-delay data (Bureau of Transportation Statistics).

## Data pipeline (`data_pipeline/`)
Iteratively built and hardened a data-import and cleaning pipeline -- both iterations are included to show the progression:
- `01_import_and_format_v1.py` -> `01_import_and_format_v2.py` -- import & column-standardization (v2 adds a column-mapping dictionary, required-column validation, and logging to handle inconsistent source headers)
- `02_filter_and_clean_v1.py` -> `02_filter_and_clean_v2.py` -- data cleaning/filtering
- `03_train_regressor_v1.py` -> `03_train_regressor_v2.py` -- polynomial regression model training
- Versioned the cleaned dataset with **DVC** and tracked experiments with **MLflow** (`MLproject`, `pipeline_env.yaml`)

## API service (`api_service/`)
Built a **FastAPI** prediction service and containerized it with **Docker**:
- `api_v1_baseline.py` -- minimal API with a hard-coded baseline prediction and no input validation
- `api_v2_production.py` -- production version: loads the trained model and feature encodings at startup, validates input format (with docstrings and defensive error handling), returns proper error codes for invalid input, and serves real predictions
- `Dockerfile_v1` / `Dockerfile_v2` -- containerization for each API version
- `test_api_v1.py` / `test_api_v2.py` -- unit tests for the API

## Skills demonstrated
MLOps architecture design, data versioning (DVC), experiment tracking (MLflow), REST API development (FastAPI), input validation & error handling, containerization (Docker), unit testing, iterative/versioned software development.
