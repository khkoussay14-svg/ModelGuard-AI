# ModelGuard AI

**Machine Learning Evaluation and Automated Testing Platform**

ModelGuard AI is a Python-based project focused on evaluating machine learning models and testing their reliability through automated tests.

The project currently uses a Decision Tree classifier trained on the Iris dataset and exposes a prediction API built with FastAPI.

## Features

- Iris classification using scikit-learn
- Data validation before prediction
- Model evaluation with accuracy, precision, recall, F1-score, and confusion matrix
- Automated unit, model, and API tests using pytest
- Code coverage reporting with pytest-cov
- Continuous Integration with GitHub Actions
- Interactive API documentation using FastAPI Swagger UI

## Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Machine Learning | scikit-learn, NumPy |
| Backend | FastAPI, Uvicorn |
| Testing | pytest, FastAPI TestClient |
| Code Coverage | pytest-cov |
| CI | GitHub Actions |
| Version Control | Git, GitHub |

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/ModelGuard-AI.git
cd ModelGuard-AI
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Install dependencies

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

The command above is intended for Windows.

### 4. Run automated tests

```bash
.venv/Scripts/python -m pytest -v
```

### 5. Run the API

```bash
.venv/Scripts/python -m uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to explore the API.

## API Example

**POST `/predict`**

Request:

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Response:

```json
{
  "prediction": "setosa"
}
```

## Testing

The current test suite covers input validation, machine learning behavior, and API endpoints.

The latest locally verified test run completed successfully with 13 passing tests.

GitHub Actions automatically runs the test suite and generates a code coverage report on pushes and pull requests.

## Roadmap

- Compare multiple machine learning models
- Add cross-validation and robustness testing
- Implement experiment tracking
- Develop a React dashboard
- Store experiment results
- Add visual evaluation reports

## Project Status

Under active development. The current version provides a working machine learning backend and an automated testing pipeline.
