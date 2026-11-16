
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.train_model import load_data, train_model
from ml.validation import validate_features
from ml.model_comparison import compare_models 



# Store the trained model and class names in application state.
ml_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Train the ML model when the application starts."""

    X_train, X_test, y_train, y_test, names = load_data()
    ml_state["model"] = train_model(X_train, y_train)
    ml_state["names"] = names

    yield

    ml_state.clear()


app = FastAPI(title="ModelGuard AI", lifespan=lifespan)


class PredictionRequest(BaseModel):
    """Define the expected prediction request format."""

    features: list[float] = Field(min_length=4, max_length=4)


@app.get("/")
def home():
    """Return a simple API status message."""

    return {"message": "Welcome to ModelGuard AI"}


@app.post("/predict")
def predict(request: PredictionRequest):
    """Predict the Iris flower species."""

    features = request.features

    # Validate the input values
    if not validate_features(features):
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail="Features must contain four positive numbers."
        )

    # Generate a prediction using the trained model
    model = ml_state["model"]
    prediction = model.predict([features])[0]

    # Convert the predicted class index to a species name
    species = ml_state["names"][prediction]

    return {"prediction": str(species)}

@app.get("/compare")
def compare_ml_models():
    """Compare the performance of multiple ML models."""

    # Run the model comparison
    results = compare_models()

    # Return evaluation metrics as JSON
    return {"models": results}
