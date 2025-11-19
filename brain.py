import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

from data import get_pipeline_data

MODEL_FILES = {
    "is_canceled": "model_is_canceled.pkl",
    "total_stay_nights": "model_total_stay_nights.pkl",
    "deposit_type": "model_deposit_type.pkl",
}
NUMERIC_FEATURES = [
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "is_canceled",
]

CATEGORICAL_FEATURES = [
    "market_segment",
    "distribution_channel",
    "customer_type",
    "deposit_type",
]

def _build_pipeline(numeric, categorical, task: str):

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    if task == "regression":
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    return Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model),
    ])


# ===================== TRAINING FUNCTIONS ===================== #

def train_is_canceled(csv_file: str = "hotel_bookings.csv"):
    """Train a classification model to predict is_canceled (0/1)."""
    target = "is_canceled"
    data = get_pipeline_data(csv_file, target=target)

    x_train = data["x_train"]
    x_test = data["x_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    numeric = data["numeric"]
    categorical = data["categorical"]

    pipe = _build_pipeline(numeric, categorical, task="classification")
    pipe.fit(x_train, y_train)

    y_pred = pipe.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    feature_names = x_train.columns.tolist()
    joblib.dump(
        {"model": pipe, "features": feature_names, "target": target},
        MODEL_FILES[target]
    )

    print(f"[train_is_canceled] Accuracy: {acc:.2%}")
    return acc


def train_total_stay_nights(csv_file: str = "hotel_bookings.csv"):
    target = "total_stay_nights"
    data = get_pipeline_data(csv_file, target=target)

    x_train = data["x_train"]
    x_test = data["x_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    numeric = data["numeric"]
    categorical = data["categorical"]

    pipe = _build_pipeline(numeric, categorical, task="regression")
    pipe.fit(x_train, y_train)

    y_pred = pipe.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    feature_names = x_train.columns.tolist()
    joblib.dump(
        {"model": pipe, "features": feature_names, "target": target},
        MODEL_FILES[target]
    )

    print(f"[train_total_stay_nights] R²: {r2:.3f}, MAE: {mae:.3f}")
    return r2, mae


def train_deposit_type(csv_file: str = "hotel_bookings.csv"):

    target = "deposit_type"
    data = get_pipeline_data(csv_file, target=target)

    x_train = data["x_train"]
    x_test = data["x_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    numeric = data["numeric"]
    categorical = data["categorical"]

    pipe = _build_pipeline(numeric, categorical, task="classification")
    pipe.fit(x_train, y_train)

    y_pred = pipe.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    feature_names = x_train.columns.tolist()
    joblib.dump(
        {"model": pipe, "features": feature_names, "target": target},
        MODEL_FILES[target]
    )

    print(f"[train_deposit_type] Accuracy: {acc:.2%}")
    return acc


def train_all(csv_file: str = "hotel_bookings.csv"):
    """Train all three models."""
    print("=== Train is_canceled (classification) ===")
    train_is_canceled(csv_file)

    print("\n=== Train total_stay_nights (regression) ===")
    train_total_stay_nights(csv_file)

    print("\n=== Train deposit_type (classification) ===")
    train_deposit_type(csv_file)


# ===================== PREDICTION FUNCTIONS ===================== #

def _predict_with_model(target: str, booking_data: dict):
    # ADD TYPE CHECKING HERE ✅
    if not isinstance(booking_data, dict):
        raise TypeError(
            f"booking_data must be a dict, got {type(booking_data).__name__}. "
            f"Value: {booking_data}"
        )

    model_info = joblib.load(MODEL_FILES[target])
    model = model_info["model"]
    features = model_info["features"]

    row = {}
    for f in features:
        if f in booking_data:
            # user đã cung cấp
            row[f] = booking_data[f]
        elif f in NUMERIC_FEATURES:
            # thiếu -> numeric default
            row[f] = 0
        elif f in CATEGORICAL_FEATURES:
            # thiếu -> categorical default
            row[f] = "Unknown"
        else:
            # fallback an toàn nếu có feature mới mà quên update list
            row[f] = 0

    df = pd.DataFrame([row])
    return model, df


def predict_is_canceled(booking_data: dict) -> str:
    try:
        model, df = _predict_with_model("is_canceled", booking_data)
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1]  # probability of class 1

        status = "WILL CANCEL" if pred == 1 else "WILL NOT CANCEL"
        risk = "HIGH" if proba > 0.6 else "MEDIUM" if proba > 0.3 else "LOW"

        return (
            f"Prediction (is_canceled): {status}\n"
            f"  Probability: {proba:.1%}\n"
            f"  Risk level: {risk}"
        )
    except TypeError as e:
        return f"Error: {e}\nPlease pass a dictionary with booking data, not a string."
    except FileNotFoundError:
        return "Model for is_canceled not found. Train it first."
    except Exception as e:
        return f"Error in predict_is_canceled: {e}"


def predict_total_stay_nights(booking_data: dict) -> str:
    try:
        model, df = _predict_with_model("total_stay_nights", booking_data)
        pred = model.predict(df)[0]

        return f"Prediction (total_stay_nights): {pred:.2f} nights"
    except TypeError as e:
        return f"Error: {e}\nPlease pass a dictionary with booking data, not a string."
    except FileNotFoundError:
        return "Model for total_stay_nights not found. Train it first."
    except Exception as e:
        return f"Error in predict_total_stay_nights: {e}"


def predict_deposit_type(booking_data: dict) -> str:

    try:
        model, df = _predict_with_model("deposit_type", booking_data)
        pred = model.predict(df)[0]

        # Optional: show top probability
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0]
            classes = model.classes_
            prob_dict = {cls: float(p) for cls, p in zip(classes, proba)}
            prob_str = ", ".join(f"{cls}: {p:.1%}" for cls, p in prob_dict.items())
        else:
            prob_str = "probabilities not available for this model"

        return (
            f"Prediction (deposit_type): {pred}\n"
            f"  Class probabilities: {prob_str}"
        )
    except TypeError as e:
        return f"Error: {e}\nPlease pass a dictionary with booking data, not a string."
    except FileNotFoundError:
        return "Model for deposit_type not found. Train it first."
    except Exception as e:
        return f"Error in predict_deposit_type: {e}"