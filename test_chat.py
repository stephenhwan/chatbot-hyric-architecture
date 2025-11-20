import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.model_selection import cross_val_score

from data import get_pipeline_data
from brain import MODEL_FILES


def evaluate_classification_model(model_file, target_name):
    """Evaluate a classification model with detailed metrics."""

    # Load model and test data
    model_info = joblib.load(model_file)
    model = model_info["model"]

    # Get test data
    data = get_pipeline_data(target=target_name)
    x_test = data["x_test"]
    y_test = data["y_test"]

    # Predictions
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)

    # Print classification report
    print(f"\n{'=' * 60}")
    print(f"CLASSIFICATION REPORT: {target_name}")
    print(f"{'=' * 60}")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)

    # ROC-AUC for binary classification
    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, y_proba[:, 1])
        print(f"\nAUC-ROC Score: {auc:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {target_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'confusion_matrix_{target_name}.png', dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to: confusion_matrix_{target_name}.png")

    # Feature importance (top 10)
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        importances = model.named_steps['model'].feature_importances_

        # Get feature names after preprocessing
        feature_names = data["x_test"].columns.tolist()

        # Sort by importance
        indices = np.argsort(importances)[::-1][:10]

        print(f"\nTop 10 Feature Importances:")
        for i, idx in enumerate(indices, 1):
            if idx < len(feature_names):
                print(f"{i}. {feature_names[idx]}: {importances[idx]:.4f}")

    return cm, y_pred, y_proba


def evaluate_regression_model(model_file, target_name):
    """Evaluate a regression model with detailed metrics."""

    # Load model and test data
    model_info = joblib.load(model_file)
    model = model_info["model"]

    # Get test data
    data = get_pipeline_data(target=target_name)
    x_test = data["x_test"]
    y_test = data["y_test"]

    # Predictions
    y_pred = model.predict(x_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n{'=' * 60}")
    print(f"REGRESSION REPORT: {target_name}")
    print(f"{'=' * 60}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Actual Value: {y_test.mean():.4f}")

    # Residual analysis
    residuals = y_test - y_pred
    print(f"\nResidual Analysis:")
    print(f"Mean Residual: {residuals.mean():.4f}")
    print(f"Std Residual: {residuals.std():.4f}")

    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Predictions vs Actual: {target_name}')
    plt.savefig(f'predictions_{target_name}.png', dpi=300, bbox_inches='tight')
    print(f"Saved prediction plot to: predictions_{target_name}.png")

    # Plot residuals
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot: {target_name}')
    plt.savefig(f'residuals_{target_name}.png', dpi=300, bbox_inches='tight')
    print(f"Saved residual plot to: residuals_{target_name}.png")

    return r2, mae, rmse, y_pred


def cross_validate_all_models():
    """Perform cross-validation on all models."""

    print(f"\n{'=' * 60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'=' * 60}")

    results = {}

    # Cancellation model (classification)
    data = get_pipeline_data(target="is_canceled")
    model_info = joblib.load(MODEL_FILES["is_canceled"])
    model = model_info["model"]

    cv_scores = cross_val_score(model, data["x_train"], data["y_train"], cv=5, scoring='accuracy')
    results["is_canceled"] = cv_scores
    print(f"\nis_canceled (Classification):")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Stay nights model (regression)
    data = get_pipeline_data(target="total_stay_nights")
    model_info = joblib.load(MODEL_FILES["total_stay_nights"])
    model = model_info["model"]

    cv_scores = cross_val_score(model, data["x_train"], data["y_train"], cv=5, scoring='r2')
    results["total_stay_nights"] = cv_scores
    print(f"\ntotal_stay_nights (Regression):")
    print(f"  CV R² Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Deposit type model (classification)
    data = get_pipeline_data(target="deposit_type")
    model_info = joblib.load(MODEL_FILES["deposit_type"])
    model = model_info["model"]

    cv_scores = cross_val_score(model, data["x_train"], data["y_train"], cv=5, scoring='accuracy')
    results["deposit_type"] = cv_scores
    print(f"\ndeposit_type (Classification):")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return results


def generate_full_report():
    """Generate complete evaluation report for all models."""

    print("\n" + "=" * 80)
    print(" " * 20 + "MODEL PERFORMANCE EVALUATION REPORT")
    print("=" * 80)

    # 1. Evaluate cancellation model
    evaluate_classification_model(
        MODEL_FILES["is_canceled"],
        "is_canceled"
    )

    # 2. Evaluate stay nights model
    evaluate_regression_model(
        MODEL_FILES["total_stay_nights"],
        "total_stay_nights"
    )

    # 3. Evaluate deposit type model
    evaluate_classification_model(
        MODEL_FILES["deposit_type"],
        "deposit_type"
    )

    # 4. Cross-validation
    cross_validate_all_models()

    print("\n" + "=" * 80)
    print("Report generation complete! Check PNG files for visualizations.")
    print("=" * 80)


if __name__ == "__main__":
    # Make sure models are trained first
    from brain import train_all

    print("Training all models first...")
    train_all()

    print("\n\nGenerating evaluation report...")
    generate_full_report()