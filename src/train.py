"""Machine Learning model development, training, serialization, and inference for CoolBuild AI."""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

try:
    from src.data_loader import (
        load_clean_dataset,
        get_feature_and_target_columns,
        PRETTY_FEATURE_NAMES,
        PRETTY_TARGET_NAMES,
        PROJECT_ROOT
    )
    from src.preprocessing import prepare_train_test_data, FEATURE_CONSTRAINTS
except ImportError:
    from data_loader import (
        load_clean_dataset,
        get_feature_and_target_columns,
        PRETTY_FEATURE_NAMES,
        PRETTY_TARGET_NAMES,
        PROJECT_ROOT
    )
    from preprocessing import prepare_train_test_data, FEATURE_CONSTRAINTS

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RANDOM_STATE = 42

MODEL_CONFIGS = {
    "linear_regression": {
        "name": "Linear Regression",
        "factory": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ]),
        "requires_scaling": True
    },
    "decision_tree": {
        "name": "Decision Tree Regressor",
        "factory": lambda: DecisionTreeRegressor(
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=RANDOM_STATE
        ),
        "requires_scaling": False
    },
    "random_forest": {
        "name": "Random Forest Regressor",
        "factory": lambda: RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=RANDOM_STATE
        ),
        "requires_scaling": False
    },
    "svr": {
        "name": "Support Vector Regression (SVR)",
        "factory": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", SVR(kernel="rbf", C=100.0, epsilon=0.1, gamma="scale"))
        ]),
        "requires_scaling": True
    }
}


def build_model_pipeline(model_key: str):
    """Instantiate a new model pipeline given its key."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key '{model_key}'. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_key]["factory"]()


def train_single_model(
    model_key: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[Any, Dict[str, float]]:
    """Train an individual algorithm and evaluate on train and test splits."""
    model = build_model_pipeline(model_key)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        "train_r2": float(r2_score(y_train, y_train_pred)),
        "train_mae": float(mean_absolute_error(y_train, y_train_pred)),
        "train_rmse": float(root_mean_squared_error(y_train, y_train_pred)),
        "test_r2": float(r2_score(y_test, y_test_pred)),
        "test_mae": float(mean_absolute_error(y_test, y_test_pred)),
        "test_rmse": float(root_mean_squared_error(y_test, y_test_pred)),
    }
    return model, metrics


def train_all_models(save_models: bool = True) -> Dict[str, Any]:
    """
    Train all 4 regression algorithms for both Heating Load and Cooling Load.
    Serializes models and saves a comprehensive training report.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_clean_dataset()
    features, targets = get_feature_and_target_columns()
    
    training_summary = {
        "random_state": RANDOM_STATE,
        "features": features,
        "targets": targets,
        "models": {}
    }
    
    for target in targets:
        print(f"\n==========================================")
        print(f"Training models for target: {target.upper()}")
        print(f"==========================================")
        
        X_train, X_test, y_train, y_test = prepare_train_test_data(
            df, target_name=target, test_size=0.2, random_state=RANDOM_STATE
        )
        
        target_results = {}
        best_r2 = -float("inf")
        best_model_key = None
        best_model_obj = None
        
        for model_key, config in MODEL_CONFIGS.items():
            model_name = config["name"]
            print(f"  Training {model_name}...")
            
            model, metrics = train_single_model(model_key, X_train, y_train, X_test, y_test)
            
            target_results[model_key] = {
                "name": model_name,
                "metrics": metrics
            }
            
            print(f"    Holdout Test -> R²: {metrics['test_r2']:.4f} | MAE: {metrics['test_mae']:.3f} | RMSE: {metrics['test_rmse']:.3f}")
            
            # Save individual model
            if save_models:
                filename = f"{target}_{model_key}.pkl"
                save_path = os.path.join(MODELS_DIR, filename)
                joblib.dump(model, save_path)
                
            if metrics["test_r2"] > best_r2:
                best_r2 = metrics["test_r2"]
                best_model_key = model_key
                best_model_obj = model
                
        # Save canonical primary model for this target
        if save_models and best_model_obj is not None:
            primary_save_path = os.path.join(MODELS_DIR, f"{target}_model.pkl")
            alias_save_path = os.path.join(MODELS_DIR, f"{target.split('_')[0]}_model.pkl")
            joblib.dump(best_model_obj, primary_save_path)
            joblib.dump(best_model_obj, alias_save_path)
            print(f"\n  -> Best Model for {target}: {MODEL_CONFIGS[best_model_key]['name']} (R² = {best_r2:.4f})")
            print(f"  -> Saved as primary model: {primary_save_path} and {alias_save_path}")
            
        training_summary["models"][target] = {
            "best_model": best_model_key,
            "algorithms": target_results
        }
        
    # Write summary json
    summary_path = os.path.join(MODELS_DIR, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=4)
    print(f"\nTraining summary report written to {summary_path}")
    
    return training_summary


# Cache loaded models in memory for fast repeated inference
_MODEL_CACHE: Dict[str, Any] = {}


def load_trained_model(target: str = "cooling_load", algorithm: Optional[str] = None):
    """Load a trained model artifact from disk with memory caching."""
    cache_key = f"{target}_{algorithm if algorithm else 'primary'}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
        
    if algorithm:
        model_path = os.path.join(MODELS_DIR, f"{target}_{algorithm}.pkl")
    else:
        # Check canonical target filename or short alias
        model_path = os.path.join(MODELS_DIR, f"{target}_model.pkl")
        if not os.path.exists(model_path):
            short_alias = os.path.join(MODELS_DIR, f"{target.split('_')[0]}_model.pkl")
            if os.path.exists(short_alias):
                model_path = short_alias
        
    if not os.path.exists(model_path):
        # Trigger training automatically if model files are missing
        print(f"Model file {model_path} not found. Running training...")
        train_all_models(save_models=True)
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Failed to find or generate model at {model_path}")
        
    model = joblib.load(model_path)
    _MODEL_CACHE[cache_key] = model
    return model


def predict_loads(
    building_params: Union[Dict[str, float], pd.DataFrame, pd.Series],
    algorithm: Optional[str] = None
) -> Dict[str, float]:
    """
    Predict Heating and Cooling loads for a given building configuration.
    
    Parameters:
    -----------
    building_params : dict or DataFrame containing all 8 building features.
    algorithm : specific algorithm key ('random_forest', 'decision_tree', 'linear_regression', 'svr').
                Defaults to the primary best model.
                
    Returns:
    --------
    dict with 'heating_load' and 'cooling_load' in kWh/m².
    """
    features, _ = get_feature_and_target_columns()
    
    # Format input into single-row DataFrame
    if isinstance(building_params, dict):
        # Validate all required features are present
        missing = [f for f in features if f not in building_params]
        if missing:
            raise ValueError(f"Missing required building parameters: {missing}")
        input_df = pd.DataFrame([{f: float(building_params[f]) for f in features}])
    elif isinstance(building_params, pd.Series):
        input_df = pd.DataFrame([building_params[features].to_dict()])
    elif isinstance(building_params, pd.DataFrame):
        input_df = building_params[features].copy()
    else:
        raise TypeError(f"Unsupported input type: {type(building_params)}")
        
    heating_model = load_trained_model(target="heating_load", algorithm=algorithm)
    cooling_model = load_trained_model(target="cooling_load", algorithm=algorithm)
    
    heating_pred = float(heating_model.predict(input_df)[0])
    cooling_pred = float(cooling_model.predict(input_df)[0])
    
    return {
        "heating_load": round(heating_pred, 2),
        "cooling_load": round(cooling_pred, 2)
    }


if __name__ == "__main__":
    train_all_models(save_models=True)
    
    # Run test prediction
    test_sample = {
        "relative_compactness": 0.74,
        "surface_area": 686.0,
        "wall_area": 245.0,
        "roof_area": 220.5,
        "overall_height": 3.5,
        "orientation": 3,
        "glazing_area": 0.25,
        "glazing_area_distribution": 1
    }
    
    print("\n=== TEST PREDICTION ON SAMPLE BUILDING ===")
    print("Input parameters:")
    for k, v in test_sample.items():
        print(f"  {k}: {v}")
        
    for algo in ["linear_regression", "decision_tree", "random_forest", "svr"]:
        preds = predict_loads(test_sample, algorithm=algo)
        print(f"Algorithm: {algo:18s} -> Heating: {preds['heating_load']:5.2f} kWh/m² | Cooling: {preds['cooling_load']:5.2f} kWh/m²")
