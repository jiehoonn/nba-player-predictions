"""
Model Training Module - Replicates Notebooks 04-05

Trains baseline (Ridge) and advanced (XGBoost) models.

Usage:
    python -m src.train_models
    OR
    from src.train_models import train_all_models
    train_all_models()
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def load_data(data_dir: str = 'data/processed') -> tuple:
    """Load train/val splits and feature metadata."""
    logger.info("Loading processed data...")

    data_dir = Path(data_dir)

    train = pd.read_parquet(data_dir / 'train.parquet')
    val = pd.read_parquet(data_dir / 'val.parquet')

    with open(data_dir / 'feature_metadata_v2.json', 'r') as f:
        metadata = json.load(f)

    feature_names = metadata['feature_names']

    logger.info(f"  ✓ Train: {len(train):,} games")
    logger.info(f"  ✓ Val:   {len(val):,} games")
    logger.info(f"  ✓ Features: {len(feature_names)}")

    return train, val, feature_names


def calculate_baseline_performance(val: pd.DataFrame) -> dict:
    """Calculate rolling average baseline (5-game avg)."""
    logger.info("\nCalculating rolling average baseline...")

    baselines = {}
    for target in ['PTS', 'REB', 'AST']:
        mae = mean_absolute_error(val[target], val[f'{target}_LAST_5'])
        r2 = r2_score(val[target], val[f'{target}_LAST_5'])
        baselines[target] = {'mae': mae, 'r2': r2}
        logger.info(f"  {target}: MAE={mae:.3f}, R²={r2:.3f}")

    return baselines


def train_ridge_models(
    X_train: pd.DataFrame,
    y_train: dict,
    X_val: pd.DataFrame,
    y_val: dict
) -> dict:
    """
    Train Ridge regression models (L2 regularization).

    Replicates Notebook 04 logic.
    """
    logger.info("\n" + "="*70)
    logger.info("TRAINING RIDGE REGRESSION MODELS")
    logger.info("="*70)

    # Optimal alphas from Notebook 04 hyperparameter tuning
    best_alphas = {'PTS': 10.0, 'REB': 1.0, 'AST': 100.0}

    models = {}
    results = {}

    for target in ['PTS', 'REB', 'AST']:
        logger.info(f"\nTraining Ridge for {target} (α={best_alphas[target]})...")

        model = Ridge(alpha=best_alphas[target], random_state=RANDOM_STATE)
        model.fit(X_train, y_train[target])

        # Evaluate
        val_pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val[target], val_pred)
        val_r2 = r2_score(y_val[target], val_pred)

        models[target] = model
        results[target] = {
            'model_type': 'Ridge',
            'alpha': best_alphas[target],
            'val_mae': val_mae,
            'val_r2': val_r2
        }

        logger.info(f"  Val MAE: {val_mae:.3f}, R²: {val_r2:.3f}")

    logger.info("\n✓ Ridge models trained")
    return models, results


def train_xgboost_models(
    X_train: pd.DataFrame,
    y_train: dict,
    X_val: pd.DataFrame,
    y_val: dict,
    quick_mode: bool = True
) -> dict:
    """
    Train XGBoost models.

    Replicates Notebook 05 logic.

    Args:
        quick_mode: If True, use pre-tuned hyperparameters (fast)
                   If False, run RandomizedSearchCV (slow)
    """
    logger.info("\n" + "="*70)
    logger.info("TRAINING XGBOOST MODELS")
    logger.info("="*70)

    if quick_mode:
        # Pre-tuned hyperparameters from Notebook 05
        best_params = {
            'PTS': {
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'min_child_weight': 5
            },
            'REB': {
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 1.0,
                'reg_lambda': 1.0,
                'min_child_weight': 3
            },
            'AST': {
                'n_estimators': 500,
                'learning_rate': 0.01,
                'max_depth': 3,
                'subsample': 0.7,
                'colsample_bytree': 1.0,
                'reg_alpha': 0,
                'reg_lambda': 10.0,
                'min_child_weight': 5
            }
        }
    else:
        # Run hyperparameter search (slow)
        logger.warning("  Running hyperparameter search (will take 15-30 minutes)...")
        best_params = {}

        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1.0],
            'reg_lambda': [1.0, 10.0, 100.0],
            'min_child_weight': [1, 3, 5]
        }

        for target in ['PTS', 'REB', 'AST']:
            xgb = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)

            search = RandomizedSearchCV(
                xgb, param_grid, n_iter=30, scoring='neg_mean_absolute_error',
                cv=3, random_state=RANDOM_STATE, n_jobs=-1, verbose=1
            )

            search.fit(X_train, y_train[target])
            best_params[target] = search.best_params_
            logger.info(f"  {target}: Best params = {best_params[target]}")

    models = {}
    results = {}

    for target in ['PTS', 'REB', 'AST']:
        logger.info(f"\nTraining XGBoost for {target}...")

        model = XGBRegressor(
            **best_params[target],
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )

        model.fit(X_train, y_train[target])

        # Evaluate
        val_pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val[target], val_pred)
        val_r2 = r2_score(y_val[target], val_pred)

        models[target] = model
        results[target] = {
            'model_type': 'XGBoost',
            'params': best_params[target],
            'val_mae': val_mae,
            'val_r2': val_r2
        }

        logger.info(f"  Val MAE: {val_mae:.3f}, R²: {val_r2:.3f}")

    logger.info("\n✓ XGBoost models trained")
    return models, results


def save_models(
    ridge_models: dict,
    xgb_models: dict,
    ridge_results: dict,
    xgb_results: dict,
    baselines: dict,
    output_dir: str = 'results/models'
) -> None:
    """Save trained models and results."""
    logger.info("\nSaving models...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Ridge models
    for target, model in ridge_models.items():
        path = output_dir / f'best_ridge_{target.lower()}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"  ✓ {path}")

    # Save XGBoost models
    for target, model in xgb_models.items():
        path = output_dir / f'best_xgb_{target.lower()}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"  ✓ {path}")

    # Save results JSON
    results_summary = {
        'date_created': pd.Timestamp.now().isoformat(),
        'baselines': baselines,
        'ridge': ridge_results,
        'xgboost': xgb_results,
        'best_models': {}
    }

    # Determine best model per target
    for target in ['PTS', 'REB', 'AST']:
        ridge_mae = ridge_results[target]['val_mae']
        xgb_mae = xgb_results[target]['val_mae']

        if ridge_mae <= xgb_mae:
            results_summary['best_models'][target] = {
                'model': 'Ridge',
                'val_mae': ridge_mae,
                'val_r2': ridge_results[target]['val_r2']
            }
        else:
            results_summary['best_models'][target] = {
                'model': 'XGBoost',
                'val_mae': xgb_mae,
                'val_r2': xgb_results[target]['val_r2']
            }

    # Save baseline results (rolling avg + Ridge models)
    # Structure matches generate_figures.py expectations
    baseline_output = {
        'date_created': pd.Timestamp.now().isoformat(),
        'baselines': baselines,  # Rolling averages
        'best_models': {  # Ridge results (baseline ML models)
            target: {
                'model': 'Ridge',
                'val_mae': ridge_results[target]['val_mae'],
                'val_r2': ridge_results[target]['val_r2']
            }
            for target in ['PTS', 'REB', 'AST']
        }
    }

    baseline_path = Path('results/baseline_models_results.json')
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, 'w') as f:
        json.dump(baseline_output, f, indent=2)

    logger.info(f"  ✓ {baseline_path}")

    # Save advanced models results (XGBoost)
    # Structure matches generate_figures.py expectations
    advanced_output = {
        'date_created': pd.Timestamp.now().isoformat(),
        'best_single_models': {  # XGBoost results (advanced models)
            target: {
                'model': 'XGBoost',
                'val_mae': xgb_results[target]['val_mae'],
                'val_r2': xgb_results[target]['val_r2']
            }
            for target in ['PTS', 'REB', 'AST']
        },
        'all_results': results_summary  # Keep full results for reference
    }

    results_path = Path('results/advanced_models_results.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(advanced_output, f, indent=2)

    logger.info(f"  ✓ {results_path}")


def train_all_models(
    data_dir: str = 'data/processed',
    output_dir: str = 'results/models',
    quick_mode: bool = True
) -> dict:
    """
    Full model training pipeline.

    Replicates Notebooks 04-05.

    Args:
        data_dir: Directory with processed data
        output_dir: Directory to save models
        quick_mode: Use pre-tuned hyperparameters (fast) vs search (slow)

    Returns:
        Dict with model paths and results
    """
    logger.info("="*70)
    logger.info("MODEL TRAINING PIPELINE")
    logger.info("="*70)

    # Load data
    train, val, feature_names = load_data(data_dir)

    # Prepare features and targets
    X_train = train[feature_names]
    X_val = val[feature_names]

    y_train = {
        'PTS': train['PTS'],
        'REB': train['REB'],
        'AST': train['AST']
    }

    y_val = {
        'PTS': val['PTS'],
        'REB': val['REB'],
        'AST': val['AST']
    }

    # Calculate baseline
    baselines = calculate_baseline_performance(val)

    # Train Ridge models
    ridge_models, ridge_results = train_ridge_models(X_train, y_train, X_val, y_val)

    # Train XGBoost models
    xgb_models, xgb_results = train_xgboost_models(
        X_train, y_train, X_val, y_val, quick_mode=quick_mode
    )

    # Save models
    save_models(ridge_models, xgb_models, ridge_results, xgb_results, baselines, output_dir)

    logger.info("\n" + "="*70)
    logger.info("✅ MODEL TRAINING COMPLETE")
    logger.info("="*70)

    # Print summary
    print("\n" + "="*70)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*70)

    print("\n| Target | Baseline MAE | Ridge MAE | XGBoost MAE | Best Model |")
    print("|--------|--------------|-----------|-------------|------------|")

    for target in ['PTS', 'REB', 'AST']:
        baseline_mae = baselines[target]['mae']
        ridge_mae = ridge_results[target]['val_mae']
        xgb_mae = xgb_results[target]['val_mae']
        best = "Ridge" if ridge_mae <= xgb_mae else "XGBoost"

        print(f"| {target:6s} | {baseline_mae:12.3f} | {ridge_mae:9.3f} | {xgb_mae:11.3f} | {best:10s} |")

    print("\n" + "="*70)

    return {
        'ridge_models': ridge_models,
        'xgb_models': xgb_models,
        'ridge_results': ridge_results,
        'xgb_results': xgb_results,
        'baselines': baselines
    }


def main():
    """Run model training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Train NBA prediction models')
    parser.add_argument('--full-search', action='store_true',
                       help='Run full hyperparameter search (slow)')
    args = parser.parse_args()

    quick_mode = not args.full_search

    if not quick_mode:
        logger.warning("Running FULL hyperparameter search (30+ minutes)...")
        input("Press Enter to continue or Ctrl+C to cancel...")

    train_all_models(quick_mode=quick_mode)


if __name__ == '__main__':
    main()
