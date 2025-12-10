"""
End-to-End Integration Tests.

Tests the complete pipeline from feature engineering → training → evaluation.

These are HIGH-LEVEL tests that validate the entire workflow works together.
Perfect for others extending this project to ensure they don't break the pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# Skip tests if data doesn't exist (CI/CD won't have data)
DATA_EXISTS = Path('data/processed/gamelogs_combined.parquet').exists()
FEATURES_EXIST = Path('data/processed/train.parquet').exists()
MODELS_EXIST = Path('results/models/best_ridge_pts.pkl').exists()

skip_no_data = pytest.mark.skipif(
    not DATA_EXISTS,
    reason="Raw data not found (run data collection first)"
)

skip_no_features = pytest.mark.skipif(
    not FEATURES_EXIST,
    reason="Features not found (run feature engineering first)"
)

skip_no_models = pytest.mark.skipif(
    not MODELS_EXIST,
    reason="Models not found (run training first)"
)


@skip_no_features
def test_pipeline_temporal_consistency():
    """
    CRITICAL: Test that entire pipeline maintains temporal consistency.

    No future data should leak into past predictions.
    """
    # Load all splits
    train = pd.read_parquet('data/processed/train.parquet')
    val = pd.read_parquet('data/processed/val.parquet')
    test = pd.read_parquet('data/processed/test.parquet')

    # Convert dates
    train['GAME_DATE'] = pd.to_datetime(train['GAME_DATE'])
    val['GAME_DATE'] = pd.to_datetime(val['GAME_DATE'])
    test['GAME_DATE'] = pd.to_datetime(test['GAME_DATE'])

    # Check temporal ordering
    assert train['GAME_DATE'].max() < val['GAME_DATE'].min(), \
        "CRITICAL: Train data leaks into validation set!"

    assert val['GAME_DATE'].max() < test['GAME_DATE'].min(), \
        "CRITICAL: Validation data leaks into test set!"

    print(f"✓ Temporal consistency verified:")
    print(f"  Train: {train['GAME_DATE'].min().date()} to {train['GAME_DATE'].max().date()}")
    print(f"  Val:   {val['GAME_DATE'].min().date()} to {val['GAME_DATE'].max().date()}")
    print(f"  Test:  {test['GAME_DATE'].min().date()} to {test['GAME_DATE'].max().date()}")


@skip_no_features
def test_pipeline_feature_consistency():
    """Test that all splits have same features."""
    train = pd.read_parquet('data/processed/train.parquet')
    val = pd.read_parquet('data/processed/val.parquet')
    test = pd.read_parquet('data/processed/test.parquet')

    # Load feature metadata
    with open('data/processed/feature_metadata_v2.json', 'r') as f:
        metadata = json.load(f)

    feature_names = metadata['feature_names']

    # Check all features exist in all splits
    for split_name, split_df in [('train', train), ('val', val), ('test', test)]:
        missing_features = [f for f in feature_names if f not in split_df.columns]
        assert len(missing_features) == 0, \
            f"{split_name} missing features: {missing_features}"

    print(f"✓ All {len(feature_names)} features present in all splits")


@skip_no_features
def test_pipeline_no_missing_values():
    """Test that processed splits have no missing values in features."""
    train = pd.read_parquet('data/processed/train.parquet')
    val = pd.read_parquet('data/processed/val.parquet')
    test = pd.read_parquet('data/processed/test.parquet')

    with open('data/processed/feature_metadata_v2.json', 'r') as f:
        metadata = json.load(f)

    feature_names = metadata['feature_names']
    targets = metadata['target_columns']

    for split_name, split_df in [('train', train), ('val', val), ('test', test)]:
        # Check features
        feature_missing = split_df[feature_names].isnull().sum().sum()
        assert feature_missing == 0, \
            f"{split_name} has {feature_missing} missing values in features"

        # Check targets
        target_missing = split_df[targets].isnull().sum().sum()
        assert target_missing == 0, \
            f"{split_name} has {target_missing} missing values in targets"

    print("✓ No missing values in any split")


@skip_no_models
@skip_no_features
def test_pipeline_models_load_and_predict():
    """Test that trained models can be loaded and make predictions."""
    # Load validation data
    val = pd.read_parquet('data/processed/val.parquet')

    with open('data/processed/feature_metadata_v2.json', 'r') as f:
        metadata = json.load(f)

    feature_names = metadata['feature_names']
    X_val = val[feature_names]

    # Load and test each model
    for target in ['PTS', 'REB', 'AST']:
        model_path = Path(f'results/models/best_ridge_{target.lower()}.pkl')

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Make predictions
        predictions = model.predict(X_val)

        # Check output shape
        assert len(predictions) == len(val), \
            f"{target} predictions have wrong length"

        # Check predictions are reasonable
        assert np.all(np.isfinite(predictions)), \
            f"{target} predictions contain NaN or inf"

        # Check predictions are in reasonable range
        if target == 'PTS':
            assert predictions.min() > -10, f"{target} predictions too low"
            assert predictions.max() < 100, f"{target} predictions too high"
        elif target == 'REB':
            assert predictions.min() > -5, f"{target} predictions too low"
            assert predictions.max() < 30, f"{target} predictions too high"
        elif target == 'AST':
            assert predictions.min() > -5, f"{target} predictions too low"
            assert predictions.max() < 25, f"{target} predictions too high"

    print("✓ All models load and make valid predictions")


@skip_no_models
@skip_no_features
def test_pipeline_model_performance_thresholds():
    """Test that models meet minimum performance thresholds."""
    # Load final results
    with open('results/final_test_results.json', 'r') as f:
        results = json.load(f)

    # Check performance thresholds
    performance_tests = [
        ('PTS', 'test_mae', 6.0, 'less'),     # MAE should be < 6.0
        ('PTS', 'test_r2', 0.45, 'greater'),  # R² should be > 0.45
        ('REB', 'test_mae', 2.5, 'less'),
        ('REB', 'test_r2', 0.40, 'greater'),
        ('AST', 'test_mae', 2.0, 'less'),
        ('AST', 'test_r2', 0.45, 'greater'),
    ]

    for target, metric, threshold, comparison in performance_tests:
        actual_value = results['results'][target][metric]

        if comparison == 'less':
            assert actual_value < threshold, \
                f"{target} {metric} = {actual_value:.3f} should be < {threshold}"
        else:  # greater
            assert actual_value > threshold, \
                f"{target} {metric} = {actual_value:.3f} should be > {threshold}"

    print("✓ All models meet performance thresholds")


@skip_no_models
@skip_no_features
def test_pipeline_generalization():
    """
    Test that models generalize well (val vs test performance).

    Degradation should be < 10% (ideally < 5%).
    """
    with open('results/final_test_results.json', 'r') as f:
        results = json.load(f)

    for target in ['PTS', 'REB', 'AST']:
        degradation = results['results'][target]['mae_degradation_pct']

        # Fail if degradation > 10%
        assert abs(degradation) < 10, \
            f"{target} degradation {degradation:+.1f}% exceeds 10% (possible overfitting)"

        # Warn if degradation > 5%
        if abs(degradation) > 5:
            print(f"  ⚠️  {target} degradation {degradation:+.1f}% > 5%")

    print("✓ Models generalize well (degradation < 10%)")


@skip_no_models
def test_pipeline_predictions_saved():
    """Test that test set predictions are saved correctly."""
    predictions_path = Path('results/test_predictions.parquet')
    assert predictions_path.exists(), "Test predictions not saved"

    # Load predictions
    preds = pd.read_parquet(predictions_path)

    # Check required columns
    required_cols = [
        'GAME_ID', 'PLAYER_ID', 'PLAYER_NAME', 'GAME_DATE',
        'PTS', 'PTS_PRED', 'PTS_ERROR',
        'REB', 'REB_PRED', 'REB_ERROR',
        'AST', 'AST_PRED', 'AST_ERROR'
    ]

    for col in required_cols:
        assert col in preds.columns, f"Missing column: {col}"

    # Check errors are non-negative
    assert (preds['PTS_ERROR'] >= 0).all(), "PTS_ERROR should be non-negative"
    assert (preds['REB_ERROR'] >= 0).all(), "REB_ERROR should be non-negative"
    assert (preds['AST_ERROR'] >= 0).all(), "AST_ERROR should be non-negative"

    print(f"✓ Test predictions saved: {len(preds):,} games")


def test_pipeline_results_json_valid():
    """Test that all results JSON files are valid."""
    json_files = [
        'results/baseline_models_results.json',
        'results/advanced_models_results.json',
        'results/final_test_results.json'
    ]

    for json_path in json_files:
        path = Path(json_path)
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                assert isinstance(data, dict), f"{json_path} should be a dict"
                print(f"  ✓ {json_path} is valid JSON")


@skip_no_features
def test_pipeline_feature_distributions_reasonable():
    """Test that engineered features have reasonable distributions."""
    train = pd.read_parquet('data/processed/train.parquet')

    with open('data/processed/feature_metadata_v2.json', 'r') as f:
        metadata = json.load(f)

    feature_names = metadata['feature_names']

    # Check each feature
    issues = []
    for feature in feature_names:
        if feature in train.columns:
            values = train[feature].dropna()

            # Check for constant features (no variance)
            if values.std() < 0.001:
                issues.append(f"{feature} is nearly constant (std={values.std():.4f})")

            # Check for extreme outliers (>10 std devs from mean)
            z_scores = np.abs((values - values.mean()) / values.std())
            extreme_outliers = (z_scores > 10).sum()
            if extreme_outliers > len(values) * 0.01:  # >1% outliers
                issues.append(f"{feature} has {extreme_outliers} extreme outliers")

    if issues:
        print("  ⚠️  Feature distribution warnings:")
        for issue in issues:
            print(f"    - {issue}")

    # Don't fail on warnings, just report
    print("✓ Feature distributions checked")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
