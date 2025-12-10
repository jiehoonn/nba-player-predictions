"""
Tests for Model Evaluation Module.

Tests high-level functionality of evaluation pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import analyze_errors


@pytest.fixture
def sample_test_results():
    """Create sample test data with predictions."""
    np.random.seed(42)

    n_samples = 500

    # Create test set
    test = pd.DataFrame({
        'PLAYER_ID': np.random.randint(0, 50, n_samples),
        'PLAYER_NAME': [f'Player_{i%50}' for i in range(n_samples)],
        'GAME_DATE': pd.date_range('2024-01-01', periods=n_samples, freq='D'),
        'GAME_ID': [f'GAME_{i}' for i in range(n_samples)],
        'MATCHUP': ['vs. BOS' if i % 2 == 0 else '@ LAL' for i in range(n_samples)],
        'PTS': np.random.uniform(5, 35, n_samples),
        'REB': np.random.uniform(2, 15, n_samples),
        'AST': np.random.uniform(1, 12, n_samples),
        'PTS_SEASON_AVG': np.random.uniform(10, 30, n_samples),
        'IS_HOME': np.random.choice([0, 1], n_samples)
    })

    # Create predictions (with some error)
    predictions = {
        'PTS': np.random.uniform(5, 35, n_samples),
        'REB': np.random.uniform(2, 15, n_samples),
        'AST': np.random.uniform(1, 12, n_samples)
    }

    results = {
        'PTS': {'test_predictions': predictions['PTS']},
        'REB': {'test_predictions': predictions['REB']},
        'AST': {'test_predictions': predictions['AST']}
    }

    return test, results


def test_error_analysis_structure(sample_test_results):
    """Test that error analysis returns correct structure."""
    test, results = sample_test_results

    error_analysis = analyze_errors(test, results)

    # Check required keys
    assert 'tier_errors' in error_analysis
    assert 'worst_threshold' in error_analysis
    assert 'worst_count' in error_analysis
    assert 'test_predictions' in error_analysis

    # Check worst_threshold is a number
    assert isinstance(error_analysis['worst_threshold'], (int, float))
    assert error_analysis['worst_threshold'] > 0

    # Check worst_count is integer
    assert isinstance(error_analysis['worst_count'], int)
    assert error_analysis['worst_count'] >= 0


def test_error_analysis_predictions_added(sample_test_results):
    """Test that predictions and errors are added to test set."""
    test, results = sample_test_results

    error_analysis = analyze_errors(test, results)
    test_with_preds = error_analysis['test_predictions']

    # Check prediction columns added
    assert 'PTS_PRED' in test_with_preds.columns
    assert 'REB_PRED' in test_with_preds.columns
    assert 'AST_PRED' in test_with_preds.columns

    # Check error columns added
    assert 'PTS_ERROR' in test_with_preds.columns
    assert 'REB_ERROR' in test_with_preds.columns
    assert 'AST_ERROR' in test_with_preds.columns


def test_error_values_non_negative(sample_test_results):
    """Test that error values are non-negative (absolute errors)."""
    test, results = sample_test_results

    error_analysis = analyze_errors(test, results)
    test_with_preds = error_analysis['test_predictions']

    # All errors should be >= 0
    assert (test_with_preds['PTS_ERROR'] >= 0).all(), "PTS_ERROR should be non-negative"
    assert (test_with_preds['REB_ERROR'] >= 0).all(), "REB_ERROR should be non-negative"
    assert (test_with_preds['AST_ERROR'] >= 0).all(), "AST_ERROR should be non-negative"


def test_player_tier_classification(sample_test_results):
    """Test that players are classified into tiers correctly."""
    test, results = sample_test_results

    error_analysis = analyze_errors(test, results)
    test_with_preds = error_analysis['test_predictions']

    # Check SCORING_TIER column exists
    assert 'SCORING_TIER' in test_with_preds.columns

    # Check tier categories
    expected_tiers = {'Bench (0-8)', 'Role (8-15)', 'Starter (15-22)', 'Star (22+)'}
    actual_tiers = set(test_with_preds['SCORING_TIER'].dropna().unique())

    # At least some tiers should exist (maybe not all if small sample)
    assert len(actual_tiers) > 0, "Should have at least one tier"
    assert actual_tiers.issubset(expected_tiers), "Tiers should match expected categories"


def test_worst_predictions_threshold(sample_test_results):
    """Test that worst predictions threshold is 95th percentile."""
    test, results = sample_test_results

    error_analysis = analyze_errors(test, results)
    test_with_preds = error_analysis['test_predictions']

    # Calculate expected threshold
    expected_threshold = test_with_preds['PTS_ERROR'].quantile(0.95)

    # Should match (within floating point tolerance)
    assert abs(error_analysis['worst_threshold'] - expected_threshold) < 0.01


def test_worst_predictions_count(sample_test_results):
    """Test that worst predictions count is approximately 5%."""
    test, results = sample_test_results

    error_analysis = analyze_errors(test, results)
    test_with_preds = error_analysis['test_predictions']

    total_games = len(test_with_preds)
    worst_count = error_analysis['worst_count']

    # Should be approximately 5% (within 1%)
    worst_pct = worst_count / total_games * 100
    assert 4 <= worst_pct <= 6, f"Worst predictions should be ~5%, got {worst_pct:.1f}%"


def test_error_calculation_correct(sample_test_results):
    """Test that errors are calculated correctly (abs(actual - predicted))."""
    test, results = sample_test_results

    error_analysis = analyze_errors(test, results)
    test_with_preds = error_analysis['test_predictions']

    # Check a few random samples
    for i in range(min(10, len(test_with_preds))):
        row = test_with_preds.iloc[i]

        expected_pts_error = abs(row['PTS'] - row['PTS_PRED'])
        actual_pts_error = row['PTS_ERROR']

        assert abs(expected_pts_error - actual_pts_error) < 0.001, \
            f"PTS_ERROR calculation incorrect at row {i}"


def test_results_json_serializable(sample_test_results):
    """Test that error analysis results can be serialized to JSON."""
    import json

    test, results = sample_test_results

    error_analysis = analyze_errors(test, results)

    # Remove DataFrame (can't serialize)
    error_summary = {
        'worst_threshold': float(error_analysis['worst_threshold']),
        'worst_count': int(error_analysis['worst_count'])
    }

    # Should not raise exception
    try:
        json_str = json.dumps(error_summary)
        assert len(json_str) > 0
    except Exception as e:
        pytest.fail(f"Error analysis results not JSON serializable: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
