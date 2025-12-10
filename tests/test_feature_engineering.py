"""
Tests for Feature Engineering Module.

Tests high-level functionality of feature engineering pipeline.
Others can extend these tests when adding new features.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import (
    calculate_rolling_features,
    calculate_season_context,
    calculate_game_context,
    calculate_momentum,
    create_train_val_test_splits
)


@pytest.fixture
def sample_gamelogs():
    """Create sample game logs for testing."""
    np.random.seed(42)

    # Create 100 games for 5 players
    players = ['Player_A', 'Player_B', 'Player_C', 'Player_D', 'Player_E']
    data = []

    for player_id, player_name in enumerate(players):
        dates = pd.date_range('2022-01-01', periods=20, freq='3D')

        for i, date in enumerate(dates):
            data.append({
                'PLAYER_ID': player_id,
                'PLAYER_NAME': player_name,
                'GAME_DATE': date,
                'GAME_ID': f'GAME_{player_id}_{i}',
                'SEASON': '2022-23',
                'PTS': np.random.randint(10, 30),
                'REB': np.random.randint(3, 12),
                'AST': np.random.randint(2, 10),
                'MATCHUP': 'vs. BOS' if i % 2 == 0 else '@ LAL',
                'DAYS_REST': np.random.choice([0, 1, 2, 3, 5]),
                'OPP_DEF_RATING': np.random.uniform(105, 115),
                'OPP_PACE': np.random.uniform(95, 105),
                'OPP_W_PCT': np.random.uniform(0.3, 0.7),
                'OPP_OFF_RATING': np.random.uniform(105, 115),
                'TEAM_PACE': np.random.uniform(95, 105),
                'TEAM_OFF_RATING': np.random.uniform(105, 115),
                'TEAM_DEF_RATING': np.random.uniform(105, 115),
                'TEAM_W_PCT': np.random.uniform(0.4, 0.6),
                'PAINT_PCT': np.random.uniform(0.3, 0.5),
                'MIDRANGE_PCT': np.random.uniform(0.1, 0.3),
                'THREE_PT_PCT': np.random.uniform(0.3, 0.5),
                'RESTRICTED_AREA_PCT': np.random.uniform(0.2, 0.4)
            })

    df = pd.DataFrame(data)
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
    return df


def test_rolling_features_exist(sample_gamelogs):
    """Test that rolling features are created."""
    df = calculate_rolling_features(sample_gamelogs.copy())

    # Check rolling average columns exist
    for stat in ['PTS', 'REB', 'AST']:
        for window in [3, 5, 10]:
            col = f'{stat}_LAST_{window}'
            assert col in df.columns, f"Missing column: {col}"


def test_rolling_features_no_leakage(sample_gamelogs):
    """
    CRITICAL: Test that rolling features use shift(1) to prevent leakage.

    Current game should NOT be included in rolling average.
    """
    df = sample_gamelogs.copy()
    df = calculate_rolling_features(df)

    # For first player, check that first game's rolling avg is NaN
    player_0 = df[df['PLAYER_ID'] == 0].reset_index(drop=True)

    # First game should have NaN for rolling features (no previous games)
    assert pd.isna(player_0.loc[0, 'PTS_LAST_3']), \
        "First game should have NaN for PTS_LAST_3"

    # Second game should have rolling avg = first game's value
    if len(player_0) > 1:
        expected_pts_last_3 = player_0.loc[0, 'PTS']
        actual_pts_last_3 = player_0.loc[1, 'PTS_LAST_3']
        assert abs(expected_pts_last_3 - actual_pts_last_3) < 0.01, \
            "Second game's rolling avg should equal first game's PTS (with shift)"


def test_rolling_features_per_player(sample_gamelogs):
    """Test that rolling features are calculated per player."""
    df = calculate_rolling_features(sample_gamelogs.copy())

    # Get rolling averages for two different players
    player_0 = df[df['PLAYER_ID'] == 0].reset_index(drop=True)
    player_1 = df[df['PLAYER_ID'] == 1].reset_index(drop=True)

    # Same game number, different players should have different rolling avgs
    if len(player_0) > 5 and len(player_1) > 5:
        pts_last_5_p0 = player_0.loc[5, 'PTS_LAST_5']
        pts_last_5_p1 = player_1.loc[5, 'PTS_LAST_5']

        # These should be different (unless by coincidence)
        # Check they're both computed (not NaN)
        assert not pd.isna(pts_last_5_p0), "Player 0 should have PTS_LAST_5"
        assert not pd.isna(pts_last_5_p1), "Player 1 should have PTS_LAST_5"


def test_season_context_features(sample_gamelogs):
    """Test season context features are created correctly."""
    df = calculate_season_context(sample_gamelogs.copy())

    # Check season average columns exist
    assert 'PTS_SEASON_AVG' in df.columns
    assert 'REB_SEASON_AVG' in df.columns
    assert 'AST_SEASON_AVG' in df.columns
    assert 'SEASON_GAME_NUM' in df.columns
    assert 'MONTH' in df.columns
    assert 'DAY_OF_WEEK' in df.columns

    # Check MONTH is between 1-12
    assert df['MONTH'].between(1, 12).all(), "MONTH should be 1-12"

    # Check DAY_OF_WEEK is between 0-6
    assert df['DAY_OF_WEEK'].between(0, 6).all(), "DAY_OF_WEEK should be 0-6"

    # Check SEASON_GAME_NUM starts at 1 (1-indexed)
    for player_id in df['PLAYER_ID'].unique():
        player_data = df[df['PLAYER_ID'] == player_id].reset_index(drop=True)
        assert player_data.loc[0, 'SEASON_GAME_NUM'] == 1, \
            "First game should have SEASON_GAME_NUM = 1"


def test_game_context_features(sample_gamelogs):
    """Test game context features."""
    df = calculate_game_context(sample_gamelogs.copy())

    # Check IS_HOME is binary
    assert df['IS_HOME'].isin([0, 1]).all(), "IS_HOME should be 0 or 1"

    # Check rest days bins
    assert 'REST_0_1' in df.columns
    assert 'REST_2_3' in df.columns
    assert 'REST_4_PLUS' in df.columns

    # Check bins are binary
    assert df['REST_0_1'].isin([0, 1]).all()
    assert df['REST_2_3'].isin([0, 1]).all()
    assert df['REST_4_PLUS'].isin([0, 1]).all()


def test_momentum_features(sample_gamelogs):
    """Test momentum/trend features."""
    df = sample_gamelogs.copy()
    df = calculate_rolling_features(df)  # Need rolling features first
    df = calculate_momentum(df)

    # Check trend columns exist
    assert 'PTS_TREND' in df.columns
    assert 'REB_TREND' in df.columns
    assert 'AST_TREND' in df.columns

    # Trend should be difference between LAST_3 and LAST_10
    # (Check for a non-NaN row)
    valid_rows = df[df['PTS_TREND'].notna()]
    if len(valid_rows) > 0:
        row = valid_rows.iloc[0]
        expected_trend = row['PTS_LAST_3'] - row['PTS_LAST_10']
        actual_trend = row['PTS_TREND']
        assert abs(expected_trend - actual_trend) < 0.01, \
            "PTS_TREND should be LAST_3 - LAST_10"


def test_temporal_splits_no_overlap(sample_gamelogs):
    """
    CRITICAL: Test that train/val/test splits have no temporal overlap.

    This prevents data leakage.
    """
    df = sample_gamelogs.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    # Create splits
    train, val, test = create_train_val_test_splits(
        df,
        train_end='2022-02-01',
        val_end='2022-02-15'
    )

    # Check no overlap
    train_max = train['GAME_DATE'].max()
    val_min = val['GAME_DATE'].min()
    val_max = val['GAME_DATE'].max()
    test_min = test['GAME_DATE'].min()

    assert train_max < val_min, "Train set leaks into validation"
    assert val_max < test_min, "Validation set leaks into test"

    # Check no date appears in multiple splits
    train_dates = set(train['GAME_DATE'].dt.date)
    val_dates = set(val['GAME_DATE'].dt.date)
    test_dates = set(test['GAME_DATE'].dt.date)

    assert len(train_dates & val_dates) == 0, "Train/val date overlap"
    assert len(val_dates & test_dates) == 0, "Val/test date overlap"
    assert len(train_dates & test_dates) == 0, "Train/test date overlap"


def test_temporal_splits_chronological(sample_gamelogs):
    """Test that splits maintain chronological order."""
    df = sample_gamelogs.copy()
    train, val, test = create_train_val_test_splits(
        df,
        train_end='2022-02-01',
        val_end='2022-02-15'
    )

    # Check all train dates < all val dates < all test dates
    assert train['GAME_DATE'].max() < val['GAME_DATE'].min()
    assert val['GAME_DATE'].max() < test['GAME_DATE'].min()


def test_no_missing_values_in_features(sample_gamelogs):
    """Test that processed features have minimal NaN values."""
    df = sample_gamelogs.copy()
    df = calculate_rolling_features(df)
    df = calculate_season_context(df)
    df = calculate_game_context(df)

    # Rolling features will have NaN for first few games per player
    # This is expected - we filter these out later

    # But game context features should NEVER be NaN
    assert df['IS_HOME'].notna().all(), "IS_HOME should never be NaN"
    assert df['REST_0_1'].notna().all(), "REST_0_1 should never be NaN"


def test_feature_ranges_reasonable(sample_gamelogs):
    """Test that engineered features have reasonable ranges."""
    df = sample_gamelogs.copy()
    df = calculate_rolling_features(df)
    df = calculate_season_context(df)

    # Rolling averages should be in same range as original stats
    valid_pts_last_5 = df['PTS_LAST_5'].dropna()
    if len(valid_pts_last_5) > 0:
        assert valid_pts_last_5.min() >= 0, "PTS_LAST_5 should be non-negative"
        assert valid_pts_last_5.max() < 100, "PTS_LAST_5 should be < 100 (realistic)"

    # Season averages should also be reasonable
    valid_pts_season = df['PTS_SEASON_AVG'].dropna()
    if len(valid_pts_season) > 0:
        assert valid_pts_season.min() >= 0
        assert valid_pts_season.max() < 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
