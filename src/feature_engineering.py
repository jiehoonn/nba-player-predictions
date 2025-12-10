"""
Feature Engineering Module - Replicates Notebook 03

Transforms raw game logs into ML-ready features with temporal awareness.

Usage:
    python -m src.feature_engineering
    OR
    from src.feature_engineering import engineer_features
    engineer_features('data/processed/gamelogs_combined.parquet')
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Tuple, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_rolling_features(df: pd.DataFrame, windows: list = [3, 5, 10]) -> pd.DataFrame:
    """
    Calculate rolling averages for PTS, REB, AST.

    CRITICAL: Uses .shift(1) to prevent data leakage (current game excluded).

    Args:
        df: DataFrame with game logs (must be sorted by PLAYER_ID, GAME_DATE)
        windows: List of window sizes

    Returns:
        DataFrame with rolling average columns
    """
    logger.info(f"Calculating rolling averages (windows: {windows})...")

    stats = ['PTS', 'REB', 'AST']

    for stat in stats:
        for window in windows:
            col_name = f'{stat}_LAST_{window}'
            # CRITICAL: shift(1) excludes current game
            df[col_name] = (df.groupby('PLAYER_ID')[stat]
                           .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean()))

    logger.info(f"  ✓ Created {len(stats) * len(windows)} rolling features")
    return df


def calculate_season_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate season-level statistics and context.

    Features:
    - PTS_SEASON_AVG, REB_SEASON_AVG, AST_SEASON_AVG (expanding mean)
    - SEASON_GAME_NUM (game number in season, 1-indexed)
    - MONTH (1-12)
    - DAY_OF_WEEK (0=Monday, 6=Sunday)
    """
    logger.info("Calculating season context features...")

    # Season averages (expanding mean with shift to prevent leakage)
    for stat in ['PTS', 'REB', 'AST']:
        df[f'{stat}_SEASON_AVG'] = (df.groupby(['PLAYER_ID', 'SEASON'])[stat]
                                     .transform(lambda x: x.shift(1).expanding().mean()))

    # Season game number (1-indexed, like notebook 03)
    df['SEASON_GAME_NUM'] = df.groupby(['PLAYER_ID', 'SEASON']).cumcount() + 1

    # Month and day of week
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['MONTH'] = df['GAME_DATE'].dt.month
    df['DAY_OF_WEEK'] = df['GAME_DATE'].dt.dayofweek  # 0=Monday, 6=Sunday

    logger.info("  ✓ Created 7 season context features")
    return df


def calculate_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate opponent-specific features.

    Features:
    - OPP_DEF_RATING (opponent defensive rating)
    - OPP_PACE (opponent pace)
    - OPP_W_PCT (opponent win percentage)
    - OPP_OFF_RATING (opponent offensive rating)
    """
    logger.info("Calculating opponent features...")

    # These features come from team stats (already in dataset)
    opp_features = ['OPP_DEF_RATING', 'OPP_PACE', 'OPP_W_PCT', 'OPP_OFF_RATING']

    for feat in opp_features:
        if feat not in df.columns:
            logger.warning(f"  ⚠️  {feat} not found in data")

    logger.info(f"  ✓ Using {len(opp_features)} opponent features")
    return df


def calculate_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate team-specific features.

    Features:
    - TEAM_PACE, TEAM_OFF_RATING, TEAM_DEF_RATING, TEAM_W_PCT
    """
    logger.info("Calculating team features...")

    team_features = ['TEAM_PACE', 'TEAM_OFF_RATING', 'TEAM_DEF_RATING', 'TEAM_W_PCT']

    for feat in team_features:
        if feat not in df.columns:
            logger.warning(f"  ⚠️  {feat} not found in data")

    logger.info(f"  ✓ Using {len(team_features)} team features")
    return df


def calculate_game_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate game context features.

    Features:
    - IS_HOME (1 if home, 0 if away)
    - REST_0_1, REST_2_3, REST_4_PLUS (binned rest days)
    """
    logger.info("Calculating game context features...")

    # Home/Away
    df['IS_HOME'] = df['MATCHUP'].str.contains('vs.').astype(int)

    # Rest days binning (avoid back-to-back paradox)
    df['REST_0_1'] = (df['DAYS_REST'].between(0, 1)).astype(int)
    df['REST_2_3'] = (df['DAYS_REST'].between(2, 3)).astype(int)
    df['REST_4_PLUS'] = (df['DAYS_REST'] >= 4).astype(int)

    logger.info("  ✓ Created 4 game context features")
    return df


def calculate_shot_tendencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate shot zone tendencies (from shot chart data).

    Features:
    - PAINT_PCT, MIDRANGE_PCT, THREE_PT_PCT, RESTRICTED_AREA_PCT

    NOTE: If shot chart data not available, uses league average defaults
    (same approach as Notebook 03 for missing data).
    """
    logger.info("Calculating shot tendency features...")

    # Default values (league averages, from Notebook 03)
    defaults = {
        'RESTRICTED_AREA_PCT': 0.25,
        'PAINT_PCT': 0.15,
        'MIDRANGE_PCT': 0.20,
        'THREE_PT_PCT': 0.40
    }

    shot_features = []
    for feat, default_val in defaults.items():
        if feat not in df.columns:
            logger.info(f"  Creating {feat} with default value ({default_val})")
            df[feat] = default_val
        shot_features.append(feat)

    logger.info(f"  ✓ Created {len(shot_features)} shot tendency features")
    return df


def calculate_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate momentum/trend features.

    Features:
    - PTS_TREND, REB_TREND, AST_TREND (last 3 vs last 10 difference)
    - PTS_VOLATILITY, REB_VOLATILITY, AST_VOLATILITY (rolling std dev)
    """
    logger.info("Calculating momentum features...")

    for stat in ['PTS', 'REB', 'AST']:
        # Trend = recent form (last 3) minus longer average (last 10)
        df[f'{stat}_TREND'] = df[f'{stat}_LAST_3'] - df[f'{stat}_LAST_10']

        # Volatility = rolling standard deviation over last 5 games
        df[f'{stat}_VOLATILITY'] = (df.groupby('PLAYER_ID')[stat]
                                     .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std()))

    logger.info("  ✓ Created 6 momentum features")
    return df


def create_train_val_test_splits(
    df: pd.DataFrame,
    train_end: str = '2023-01-01',
    val_end: str = '2024-01-01'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create temporal train/validation/test splits.

    CRITICAL: Never shuffle! This is time series data.

    Args:
        df: DataFrame with engineered features
        train_end: End date for training (exclusive)
        val_end: End date for validation (exclusive)

    Returns:
        Tuple of (train, val, test)
    """
    logger.info("Creating temporal splits...")

    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    train = df[df['GAME_DATE'] < train_end].copy()
    val = df[(df['GAME_DATE'] >= train_end) & (df['GAME_DATE'] < val_end)].copy()
    test = df[df['GAME_DATE'] >= val_end].copy()

    logger.info(f"  Train: {len(train):,} games ({train['GAME_DATE'].min().date()} to {train['GAME_DATE'].max().date()})")
    logger.info(f"  Val:   {len(val):,} games ({val['GAME_DATE'].min().date()} to {val['GAME_DATE'].max().date()})")
    logger.info(f"  Test:  {len(test):,} games ({test['GAME_DATE'].min().date()} to {test['GAME_DATE'].max().date()})")

    return train, val, test


def engineer_features(
    input_path: str = 'data/processed/gamelogs_combined.parquet',
    output_dir: str = 'data/processed'
) -> Dict[str, any]:
    """
    Full feature engineering pipeline.

    Replicates Notebook 03 logic exactly.

    Args:
        input_path: Path to combined game logs
        output_dir: Output directory for processed data

    Returns:
        Dict with paths to output files and metadata
    """
    logger.info("="*70)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("="*70)

    # Load data
    logger.info(f"\nLoading data from {input_path}...")
    df = pd.read_parquet(input_path)
    logger.info(f"  Loaded {len(df):,} games")

    # Sort by player and date (critical for rolling features)
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)

    # Calculate features
    df = calculate_rolling_features(df)
    df = calculate_season_context(df)
    df = calculate_opponent_features(df)
    df = calculate_team_features(df)
    df = calculate_game_context(df)
    df = calculate_shot_tendencies(df)
    df = calculate_momentum(df)

    # Define feature list (38 features in exact order models expect)
    feature_names = [
        # Rolling averages (9)
        'PTS_LAST_3', 'PTS_LAST_5', 'PTS_LAST_10',
        'REB_LAST_3', 'REB_LAST_5', 'REB_LAST_10',
        'AST_LAST_3', 'AST_LAST_5', 'AST_LAST_10',

        # Season context (6)
        'PTS_SEASON_AVG', 'REB_SEASON_AVG', 'AST_SEASON_AVG',
        'SEASON_GAME_NUM', 'MONTH', 'DAY_OF_WEEK',

        # Opponent (4)
        'OPP_DEF_RATING', 'OPP_PACE', 'OPP_W_PCT', 'OPP_OFF_RATING',

        # Team (4) - order matters!
        'TEAM_DEF_RATING', 'TEAM_PACE', 'TEAM_W_PCT', 'TEAM_OFF_RATING',

        # Game context (5) - order matters!
        'IS_HOME', 'DAYS_REST', 'REST_0_1', 'REST_2_3', 'REST_4_PLUS',

        # Shot tendencies (4) - order matters!
        'RESTRICTED_AREA_PCT', 'PAINT_PCT', 'MIDRANGE_PCT', 'THREE_PT_PCT',

        # Momentum (6)
        'PTS_TREND', 'REB_TREND', 'AST_TREND',
        'PTS_VOLATILITY', 'REB_VOLATILITY', 'AST_VOLATILITY'
    ]

    # Verify all features exist
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        logger.warning(f"Missing features: {missing}")
        feature_names = [f for f in feature_names if f in df.columns]

    logger.info(f"\n✓ Total features engineered: {len(feature_names)}")

    # Drop rows with NaN in features (first few games per player)
    initial_len = len(df)
    df = df.dropna(subset=feature_names + ['PTS', 'REB', 'AST'])
    dropped = initial_len - len(df)
    logger.info(f"  Dropped {dropped:,} rows with NaN ({dropped/initial_len*100:.1f}%)")

    # Create temporal splits
    train, val, test = create_train_val_test_splits(df)

    # Save processed data
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\nSaving processed data...")
    train.to_parquet(output_dir / 'train.parquet', index=False)
    val.to_parquet(output_dir / 'val.parquet', index=False)
    test.to_parquet(output_dir / 'test.parquet', index=False)
    df.to_parquet(output_dir / 'features_engineered.parquet', index=False)

    # Save metadata
    metadata = {
        'feature_names': feature_names,
        'target_columns': ['PTS', 'REB', 'AST'],
        'tracking_columns': ['GAME_ID', 'PLAYER_ID', 'PLAYER_NAME', 'GAME_DATE', 'MATCHUP'],
        'feature_breakdown': {
            'rolling_averages': 9,
            'season_context': 6,
            'opponent_context': 4,
            'team_context': 4,
            'game_context': 5,
            'shot_tendencies': 4,
            'momentum': 6
        },
        'total_features': len(feature_names),
        'train_size': len(train),
        'val_size': len(val),
        'test_size': len(test),
        'train_date_range': f"{train['GAME_DATE'].min().date()} to {train['GAME_DATE'].max().date()}",
        'val_date_range': f"{val['GAME_DATE'].min().date()} to {val['GAME_DATE'].max().date()}",
        'test_date_range': f"{test['GAME_DATE'].min().date()} to {test['GAME_DATE'].max().date()}"
    }

    with open(output_dir / 'feature_metadata_v2.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"  ✓ train.parquet: {len(train):,} games")
    logger.info(f"  ✓ val.parquet: {len(val):,} games")
    logger.info(f"  ✓ test.parquet: {len(test):,} games")
    logger.info(f"  ✓ feature_metadata_v2.json")

    logger.info("\n" + "="*70)
    logger.info("✅ FEATURE ENGINEERING COMPLETE")
    logger.info("="*70)

    return {
        'feature_names': feature_names,
        'metadata': metadata,
        'train_path': str(output_dir / 'train.parquet'),
        'val_path': str(output_dir / 'val.parquet'),
        'test_path': str(output_dir / 'test.parquet')
    }


def main():
    """Run feature engineering pipeline."""
    result = engineer_features()

    print("\n" + "="*70)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*70)
    print(f"\nTotal features: {result['metadata']['total_features']}")
    print(f"\nFeature breakdown:")
    for category, count in result['metadata']['feature_breakdown'].items():
        print(f"  {category.replace('_', ' ').title():.<30} {count}")

    print(f"\nDataset splits:")
    print(f"  Train: {result['metadata']['train_size']:,} games")
    print(f"  Val:   {result['metadata']['val_size']:,} games")
    print(f"  Test:  {result['metadata']['test_size']:,} games")

    print(f"\nOutput files:")
    print(f"  {result['train_path']}")
    print(f"  {result['val_path']}")
    print(f"  {result['test_path']}")


if __name__ == '__main__':
    main()
