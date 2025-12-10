"""
Prediction script - Make predictions with trained models.

Usage:
    python -m src.predict --player "LeBron James" --opponent "BOS"
    OR
    make predict  (interactive mode)
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json


def load_models():
    """Load trained Ridge regression models."""
    models_dir = Path('results/models')

    with open(models_dir / 'best_ridge_pts.pkl', 'rb') as f:
        model_pts = pickle.load(f)
    with open(models_dir / 'best_ridge_reb.pkl', 'rb') as f:
        model_reb = pickle.load(f)
    with open(models_dir / 'best_ridge_ast.pkl', 'rb') as f:
        model_ast = pickle.load(f)

    return model_pts, model_reb, model_ast


def load_feature_metadata():
    """Load feature names and metadata."""
    with open('data/processed/feature_metadata_v2.json', 'r') as f:
        metadata = json.load(f)
    return metadata['feature_names']


def get_player_features(player_name, opponent_abbrev, is_home=True, days_rest=2):
    """
    Get feature vector for a player vs specific opponent.

    This is a simplified version - in production, would fetch real-time data.
    For demo, loads historical averages from processed data.
    """
    # Load processed data to get player's recent stats
    df = pd.read_parquet('data/processed/features_engineered.parquet')

    # Filter to this player's most recent games
    player_data = df[df['PLAYER_NAME'] == player_name].sort_values('GAME_DATE', ascending=False)

    if len(player_data) == 0:
        raise ValueError(f"Player '{player_name}' not found in dataset")

    # Use most recent game's features as baseline
    latest_features = player_data.iloc[0]

    feature_names = load_feature_metadata()
    features = latest_features[feature_names].values.astype(float)

    return features, player_data.iloc[0]


def predict(player_name, opponent_abbrev, is_home=True, days_rest=2):
    """Make prediction for a player vs opponent."""
    print(f"\n{'='*60}")
    print(f"NBA PLAYER PREDICTION")
    print(f"{'='*60}")
    print(f"\nPlayer: {player_name}")
    print(f"Opponent: {opponent_abbrev}")
    print(f"Location: {'Home' if is_home else 'Away'}")
    print(f"Days Rest: {days_rest}")

    # Load models
    model_pts, model_reb, model_ast = load_models()

    # Get features
    features, player_info = get_player_features(player_name, opponent_abbrev, is_home, days_rest)
    features = features.reshape(1, -1)

    # Make predictions
    pred_pts = model_pts.predict(features)[0]
    pred_reb = model_reb.predict(features)[0]
    pred_ast = model_ast.predict(features)[0]

    # Show results
    print(f"\n{'='*60}")
    print(f"PREDICTED STATS")
    print(f"{'='*60}")
    print(f"\nPTS: {pred_pts:.1f}")
    print(f"REB: {pred_reb:.1f}")
    print(f"AST: {pred_ast:.1f}")

    # Show season averages for comparison
    print(f"\n{'='*60}")
    print(f"SEASON AVERAGES (for comparison)")
    print(f"{'='*60}")
    print(f"\nPTS: {player_info['PTS_SEASON_AVG']:.1f}")
    print(f"REB: {player_info['REB_SEASON_AVG']:.1f}")
    print(f"AST: {player_info['AST_SEASON_AVG']:.1f}")

    return {
        'predictions': {
            'PTS': float(pred_pts),
            'REB': float(pred_reb),
            'AST': float(pred_ast)
        },
        'season_averages': {
            'PTS': float(player_info['PTS_SEASON_AVG']),
            'REB': float(player_info['REB_SEASON_AVG']),
            'AST': float(player_info['AST_SEASON_AVG'])
        }
    }


def interactive_mode():
    """Interactive prediction mode."""
    print("\n" + "="*60)
    print("NBA PLAYER PREDICTION - INTERACTIVE MODE")
    print("="*60)

    # Load data to show available players
    df = pd.read_parquet('data/processed/features_engineered.parquet')
    players = df['PLAYER_NAME'].unique()

    print(f"\nAvailable players: {len(players)}")
    print("\nExample players:")
    for player in sorted(players)[:10]:
        print(f"  - {player}")

    # Get user input
    player_name = input("\nEnter player name: ").strip()

    if player_name not in players:
        print(f"\nPlayer '{player_name}' not found. Using 'LeBron James' as example.")
        player_name = "LeBron James"

    opponent = input("Enter opponent team abbreviation (e.g., BOS, LAL): ").strip().upper()

    # Make prediction
    result = predict(player_name, opponent, is_home=True, days_rest=2)

    print(f"\n{'='*60}\n")
    return result


def main():
    parser = argparse.ArgumentParser(description='NBA Player Prediction')
    parser.add_argument('--player', type=str, help='Player name')
    parser.add_argument('--opponent', type=str, help='Opponent team abbreviation')
    parser.add_argument('--home', action='store_true', help='Home game (default: True)')
    parser.add_argument('--away', action='store_true', help='Away game')
    parser.add_argument('--rest', type=int, default=2, help='Days rest (default: 2)')

    args = parser.parse_args()

    if args.player and args.opponent:
        # Command-line mode
        is_home = not args.away
        predict(args.player, args.opponent, is_home, args.rest)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == '__main__':
    main()
