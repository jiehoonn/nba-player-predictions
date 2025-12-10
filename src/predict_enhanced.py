"""
Enhanced Prediction CLI - Make matchup-specific predictions with context.

This version constructs proper feature vectors based on:
- Player's recent performance (rolling averages)
- Opponent's defensive rating
- Game context (home/away, rest days)
- Team pace and efficiency

Usage:
    python -m src.predict_enhanced --player "LeBron James" --opponent "BOS"
    python -m src.predict_enhanced  (interactive mode)
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from typing import Dict, Tuple


def load_models():
    """Load best trained models (Ridge or XGBoost)."""
    models_dir = Path('results/models')

    # Load the best models based on validation performance
    models = {}
    for target in ['pts', 'reb', 'ast']:
        # Try Ridge first (our best models)
        ridge_path = models_dir / f'best_ridge_{target}.pkl'
        if ridge_path.exists():
            with open(ridge_path, 'rb') as f:
                models[target.upper()] = pickle.load(f)

    return models


def load_feature_metadata():
    """Load feature names and metadata."""
    with open('data/processed/feature_metadata_v2.json', 'r') as f:
        metadata = json.load(f)
    return metadata['feature_names']


def get_player_recent_games(player_name: str, n_games: int = 10) -> pd.DataFrame:
    """Get player's most recent games with all features."""
    df = pd.read_parquet('data/processed/features_engineered.parquet')

    player_data = df[df['PLAYER_NAME'] == player_name].sort_values('GAME_DATE', ascending=False)

    if len(player_data) == 0:
        raise ValueError(f"Player '{player_name}' not found in dataset")

    return player_data.head(n_games)


def get_team_abbreviations() -> Dict[str, str]:
    """Get mapping of team names to abbreviations."""
    df = pd.read_parquet('data/processed/gamelogs_combined.parquet')

    # Extract unique teams
    teams = {}
    for _, row in df[['TEAM_NAME', 'TEAM_ABBREV']].drop_duplicates().iterrows():
        teams[row['TEAM_ABBREV']] = row['TEAM_NAME']

    return teams


def construct_feature_vector(
    player_recent: pd.DataFrame,
    opponent_abbrev: str,
    is_home: bool = True,
    days_rest: int = 2
) -> Tuple[np.ndarray, Dict]:
    """
    Construct feature vector for prediction.

    Uses player's recent performance and adjusts for:
    - Opponent defense
    - Home/away
    - Rest days
    """
    # Get most recent game as baseline
    latest = player_recent.iloc[0]

    # Get all feature names
    feature_names = load_feature_metadata()

    # Start with latest game's features
    features = latest[feature_names].copy()

    # Adjust home/away if different
    if 'HOME' in features.index:
        features['HOME'] = 1.0 if is_home else 0.0

    # Adjust rest days if different
    if 'REST_DAYS' in features.index:
        features['REST_DAYS'] = float(days_rest)
        features['IS_B2B'] = 1.0 if days_rest == 0 else 0.0

    # Load opponent team data to get defensive stats
    df_all = pd.read_parquet('data/processed/gamelogs_combined.parquet')

    # Get opponent's defensive rating (average from their recent games)
    # For BOS's defensive rating, get games where TEAM_ABBREV == "BOS"
    opp_games = df_all[df_all['TEAM_ABBREV'] == opponent_abbrev].sort_values('GAME_DATE', ascending=False)

    if len(opp_games) > 0:
        opp_recent = opp_games.head(20)  # Last 20 games

        # Update opponent features if they exist
        if 'OPP_DEF_RATING' in features.index and 'TEAM_DEF_RATING' in opp_recent.columns:
            features['OPP_DEF_RATING'] = opp_recent['TEAM_DEF_RATING'].mean()

        if 'OPP_PACE' in features.index and 'TEAM_PACE' in opp_recent.columns:
            features['OPP_PACE'] = opp_recent['TEAM_PACE'].mean()

    # Create context dict for display
    context = {
        'player_last_5_avg': {
            'PTS': latest['PTS_LAST_5'],
            'REB': latest['REB_LAST_5'],
            'AST': latest['AST_LAST_5']
        },
        'opponent_def_rating': features.get('OPP_DEF_RATING', 'N/A'),
        'home_away': 'Home' if is_home else 'Away',
        'days_rest': days_rest,
        'is_back_to_back': days_rest == 0
    }

    return features.values.astype(float), context


def make_prediction(
    player_name: str,
    opponent_abbrev: str,
    is_home: bool = True,
    days_rest: int = 2,
    verbose: bool = True
) -> Dict:
    """
    Make prediction for a specific matchup.

    Args:
        player_name: Full player name (e.g., "LeBron James")
        opponent_abbrev: Opponent team abbreviation (e.g., "BOS")
        is_home: True if home game
        days_rest: Number of days since last game
        verbose: Print detailed output

    Returns:
        Dict with predictions and context
    """
    if verbose:
        print("\n" + "="*70)
        print("NBA PLAYER PREDICTION - ENHANCED")
        print("="*70)
        print(f"\n🏀 Player: {player_name}")
        print(f"🏟️  Opponent: {opponent_abbrev}")
        print(f"📍 Location: {'Home' if is_home else 'Away'}")
        print(f"😴 Days Rest: {days_rest}{' (Back-to-Back)' if days_rest == 0 else ''}")

    # Load models
    models = load_models()

    # Get player's recent games
    player_recent = get_player_recent_games(player_name, n_games=10)

    # Construct feature vector
    features, context = construct_feature_vector(player_recent, opponent_abbrev, is_home, days_rest)
    features = features.reshape(1, -1)

    # Make predictions
    predictions = {}
    for target, model in models.items():
        pred = model.predict(features)[0]
        predictions[target] = float(pred)

    if verbose:
        print("\n" + "="*70)
        print("📊 PREDICTED PERFORMANCE")
        print("="*70)
        print(f"\n  PTS: {predictions['PTS']:>6.1f}")
        print(f"  REB: {predictions['REB']:>6.1f}")
        print(f"  AST: {predictions['AST']:>6.1f}")

        print("\n" + "="*70)
        print("📈 CONTEXT & FACTORS")
        print("="*70)
        print(f"\n  Last 5 Games Average:")
        print(f"    PTS: {context['player_last_5_avg']['PTS']:.1f}")
        print(f"    REB: {context['player_last_5_avg']['REB']:.1f}")
        print(f"    AST: {context['player_last_5_avg']['AST']:.1f}")

        if isinstance(context['opponent_def_rating'], (int, float)):
            print(f"\n  Opponent Defense Rating: {context['opponent_def_rating']:.1f}")

        # Calculate difference from recent average
        pts_diff = predictions['PTS'] - context['player_last_5_avg']['PTS']
        reb_diff = predictions['REB'] - context['player_last_5_avg']['REB']
        ast_diff = predictions['AST'] - context['player_last_5_avg']['AST']

        print("\n" + "="*70)
        print("🔍 PREDICTION vs RECENT FORM")
        print("="*70)
        print(f"\n  PTS: {pts_diff:+.1f} ({'+better' if pts_diff > 0 else 'worse'} than recent avg)")
        print(f"  REB: {reb_diff:+.1f} ({'+better' if reb_diff > 0 else 'worse'} than recent avg)")
        print(f"  AST: {ast_diff:+.1f} ({'+better' if ast_diff > 0 else 'worse'} than recent avg)")

        print("\n" + "="*70 + "\n")

    return {
        'predictions': predictions,
        'context': context,
        'player_name': player_name,
        'opponent': opponent_abbrev,
        'is_home': is_home,
        'days_rest': days_rest
    }


def interactive_mode():
    """Interactive prediction mode with enhanced interface."""
    print("\n" + "="*70)
    print("NBA PLAYER PREDICTION - INTERACTIVE MODE")
    print("="*70)

    # Load data to show available players
    df = pd.read_parquet('data/processed/features_engineered.parquet')
    players = sorted(df['PLAYER_NAME'].unique())

    print(f"\n📋 Available players: {len(players)}")
    print("\n🌟 Example star players:")
    star_players = [p for p in players if any(star in p for star in
                   ['LeBron', 'Curry', 'Durant', 'Giannis', 'Jokic', 'Embiid', 'Tatum', 'Doncic'])]
    for player in star_players[:10]:
        print(f"  • {player}")

    # Get player name with autocomplete hint
    print("\n" + "-"*70)
    player_input = input("\n👤 Enter player name (or part of name): ").strip()

    # Find matching players
    matches = [p for p in players if player_input.lower() in p.lower()]

    if len(matches) == 0:
        print(f"\n❌ No player found matching '{player_input}'")
        print("Using 'LeBron James' as example...")
        player_name = "LeBron James"
    elif len(matches) == 1:
        player_name = matches[0]
        print(f"\n✅ Found: {player_name}")
    else:
        print(f"\n🔍 Found {len(matches)} matching players:")
        for i, match in enumerate(matches[:10], 1):
            print(f"  {i}. {match}")

        if len(matches) > 10:
            print(f"  ... and {len(matches) - 10} more")

        try:
            choice = int(input("\nSelect player number (or 1 for first): ").strip() or "1")
            player_name = matches[choice - 1]
        except (ValueError, IndexError):
            player_name = matches[0]
            print(f"Using: {player_name}")

    # Get opponent
    teams = get_team_abbreviations()
    print("\n" + "-"*70)
    print("\n🏟️  Example teams: BOS, LAL, GSW, MIL, PHI, DEN, MIA, CHI")
    opponent = input("\nEnter opponent team abbreviation: ").strip().upper()

    if opponent not in teams:
        print(f"\n⚠️  Warning: '{opponent}' not in recent data, but proceeding...")
    else:
        print(f"✅ Opponent: {teams[opponent]} ({opponent})")

    # Get home/away
    print("\n" + "-"*70)
    location = input("\n📍 Home or Away? (H/A) [default: H]: ").strip().upper() or "H"
    is_home = location == "H"

    # Get rest days
    print("\n" + "-"*70)
    rest_input = input("\n😴 Days rest (0=back-to-back, 1-7) [default: 2]: ").strip() or "2"
    try:
        days_rest = int(rest_input)
        days_rest = max(0, min(7, days_rest))  # Clamp to 0-7
    except ValueError:
        days_rest = 2
        print("Using 2 days rest (default)")

    # Make prediction
    result = make_prediction(player_name, opponent, is_home, days_rest, verbose=True)

    return result


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='NBA Player Performance Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.predict_enhanced --player "LeBron James" --opponent "BOS"
  python -m src.predict_enhanced --player "Stephen Curry" --opponent "LAL" --away
  python -m src.predict_enhanced --player "Giannis Antetokounmpo" --opponent "MIA" --rest 0
  python -m src.predict_enhanced  (interactive mode)
        """
    )

    parser.add_argument('--player', type=str, help='Player name (full name or partial match)')
    parser.add_argument('--opponent', type=str, help='Opponent team abbreviation (e.g., BOS, LAL)')
    parser.add_argument('--home', action='store_true', default=True, help='Home game (default)')
    parser.add_argument('--away', action='store_true', help='Away game')
    parser.add_argument('--rest', type=int, default=2, help='Days rest (0-7, default: 2)')
    parser.add_argument('--quiet', action='store_true', help='Minimal output (JSON only)')

    args = parser.parse_args()

    if args.player and args.opponent:
        # Command-line mode
        is_home = not args.away
        result = make_prediction(
            args.player,
            args.opponent,
            is_home,
            args.rest,
            verbose=not args.quiet
        )

        if args.quiet:
            # Output JSON for scripting
            import json
            print(json.dumps(result, indent=2))
    else:
        # Interactive mode
        interactive_mode()


if __name__ == '__main__':
    main()
