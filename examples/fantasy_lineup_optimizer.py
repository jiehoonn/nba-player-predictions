"""
Fantasy Basketball Lineup Optimizer

Uses the prediction model to compare multiple players and optimize lineup.

Usage:
    python examples/fantasy_lineup_optimizer.py
"""

import subprocess
import json
import pandas as pd
import sys
from pathlib import Path
from typing import List, Dict


def predict_player(player_name: str, opponent: str, is_home: bool = True, days_rest: int = 2) -> Dict:
    """Make prediction for a single player."""
    # Use current Python interpreter
    python_exe = sys.executable

    cmd = [
        python_exe, '-m', 'src.predict_enhanced',
        '--player', player_name,
        '--opponent', opponent,
        '--quiet'
    ]

    if not is_home:
        cmd.append('--away')

    cmd.extend(['--rest', str(days_rest)])

    # Get project root directory
    project_root = Path(__file__).parent.parent

    # Suppress Python warnings in subprocess
    import os
    env = os.environ.copy()
    env['PYTHONWARNINGS'] = 'ignore'

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root, env=env)

    # Parse JSON output (skip warning lines)
    # JSON starts with { and may span multiple lines
    output = result.stdout.strip()

    # Find the start of JSON (first {)
    json_start = output.find('{')
    if json_start == -1:
        raise ValueError(f"No JSON found in output for {player_name}")

    # Extract JSON (from first { to end)
    json_str = output[json_start:]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON for {player_name}: {e}")


def compare_players(matchups: List[Dict]) -> pd.DataFrame:
    """
    Compare multiple players for fantasy lineup decisions.

    Args:
        matchups: List of dicts with keys: player, opponent, is_home, days_rest

    Returns:
        DataFrame with predictions and fantasy scores
    """
    results = []

    print("\n" + "="*80)
    print("FANTASY BASKETBALL LINEUP OPTIMIZER")
    print("="*80 + "\n")

    for i, matchup in enumerate(matchups, 1):
        print(f"[{i}/{len(matchups)}] Predicting {matchup['player']} vs {matchup['opponent']}...")

        try:
            pred = predict_player(
                matchup['player'],
                matchup['opponent'],
                matchup.get('is_home', True),
                matchup.get('days_rest', 2)
            )

            # Calculate fantasy points (standard scoring)
            # PTS = 1, REB = 1.2, AST = 1.5, (simplified)
            fantasy_pts = (
                pred['predictions']['PTS'] * 1.0 +
                pred['predictions']['REB'] * 1.2 +
                pred['predictions']['AST'] * 1.5
            )

            results.append({
                'Player': matchup['player'],
                'Opponent': matchup['opponent'],
                'Location': 'Home' if pred['is_home'] else 'Away',
                'Rest': pred['days_rest'],
                'Pred_PTS': pred['predictions']['PTS'],
                'Pred_REB': pred['predictions']['REB'],
                'Pred_AST': pred['predictions']['AST'],
                'Fantasy_Score': fantasy_pts,
                'Last5_PTS': pred['context']['player_last_5_avg']['PTS'],
                'Opp_Def': pred['context']['opponent_def_rating']
            })

        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            continue

    df = pd.DataFrame(results)
    return df.sort_values('Fantasy_Score', ascending=False)


def main():
    """Example: Compare players for tonight's lineup."""

    # Define tonight's matchups
    matchups = [
        {'player': 'LeBron James', 'opponent': 'BOS', 'is_home': True, 'days_rest': 2},
        {'player': 'Stephen Curry', 'opponent': 'LAL', 'is_home': True, 'days_rest': 2},
        {'player': 'Giannis Antetokounmpo', 'opponent': 'MIA', 'is_home': True, 'days_rest': 0},  # B2B
        {'player': 'Kevin Durant', 'opponent': 'MIL', 'is_home': False, 'days_rest': 2},
        {'player': 'James Harden', 'opponent': 'PHI', 'is_home': True, 'days_rest': 2},
    ]

    # Get predictions
    df = compare_players(matchups)

    # Display results
    print("\n" + "="*80)
    print("FANTASY LINEUP RECOMMENDATIONS (Sorted by Fantasy Score)")
    print("="*80 + "\n")

    print(df.to_string(index=False))

    print("\n" + "="*80)
    print("TOP 3 PICKS FOR TONIGHT")
    print("="*80 + "\n")

    for i, row in df.head(3).iterrows():
        print(f"#{i+1}. {row['Player']}")
        print(f"   Matchup: vs {row['Opponent']} ({row['Location']}, {row['Rest']} days rest)")
        print(f"   Predicted: {row['Pred_PTS']:.1f} PTS, {row['Pred_REB']:.1f} REB, {row['Pred_AST']:.1f} AST")
        print(f"   Fantasy Score: {row['Fantasy_Score']:.1f}")
        print(f"   Opponent Defense: {row['Opp_Def']:.1f}")
        print()

    print("\n" + "="*80)
    print("NOTES")
    print("="*80)
    print("• Fantasy Score = (PTS × 1.0) + (REB × 1.2) + (AST × 1.5)")
    print("• Lower opponent defense rating = easier matchup")
    print("• Back-to-back games (0 rest) typically hurt performance")
    print("• Away games can impact performance\n")


if __name__ == '__main__':
    main()
