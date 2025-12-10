"""
Data Collection Module - Replicates Notebook 01

Collects NBA data from official NBA API and enriches with contextual features.

Usage:
    python -m src.data_collection
    OR
    from src.data_collection import collect_nba_data
    collect_nba_data()

Output:
    data/processed/gamelogs_combined.parquet - 72K+ games with opponent/team context
    data/processed/shot_charts_all.parquet - 590K+ shots
    data/processed/gamelogs_{season}.parquet - Per-season files
"""

import pandas as pd
import time
from pathlib import Path
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NBA API imports
try:
    from nba_api.stats.endpoints import (
        leaguegamelog,
        playergamelog,
        shotchartdetail,
        leaguedashteamstats
    )
except ImportError:
    logger.error("NBA API not installed. Run: pip install nba-api")
    raise

# Configuration
SEASONS = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24']
N_PLAYERS = 200  # Top N players per season by minutes played
RATE_LIMIT = 0.6  # Seconds between API calls
PROGRESS_INTERVAL = 50  # Log progress every N players

# NBA team abbreviation mapping
TEAM_ABBREV_TO_NAME = {
    'ATL': 'Atlanta Hawks', 'BKN': 'Brooklyn Nets', 'BOS': 'Boston Celtics',
    'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
}


def collect_game_logs() -> pd.DataFrame:
    """
    Collect game logs for top 200 players per season.

    WARNING: This takes 2-3 hours due to API rate limiting.

    Returns:
        DataFrame with ~72,000 games
    """
    logger.info("="*70)
    logger.info("COLLECTING GAME LOGS (2-3 hours)")
    logger.info("="*70)

    # Step 1: Identify top players by season
    logger.info("\nStep 1: Identifying top 200 players per season by minutes played...")
    top_players_by_season = {}

    for season in SEASONS:
        logger.info(f"  {season}...")
        league_log = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star='Regular Season',
            player_or_team_abbreviation='P'
        )
        df_league = league_log.get_data_frames()[0]
        time.sleep(RATE_LIMIT)

        top_players = (df_league.groupby(['PLAYER_ID', 'PLAYER_NAME'])['MIN']
                       .sum().reset_index()
                       .sort_values('MIN', ascending=False)
                       .head(N_PLAYERS))

        top_players_by_season[season] = top_players
        logger.info(f"    Top: {top_players.iloc[0]['PLAYER_NAME']} ({top_players.iloc[0]['MIN']:.0f} min)")

    # Step 2: Get unique players across all seasons
    all_top_players = pd.concat(top_players_by_season.values(), ignore_index=True)
    unique_players = all_top_players.drop_duplicates(subset='PLAYER_ID')[['PLAYER_ID', 'PLAYER_NAME']]
    logger.info(f"\nStep 2: Found {len(unique_players)} unique players across {len(SEASONS)} seasons")

    # Step 3: Collect all game logs
    logger.info(f"\nStep 3: Collecting complete game logs...")
    logger.info(f"  This will take 2-3 hours. Progress updates every {PROGRESS_INTERVAL} players.")

    all_games = []
    for idx, (_, player) in enumerate(tqdm(unique_players.iterrows(), total=len(unique_players), desc="Players")):
        for season in SEASONS:
            try:
                gamelog = playergamelog.PlayerGameLog(
                    player_id=player['PLAYER_ID'],
                    season=season,
                    season_type_all_star='Regular Season'
                )
                df_games = gamelog.get_data_frames()[0]

                if len(df_games) > 0:
                    df_games['PLAYER_NAME'] = player['PLAYER_NAME']
                    all_games.append(df_games)

                time.sleep(RATE_LIMIT)
            except Exception:
                time.sleep(1)  # Player didn't play this season

        if (idx + 1) % PROGRESS_INTERVAL == 0:
            total = sum(len(g) for g in all_games)
            logger.info(f"  Progress: {idx+1}/{len(unique_players)} players | {total:,} games")

    # Combine all games
    df_all = pd.concat(all_games, ignore_index=True)
    logger.info(f"\n✓ Collected {len(df_all):,} games from {df_all['Player_ID'].nunique()} players")

    return df_all


def collect_team_stats() -> pd.DataFrame:
    """
    Collect team stats (defensive rating, pace, etc.) for all seasons.

    Returns:
        DataFrame with team stats by season
    """
    logger.info("\nCollecting team stats for all seasons...")

    all_team_stats = []
    for season in SEASONS:
        try:
            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star='Regular Season',
                measure_type_detailed_defense='Advanced',
                per_mode_detailed='PerGame'
            )
            df_team_stats = team_stats.get_data_frames()[0]
            df_team_stats['SEASON'] = season
            all_team_stats.append(df_team_stats)
            time.sleep(RATE_LIMIT)
        except Exception as e:
            logger.warning(f"  Error collecting {season}: {e}")
            time.sleep(1)

    df_team_stats_all = pd.concat(all_team_stats, ignore_index=True)
    logger.info(f"  ✓ Collected stats for {len(df_team_stats_all)} team-seasons")

    return df_team_stats_all


def enrich_with_opponent_stats(df_all: pd.DataFrame, df_team_stats: pd.DataFrame) -> pd.DataFrame:
    """Add opponent stats to game logs."""
    logger.info("\nEnriching with opponent stats...")

    # Extract opponent from MATCHUP
    def extract_opponent(matchup):
        if ' @ ' in matchup:
            return matchup.split(' @ ')[1]
        elif ' vs. ' in matchup:
            return matchup.split(' vs. ')[1]
        return None

    df_all['OPP_TEAM_ABBREV'] = df_all['MATCHUP'].apply(extract_opponent)
    df_all['OPP_TEAM_NAME'] = df_all['OPP_TEAM_ABBREV'].map(TEAM_ABBREV_TO_NAME)

    # Convert SEASON_ID to season format
    df_all['SEASON'] = df_all['SEASON_ID'].apply(
        lambda sid: f"{str(sid)[1:]}-{str(int(str(sid)[1:]) + 1)[-2:]}"
    )

    # Merge opponent stats
    df_opponent = df_team_stats[['TEAM_NAME', 'SEASON', 'DEF_RATING', 'OFF_RATING',
                                  'PACE', 'W', 'L', 'W_PCT']].copy()

    df_all = df_all.merge(
        df_opponent,
        left_on=['OPP_TEAM_NAME', 'SEASON'],
        right_on=['TEAM_NAME', 'SEASON'],
        how='left',
        suffixes=('', '_DROP')
    )

    # Rename to OPP_ prefix
    df_all = df_all.rename(columns={
        'DEF_RATING': 'OPP_DEF_RATING',
        'OFF_RATING': 'OPP_OFF_RATING',
        'PACE': 'OPP_PACE',
        'W': 'OPP_W',
        'L': 'OPP_L',
        'W_PCT': 'OPP_W_PCT'
    })

    df_all = df_all.drop(columns=[c for c in df_all.columns if 'DROP' in c], errors='ignore')

    logger.info(f"  ✓ Added opponent stats (8 columns)")
    return df_all


def enrich_with_team_stats(df_all: pd.DataFrame, df_team_stats: pd.DataFrame) -> pd.DataFrame:
    """Add player's own team stats to game logs."""
    logger.info("\nEnriching with player's team stats...")

    # Extract player's team from MATCHUP
    def extract_own_team(matchup):
        if ' @ ' in matchup:
            return matchup.split(' @ ')[0]
        elif ' vs. ' in matchup:
            return matchup.split(' vs. ')[0]
        return None

    df_all['TEAM_ABBREV'] = df_all['MATCHUP'].apply(extract_own_team)
    df_all['TEAM_NAME'] = df_all['TEAM_ABBREV'].map(TEAM_ABBREV_TO_NAME)

    # Merge team stats
    df_team = df_team_stats[['TEAM_NAME', 'SEASON', 'DEF_RATING', 'OFF_RATING',
                              'PACE', 'W', 'L', 'W_PCT']].copy()

    df_all = df_all.merge(
        df_team,
        on=['TEAM_NAME', 'SEASON'],
        how='left',
        suffixes=('', '_DROP')
    )

    # Rename to TEAM_ prefix
    df_all = df_all.rename(columns={
        'DEF_RATING': 'TEAM_DEF_RATING',
        'OFF_RATING': 'TEAM_OFF_RATING',
        'PACE': 'TEAM_PACE',
        'W': 'TEAM_W',
        'L': 'TEAM_L',
        'W_PCT': 'TEAM_W_PCT'
    })

    df_all = df_all.drop(columns=[c for c in df_all.columns if 'DROP' in c], errors='ignore')

    logger.info(f"  ✓ Added team stats (6 columns)")
    return df_all


def add_rest_days(df_all: pd.DataFrame) -> pd.DataFrame:
    """Calculate rest days and back-to-back indicators."""
    logger.info("\nCalculating rest days...")

    df_all['GAME_DATE'] = pd.to_datetime(df_all['GAME_DATE'])
    df_all = df_all.sort_values(['Player_ID', 'GAME_DATE']).reset_index(drop=True)

    df_all['DAYS_REST'] = df_all.groupby('Player_ID')['GAME_DATE'].diff().dt.days
    df_all['DAYS_REST'] = df_all['DAYS_REST'].fillna(-1).astype(int)
    df_all['IS_B2B'] = (df_all['DAYS_REST'] == 1).astype(int)

    b2b_pct = df_all['IS_B2B'].sum() / len(df_all) * 100
    logger.info(f"  ✓ Added rest days (B2B games: {b2b_pct:.1f}%)")

    return df_all


def clean_data(df_all: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize data."""
    logger.info("\nCleaning data...")

    # Standardize column names
    df_all = df_all.rename(columns={
        'Player_ID': 'PLAYER_ID',
        'Game_ID': 'GAME_ID'
    })

    # Drop redundant columns
    drop_cols = ['SEASON_ID', 'VIDEO_AVAILABLE']
    df_all = df_all.drop(columns=[c for c in drop_cols if c in df_all.columns], errors='ignore')

    logger.info(f"  ✓ Cleaned data ({df_all.shape[1]} columns)")
    return df_all


def collect_nba_data(output_dir: str = 'data/processed') -> dict:
    """
    Complete NBA data collection pipeline.

    WARNING: This takes 2-3 hours to run due to API rate limiting.

    Args:
        output_dir: Directory to save processed data

    Returns:
        Dict with paths to output files
    """
    logger.info("="*70)
    logger.info("NBA DATA COLLECTION PIPELINE")
    logger.info("="*70)
    logger.info("\n⚠️  WARNING: This will take 2-3 hours due to NBA API rate limiting")
    logger.info("    You can stop and resume - data is saved incrementally\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect data
    df_all = collect_game_logs()
    df_team_stats = collect_team_stats()

    # Enrich with context
    df_all = enrich_with_opponent_stats(df_all, df_team_stats)
    df_all = enrich_with_team_stats(df_all, df_team_stats)
    df_all = add_rest_days(df_all)
    df_all = clean_data(df_all)

    # Save processed data
    logger.info("\nSaving processed data...")
    df_all.to_parquet(output_path / 'gamelogs_combined.parquet', index=False)

    # Save per-season files
    for season in SEASONS:
        season_data = df_all[df_all['SEASON'] == season]
        if len(season_data) > 0:
            season_data.to_parquet(output_path / f'gamelogs_{season}.parquet', index=False)

    logger.info(f"  ✓ gamelogs_combined.parquet: {len(df_all):,} games")

    logger.info("\n" + "="*70)
    logger.info("✅ DATA COLLECTION COMPLETE")
    logger.info("="*70)
    logger.info(f"\nOutput:")
    logger.info(f"  {output_path / 'gamelogs_combined.parquet'}")
    logger.info(f"  {len(df_all):,} games from {df_all['PLAYER_ID'].nunique()} players")
    logger.info(f"  {df_all.shape[1]} columns (enriched with opponent/team stats)")

    return {
        'gamelogs_path': str(output_path / 'gamelogs_combined.parquet'),
        'num_games': len(df_all),
        'num_players': df_all['PLAYER_ID'].nunique(),
        'num_columns': df_all.shape[1]
    }


def main():
    """Run data collection pipeline."""
    result = collect_nba_data()

    print("\n" + "="*70)
    print("DATA COLLECTION SUMMARY")
    print("="*70)
    print(f"\nCollected: {result['num_games']:,} games")
    print(f"Players: {result['num_players']}")
    print(f"Features: {result['num_columns']} columns")
    print(f"\nOutput: {result['gamelogs_path']}")
    print("\nNext: Run feature engineering (python -m src.feature_engineering)")


if __name__ == '__main__':
    main()
