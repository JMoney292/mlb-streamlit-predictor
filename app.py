# app.py
# Streamlit MLB Winner + Predicted Score (Baseline MVP)
# Paste this into Streamlit as app.py
#
# What this does:
# - You pick Away + Home teams (by abbreviation)
# - Optionally enter each starting pitcher ERA
# - App pulls each team's season-to-date + last-N completed games (via MLB Stats API)
# - Produces a baseline win probability + predicted score
#
# IMPORTANT:
# - This is a baseline heuristic model (not a trained ML model). It WILL run and give picks,
#   but profitability/accuracy is not guaranteed. Use it as an MVP you can improve.

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
import streamlit as st


API_BASE = "https://statsapi.mlb.com/api/v1"
USER_AGENT = "streamlit-mlb-model/1.0"

DEFAULT_LEAGUE_RPG_PER_TEAM = 4.40  # league avg runs per team per game baseline
DEFAULT_LEAGUE_ERA = 4.20           # league avg starter ERA baseline
DEFAULT_HFA_LOGIT = 0.22            # home-field advantage in log-odds


# ----------------------------
# Helpers
# ----------------------------
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def api_get(path: str, params: Optional[Dict[str, Any]] = None, retries: int = 3, sleep_s: float = 0.7) -> Dict[str, Any]:
    """Small, polite wrapper around MLB Stats API. Retries a few times."""
    url = f"{API_BASE}{path}"
    headers = {"User-Agent": USER_AGENT}

    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params or {}, headers=headers, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(sleep_s * (i + 1))
    raise RuntimeError(f"API request failed: {url} params={params} err={last_err}")


@st.cache_data(ttl=60 * 60 * 12)  # 12h cache
def load_teams() -> pd.DataFrame:
    """Load MLB teams list once and cache."""
    data = api_get("/teams", params={"sportId": 1})
    teams = data.get("teams", [])
    rows = []
    for t in teams:
        rows.append({
            "id": t.get("id"),
            "name": t.get("name"),
            "abbreviation": (t.get("abbreviation") or "").upper(),
            "teamName": t.get("teamName"),
            "locationName": t.get("locationName"),
            "clubName": t.get("clubName"),
            "league": (t.get("league", {}) or {}).get("name"),
            "division": (t.get("division", {}) or {}).get("name"),
        })
    df = pd.DataFrame(rows).dropna(subset=["id", "abbreviation"])
    df = df.sort_values(["league", "division", "name"]).reset_index(drop=True)
    return df


@st.cache_data(ttl=60 * 20)  # 20m cache
def fetch_team_schedule(team_id: int, season: int) -> List[Dict[str, Any]]:
    """
    Fetch full regular-season schedule for a team (includes final scores in linescore when final).
    This is ~162 games, so it’s reasonable per team.
    """
    data = api_get("/schedule", params={
        "sportId": 1,
        "teamId": team_id,
        "season": season,
        "gameTypes": "R",
        "hydrate": "linescore"
    })
    dates = data.get("dates", []) or []
    games = []
    for d in dates:
        for g in (d.get("games", []) or []):
            games.append(g)
    return games


def schedule_to_game_log(games: List[Dict[str, Any]], team_id: int) -> pd.DataFrame:
    """
    Convert schedule JSON to a compact game log (completed games only).
    Includes: game_date, opponent_id, is_home, runs_for, runs_against.
    """
    rows = []
    for g in games:
        status = (g.get("status", {}) or {}).get("detailedState", "")
        if status != "Final":
            continue

        teams = g.get("teams", {}) or {}
        home = (teams.get("home", {}) or {}).get("team", {}) or {}
        away = (teams.get("away", {}) or {}).get("team", {}) or {}
        home_id = home.get("id")
        away_id = away.get("id")
        if home_id is None or away_id is None:
            continue

        is_home = (home_id == team_id)
        opp_id = away_id if is_home else home_id

        # Linescore runs
        linescore = g.get("linescore", {}) or {}
        runs_home = (linescore.get("teams", {}) or {}).get("home", {}).get("runs")
        runs_away = (linescore.get("teams", {}) or {}).get("away", {}).get("runs")

        if runs_home is None or runs_away is None:
            # Sometimes linescore missing; skip
            continue

        runs_for = runs_home if is_home else runs_away
        runs_against = runs_away if is_home else runs_home

        # gameDate like "2026-04-02T19:10:00Z"
        game_date_str = g.get("gameDate")
        try:
            game_dt = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
        except Exception:
            continue

        rows.append({
            "game_dt_utc": game_dt,
            "is_home": bool(is_home),
            "opponent_id": int(opp_id),
            "runs_for": int(runs_for),
            "runs_against": int(runs_against),
        })

    if not rows:
        return pd.DataFrame(columns=["game_dt_utc", "is_home", "opponent_id", "runs_for", "runs_against"])

    df = pd.DataFrame(rows).sort_values("game_dt_utc").reset_index(drop=True)
    return df


@dataclass
class TeamStats:
    games_played: int
    rs_pg: float
    ra_pg: float
    rd_pg: float
    lastn_rs_pg: float
    lastn_ra_pg: float
    lastn_rd_pg: float


def compute_team_stats(game_log: pd.DataFrame, last_n: int) -> TeamStats:
    if game_log.empty:
        return TeamStats(
            games_played=0,
            rs_pg=DEFAULT_LEAGUE_RPG_PER_TEAM,
            ra_pg=DEFAULT_LEAGUE_RPG_PER_TEAM,
            rd_pg=0.0,
            lastn_rs_pg=DEFAULT_LEAGUE_RPG_PER_TEAM,
            lastn_ra_pg=DEFAULT_LEAGUE_RPG_PER_TEAM,
            lastn_rd_pg=0.0,
        )

    games_played = int(len(game_log))
    rs_pg = float(game_log["runs_for"].mean())
    ra_pg = float(game_log["runs_against"].mean())
    rd_pg = rs_pg - ra_pg

    tail = game_log.tail(max(1, min(last_n, games_played)))
    lastn_rs_pg = float(tail["runs_for"].mean())
    lastn_ra_pg = float(tail["runs_against"].mean())
    lastn_rd_pg = lastn_rs_pg - lastn_ra_pg

    return TeamStats(
        games_played=games_played,
        rs_pg=rs_pg,
        ra_pg=ra_pg,
        rd_pg=rd_pg,
        lastn_rs_pg=lastn_rs_pg,
        lastn_ra_pg=lastn_ra_pg,
        lastn_rd_pg=lastn_rd_pg,
    )


def reliability_weight(games_played: int, floor: int = 10, full: int = 50) -> float:
    """
    Early season is noisy. This returns a 0..1 weight based on sample size.
    - <=floor games -> near 0
    - >=full games -> 1
    """
    if games_played <= floor:
        return 0.15
    if games_played >= full:
        return 1.0
    # linear ramp
    return 0.15 + 0.85 * (games_played - floor) / (full - floor)


def predict_home_win_prob(
    home: TeamStats,
    away: TeamStats,
    home_sp_era: Optional[float],
    away_sp_era: Optional[float],
) -> Tuple[float, Dict[str, float]]:
    """
    Baseline probability model (heuristic):
    logit = HFA
          + b_season * (home.rd_pg - away.rd_pg)
          + b_recent * (home.lastn_rd_pg - away.lastn_rd_pg)
          + b_pitch  * (away_sp_era - home_sp_era)
    """
    h_era = home_sp_era if home_sp_era is not None else DEFAULT_LEAGUE_ERA
    a_era = away_sp_era if away_sp_era is not None else DEFAULT_LEAGUE_ERA

    # Weights shrink early-season signals
    w_home = reliability_weight(home.games_played)
    w_away = reliability_weight(away.games_played)
    w = min(w_home, w_away)

    # Coefficients (tunable)
    b_season = 0.38 * w
    b_recent = 0.22 * w
    b_pitch = 0.18  # pitcher input is often meaningful even early-season

    season_edge = (home.rd_pg - away.rd_pg)
    recent_edge = (home.lastn_rd_pg - away.lastn_rd_pg)
    pitch_edge = (a_era - h_era)  # positive helps home if away ERA is higher

    logit = DEFAULT_HFA_LOGIT + b_season * season_edge + b_recent * recent_edge + b_pitch * pitch_edge
    p_home = clamp(sigmoid(logit), 0.05, 0.95)

    parts = {
        "logit": float(logit),
        "w_sample": float(w),
        "season_edge": float(season_edge),
        "recent_edge": float(recent_edge),
        "pitch_edge": float(pitch_edge),
        "b_season": float(b_season),
        "b_recent": float(b_recent),
        "b_pitch": float(b_pitch),
    }
    return p_home, parts


def predict_expected_runs(
    home: TeamStats,
    away: TeamStats,
    home_sp_era: Optional[float],
    away_sp_era: Optional[float],
    league_rpg: float = DEFAULT_LEAGUE_RPG_PER_TEAM,
) -> Tuple[float, float]:
    """
    Simple run expectation:
    - Base runs per team per game = league_rpg
    - Adjust with offense (rs_pg) and defense (ra_pg) + recent form
    - Adjust with opposing starter ERA

    This is NOT a full sabermetric run model, but gives reasonable-looking scores.
    """
    h_era = home_sp_era if home_sp_era is not None else DEFAULT_LEAGUE_ERA
    a_era = away_sp_era if away_sp_era is not None else DEFAULT_LEAGUE_ERA

    # Blend season + recent (recent slightly)
    w_home = reliability_weight(home.games_played)
    w_away = reliability_weight(away.games_played)
    w = min(w_home, w_away)

    home_off = (0.75 * home.rs_pg + 0.25 * home.lastn_rs_pg)
    home_def = (0.75 * home.ra_pg + 0.25 * home.lastn_ra_pg)
    away_off = (0.75 * away.rs_pg + 0.25 * away.lastn_rs_pg)
    away_def = (0.75 * away.ra_pg + 0.25 * away.lastn_ra_pg)

    # Normalize vs league
    home_off_mult = (home_off / league_rpg) if league_rpg > 0 else 1.0
    away_off_mult = (away_off / league_rpg) if league_rpg > 0 else 1.0
    home_def_mult = (home_def / league_rpg) if league_rpg > 0 else 1.0
    away_def_mult = (away_def / league_rpg) if league_rpg > 0 else 1.0

    # Pitcher multiplier (opponent faces this pitcher)
    # Higher ERA -> more runs allowed
    sp_scale = 0.55  # how strongly we use ERA (tunable)
    home_pitch_mult = (a_era / DEFAULT_LEAGUE_ERA) ** sp_scale  # home hitters vs away SP
    away_pitch_mult = (h_era / DEFAULT_LEAGUE_ERA) ** sp_scale  # away hitters vs home SP

    # Defense multiplier: if opponent allows more than league, increase expected runs
    # Use sqrt to soften
    home_vs_away_def = math.sqrt(away_def_mult)
    away_vs_home_def = math.sqrt(home_def_mult)

    # Home field tends to slightly increase home scoring
    hfa_runs_mult = 1.03

    lam_home = league_rpg * home_off_mult * home_vs_away_def * home_pitch_mult * hfa_runs_mult
    lam_away = league_rpg * away_off_mult * away_vs_home_def * away_pitch_mult

    # Early season shrink toward league average
    shrink = 1.0 - w
    lam_home = lam_home * w + league_rpg * shrink
    lam_away = lam_away * w + league_rpg * shrink

    lam_home = clamp(lam_home, 1.5, 8.5)
    lam_away = clamp(lam_away, 1.5, 8.5)
    return lam_home, lam_away


def score_from_lambdas(lam_home: float, lam_away: float, p_home: float) -> Tuple[int, int]:
    """
    Convert expected runs into an integer score.
    If tie, nudge winner by +1.
    """
    ph = int(round(lam_home))
    pa = int(round(lam_away))
    ph = max(0, ph)
    pa = max(0, pa)

    if ph == pa:
        if p_home >= 0.5:
            ph += 1
        else:
            pa += 1
    return ph, pa


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="MLB Winner Model (Streamlit MVP)", layout="wide")

st.title("⚾ MLB Winner + Predicted Score (Streamlit MVP)")
st.caption(
    "Baseline model using team run differential (season + recent) + optional starting pitcher ERA. "
    "This is an MVP you can improve into a trained model."
)

teams_df = load_teams()
abbrs = teams_df["abbreviation"].tolist()
abbr_to_name = dict(zip(teams_df["abbreviation"], teams_df["name"]))
abbr_to_id = dict(zip(teams_df["abbreviation"], teams_df["id"]))

with st.sidebar:
    st.header("Settings")
    current_year = datetime.utcnow().year
    season = st.number_input("Season (regular season)", min_value=2000, max_value=current_year + 1, value=current_year, step=1)
    last_n = st.slider("Recent form window (last N games)", min_value=5, max_value=20, value=10, step=1)
    league_rpg = st.number_input("League avg runs per team/game (baseline)", min_value=3.0, max_value=6.5, value=float(DEFAULT_LEAGUE_RPG_PER_TEAM), step=0.05)
    st.divider()
    st.write("Notes")
    st.write("- If a season has no completed games yet, the model falls back to league-average baselines.")
    st.write("- Pitcher ERA helps, but it’s noisy; later we can upgrade to xERA/FIP-style inputs.")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    away_abbr = st.selectbox("Away team", abbrs, index=0, format_func=lambda a: f"{a} — {abbr_to_name.get(a, a)}")
with col2:
    home_abbr = st.selectbox("Home team", abbrs, index=1, format_func=lambda a: f"{a} — {abbr_to_name.get(a, a)}")

with col3:
    use_pitchers = st.checkbox("Include starting pitcher ERA", value=True)

pitch_col1, pitch_col2 = st.columns(2)
away_era = None
home_era = None

if use_pitchers:
    with pitch_col1:
        away_era = st.number_input("Away SP ERA (optional)", min_value=0.00, max_value=20.00, value=float(DEFAULT_LEAGUE_ERA), step=0.01)
    with pitch_col2:
        home_era = st.number_input("Home SP ERA (optional)", min_value=0.00, max_value=20.00, value=float(DEFAULT_LEAGUE_ERA), step=0.01)

predict_btn = st.button("Predict", type="primary")

if predict_btn:
    try:
        away_id = int(abbr_to_id[away_abbr])
        home_id = int(abbr_to_id[home_abbr])
    except Exception:
        st.error("Could not map team abbreviations to team IDs.")
        st.stop()

    with st.spinner("Pulling team schedules and computing form..."):
        away_sched = fetch_team_schedule(away_id, int(season))
        home_sched = fetch_team_schedule(home_id, int(season))

        away_log = schedule_to_game_log(away_sched, away_id)
        home_log = schedule_to_game_log(home_sched, home_id)

        away_stats = compute_team_stats(away_log, last_n=last_n)
        home_stats = compute_team_stats(home_log, last_n=last_n)

        p_home, parts = predict_home_win_prob(
            home=home_stats,
            away=away_stats,
            home_sp_era=safe_float(home_era) if use_pitchers else None,
            away_sp_era=safe_float(away_era) if use_pitchers else None,
        )

        lam_home, lam_away = predict_expected_runs(
            home=home_stats,
            away=away_stats,
            home_sp_era=safe_float(home_era) if use_pitchers else None,
            away_sp_era=safe_float(away_era) if use_pitchers else None,
            league_rpg=float(league_rpg),
        )

        pred_home, pred_away = score_from_lambdas(lam_home, lam_away, p_home=p_home)

        pick = home_abbr if p_home >= 0.5 else away_abbr
        pick_prob = p_home if pick == home_abbr else (1.0 - p_home)

    res1, res2, res3 = st.columns(3)
    with res1:
        st.metric("Pick", f"{pick} — {abbr_to_name.get(pick, pick)}")
    with res2:
        st.metric("Win probability", f"{pick_prob:.1%}")
    with res3:
        st.metric("Predicted score", f"{home_abbr} {pred_home} — {away_abbr} {pred_away}")

    st.divider()

    # Show stats tables
    def stats_dict(abbr: str, s: TeamStats) -> Dict[str, Any]:
        return {
            "Team": f"{abbr} — {abbr_to_name.get(abbr, abbr)}",
            "Games played": s.games_played,
            "Season RS/G": round(s.rs_pg, 2),
            "Season RA/G": round(s.ra_pg, 2),
            "Season RD/G": round(s.rd_pg, 2),
            f"Last {last_n} RS/G": round(s.lastn_rs_pg, 2),
            f"Last {last_n} RA/G": round(s.lastn_ra_pg, 2),
            f"Last {last_n} RD/G": round(s.lastn_rd_pg, 2),
        }

    df_stats = pd.DataFrame([stats_dict(away_abbr, away_stats), stats_dict(home_abbr, home_stats)])
    st.subheader("Team form inputs used")
    st.dataframe(df_stats, use_container_width=True)

    st.subheader("Model components (for tuning)")
    comp = {
        "sample_weight (0..1)": round(parts["w_sample"], 3),
        "season_edge (RD/G home - away)": round(parts["season_edge"], 3),
        "recent_edge (lastN RD/G home - away)": round(parts["recent_edge"], 3),
        "pitch_edge (awayERA - homeERA)": round(parts["pitch_edge"], 3),
        "b_season": round(parts["b_season"], 3),
        "b_recent": round(parts["b_recent"], 3),
        "b_pitch": round(parts["b_pitch"], 3),
        "logit": round(parts["logit"], 3),
        "p_home": round(p_home, 4),
        "lambda_home_runs": round(lam_home, 2),
        "lambda_away_runs": round(lam_away, 2),
    }
    st.json(comp)

    st.caption(
        "Next upgrades (if you want real edge): train on multiple seasons + add bullpen fatigue, "
        "lineups, park/weather, and calibrate probabilities."
    )
