"""Local-development implementation of the deployment-supplied ``api.server``.

``api/application.py`` imports ``app`` from ``api.server``, which is gitignored
and therefore absent from a fresh clone. This module builds an equivalent
FastAPI application from tracked code only, so ``make doctor`` passes and
``make fullstack`` can serve the Next.js dashboard without the private modules.

The application is assembled by :func:`install`, which is handed the namespace
of the generated ``api/server.py`` shim. Handlers resolve ``read_dataframe``,
``_review_runs_in_flight``, and ``_run_agent_review_background`` through that
namespace on every call, because ``api.server`` is the patch surface the API
test suite targets.

Value-row reads all go through :func:`api.value_visibility.value_visibility_scope`,
which hides unpublished, unjoinable, and synthetic rows unless ``DEMO_MODE`` is
on. Nothing here reimplements projection or pricing logic: it reads what the
pipeline persisted.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from api.auth import (
    UserCreate,
    UserLogin,
    UserResponse,
    authenticate_user,
    create_user,
    logout_user,
    validate_session,
)
from api.cache import make_cache_key, value_bets_cache
from api.explainability import build_why_payload
from api.nba_router import router as nba_router
from api.pipeline_router import (
    require_pipeline_operator,
    require_pipeline_reader,
    router as pipeline_router,
)
from api.value_visibility import value_visibility_scope
from config import config
from risk_manager import compute_exposure, detect_correlations, detect_team_stacks
from utils.db import execute, fetchall, fetchone, read_dataframe

logger = logging.getLogger(__name__)

DEFAULT_DEV_ORIGINS = ("http://localhost:3000", "http://localhost:3001")

VALUE_ROW_SQL = """
    SELECT
        v.season, v.week, v.player_id, v.event_id, v.team, v.market, v.sportsbook,
        v.line, v.price, v.side, v.mu, v.sigma, v.p_win, v.edge_percentage,
        v.expected_roi, v.kelly_fraction, v.stake, v.generated_at,
        v.confidence_score, v.confidence_tier, v.published_run_id,
        d.player_name, d.position,
        g.home_team, g.away_team
    FROM materialized_value_view v
    LEFT JOIN player_dim d ON d.player_id = v.player_id
    LEFT JOIN games g ON g.game_id = v.event_id
    WHERE {visibility}
      AND v.season = ? AND v.week = ?
    ORDER BY v.edge_percentage DESC
"""

CSV_COLUMNS = (
    "season",
    "week",
    "player_id",
    "player_name",
    "position",
    "team",
    "opponent",
    "market",
    "sportsbook",
    "side",
    "line",
    "price",
    "mu",
    "sigma",
    "p_win",
    "edge_percentage",
    "expected_roi",
    "kelly_fraction",
    "stake",
    "confidence_score",
    "confidence_tier",
    "event_id",
    "generated_at",
)

_bearer = HTTPBearer(auto_error=False)


# ── Coercion helpers ────────────────────────────────────────────────────────
# Every payload value is rebuilt as a plain Python scalar. pandas hands back
# numpy types that FastAPI cannot serialize, and NaN must become null rather
# than the string "nan".


def _opt_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _opt_int(value: Any) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _opt_str(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _num(value: Any, default: float = 0.0) -> float:
    coerced = _opt_float(value)
    return default if coerced is None else coerced


def _opponent(row: Any) -> Optional[str]:
    team = _opt_str(row.get("team"))
    home = _opt_str(row.get("home_team"))
    away = _opt_str(row.get("away_team"))
    if team is None or home is None or away is None:
        return None
    if team == home:
        return away
    if team == away:
        return home
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Authentication dependency ───────────────────────────────────────────────
# Defined at module scope so `app.dependency_overrides[get_current_user]` in the
# test suite targets the same object the routes depend on.


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> UserResponse:
    """Resolve the signed-in user from a bearer session token."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = validate_session(credentials.credentials)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user


async def _optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[UserResponse]:
    if credentials is None or not credentials.credentials:
        return None
    return validate_session(credentials.credentials)


class BetCreate(BaseModel):
    """A bet the user is recording on their own slip.

    ``stake``/``edge_at_placement`` are accepted as aliases because the Next.js
    client posts those names while the stored columns are ``stake_units`` and
    ``model_edge``.
    """

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    season: int
    week: int
    player_id: str
    player_name: Optional[str] = None
    market: str
    sportsbook: str
    side: Literal["over", "under"] = "over"
    line: float
    price: int
    stake_units: float = Field(validation_alias=AliasChoices("stake_units", "stake"))
    stake_dollars: Optional[float] = None
    model_edge: Optional[float] = Field(
        default=None, validation_alias=AliasChoices("model_edge", "edge_at_placement")
    )
    confidence_tier: Optional[str] = None


def _allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "")
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or list(DEFAULT_DEV_ORIGINS)


def _run_agent_review_background(run_id: str, season: int, week: int) -> None:
    """Run the tracked agent coordinator for a published card."""
    from agents.coordinator import run_all_agents

    decisions = run_all_agents(season=season, week=week)
    logger.info(
        "Agent review finished",
        extra={
            "event": "api.agent_review_complete",
            "run_id": run_id,
            "season": season,
            "week": week,
            "decisions": len(decisions),
        },
    )


def install(namespace: dict[str, Any]) -> FastAPI:
    """Build the application and publish its names into ``namespace``.

    ``namespace`` is the module dict of the generated ``api/server.py``. Hooks
    are seeded with ``setdefault`` so a shim may override them before calling.
    """
    namespace.setdefault("read_dataframe", read_dataframe)
    namespace.setdefault("_run_agent_review_background", _run_agent_review_background)
    namespace.setdefault("_review_runs_in_flight", set())

    app = _build_app(namespace)

    namespace["app"] = app
    namespace["get_current_user"] = get_current_user
    namespace["BetCreate"] = BetCreate
    return app


def _build_app(ns: dict[str, Any]) -> FastAPI:
    app = FastAPI(
        title="NFL Algorithm API (local stack)",
        description=(
            "Local-development API assembled from tracked modules. Replace with the "
            "deployment-supplied api/server.py in production."
        ),
        version="2.1.0-local",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(pipeline_router)
    app.include_router(nba_router)

    def query(sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
        """Read through the namespace so ``api.server.read_dataframe`` patches apply."""
        return ns["read_dataframe"](sql, params)

    def load_value_rows(season: int, week: int) -> pd.DataFrame:
        visibility, visibility_params = value_visibility_scope()
        return query(
            VALUE_ROW_SQL.format(visibility=visibility),
            (*visibility_params, season, week),
        )

    def bet_payload(row: Any, *, why: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        edge = _num(row.get("edge_percentage"))
        payload: dict[str, Any] = {
            "player_id": _opt_str(row.get("player_id")) or "",
            "player_name": _opt_str(row.get("player_name")),
            "position": _opt_str(row.get("position")),
            "team": _opt_str(row.get("team")),
            "opponent": _opponent(row),
            "market": _opt_str(row.get("market")) or "",
            "sportsbook": _opt_str(row.get("sportsbook")) or "",
            "line": _num(row.get("line")),
            "price": _opt_int(row.get("price")) or 0,
            "side": _opt_str(row.get("side")),
            "mu": _num(row.get("mu")),
            "sigma": _num(row.get("sigma")),
            "p_win": _num(row.get("p_win")),
            "edge_percentage": edge,
            "expected_roi": _num(row.get("expected_roi")),
            "kelly_fraction": _num(row.get("kelly_fraction")),
            "stake": _num(row.get("stake")),
            "confidence_score": _opt_float(row.get("confidence_score")),
            "confidence_tier": _opt_str(row.get("confidence_tier")),
            "event_id": _opt_str(row.get("event_id")),
            "generated_at": _opt_str(row.get("generated_at")),
            "recommendation": "BET" if edge >= config.betting.min_edge_threshold else "PASS",
        }
        if why is not None:
            payload["why"] = why
        return payload

    # ── Metadata ────────────────────────────────────────────────────────────

    @app.get("/api/meta", tags=["nfl"])
    def get_meta() -> dict[str, Any]:
        """Weeks, sportsbooks, and markets present on the public card."""
        visibility, visibility_params = value_visibility_scope()
        try:
            weeks = query(
                f"""
                SELECT DISTINCT v.season, v.week
                FROM materialized_value_view v
                WHERE {visibility}
                ORDER BY v.season DESC, v.week DESC
                """,
                visibility_params,
            )
            books = query(
                f"""
                SELECT DISTINCT v.sportsbook
                FROM materialized_value_view v
                WHERE {visibility}
                ORDER BY v.sportsbook
                """,
                visibility_params,
            )
            markets = query(
                f"""
                SELECT DISTINCT v.market
                FROM materialized_value_view v
                WHERE {visibility}
                ORDER BY v.market
                """,
                visibility_params,
            )
        except Exception:
            # A failed query must not read as "no bets this week": that is the
            # difference between an outage and a genuinely empty card.
            logger.exception("Metadata query failed", extra={"event": "api.meta_failed"})
            raise HTTPException(status_code=500, detail="Metadata unavailable")

        return {
            "available_weeks": [
                {"season": int(row.season), "week": int(row.week)}
                for row in weeks.itertuples(index=False)
            ],
            "sportsbooks": [str(value) for value in books["sportsbook"].tolist()],
            "markets": [str(value) for value in markets["market"].tolist()],
        }

    # ── Value bets ──────────────────────────────────────────────────────────

    @app.get("/api/value-bets", tags=["nfl"])
    def get_value_bets(
        season: int = Query(...),
        week: int = Query(...),
        min_edge: float = Query(0.0),
        sportsbook: Optional[str] = Query(None),
        market: Optional[str] = Query(None),
        position: Optional[str] = Query(None),
        best_line_only: bool = Query(False),
        include_why: bool = Query(False),
        limit: int = Query(500, ge=1, le=2000),
    ) -> dict[str, Any]:
        """Return the published value card for a week."""
        cache_key = make_cache_key(
            "nfl-value-bets",
            season=season,
            week=week,
            min_edge=min_edge,
            sportsbook=sportsbook,
            market=market,
            position=position,
            best_line_only=best_line_only,
            include_why=include_why,
            limit=limit,
            # Demo and production must never share an entry: the two answers
            # differ in which rows are visible at all.
            demo_mode=bool(config.api.demo_mode),
        )
        cached = value_bets_cache.get(cache_key)
        if cached is not None:
            return cached

        df = load_value_rows(season, week)
        total_count = int(len(df))

        if not df.empty:
            df = df[df["edge_percentage"].astype(float) >= min_edge]
        if not df.empty and sportsbook:
            df = df[df["sportsbook"].astype(str).str.lower() == sportsbook.lower()]
        if not df.empty and market:
            df = df[df["market"].astype(str) == market]
        if not df.empty and position:
            df = df[df["position"].astype(str).str.upper() == position.upper()]
        if not df.empty and best_line_only:
            df = (
                df.sort_values("edge_percentage", ascending=False)
                .drop_duplicates(subset=["player_id", "market", "side"], keep="first")
                .reset_index(drop=True)
            )
        df = df.head(limit)

        bets: list[dict[str, Any]] = []
        for row in df.to_dict("records"):
            why = None
            if include_why:
                why = build_why_payload(
                    season, week, str(row.get("player_id")), str(row.get("market"))
                )
            bets.append(bet_payload(row, why=why))

        response = {
            "season": season,
            "week": week,
            "bets": bets,
            "total": len(bets),
            "total_count": total_count,
            "filtered_count": len(bets),
            "filters": {
                "min_edge": min_edge,
                "sportsbook": sportsbook,
                "market": market,
                "position": position,
                "best_line_only": best_line_only,
            },
        }
        value_bets_cache.set(cache_key, response)
        return response

    @app.get("/api/explain/{player_id}/{market}", tags=["nfl"])
    def explain_bet(
        player_id: str,
        market: str,
        season: int = Query(...),
        week: int = Query(...),
    ) -> dict[str, Any]:
        """Return the explainability payload for one player/market."""
        return {
            "season": season,
            "week": week,
            "player_id": player_id,
            "market": market,
            "why": build_why_payload(season, week, player_id, market),
        }

    # ── Performance and outcomes ────────────────────────────────────────────

    @app.get("/api/performance", tags=["nfl"])
    def get_performance(season: Optional[int] = Query(None)) -> dict[str, Any]:
        """Graded results by week, plus season-to-date totals."""
        sql = (
            "SELECT season, week, total_bets, wins, losses, pushes, profit_units, "
            "roi_pct, avg_edge, best_bet, worst_bet FROM weekly_performance"
        )
        params: tuple[Any, ...] = ()
        if season is not None:
            sql += " WHERE season = ?"
            params = (season,)
        sql += " ORDER BY season DESC, week DESC"

        weeks = [
            {
                "season": int(row[0]),
                "week": int(row[1]),
                "total_bets": int(row[2]),
                "wins": int(row[3]),
                "losses": int(row[4]),
                "pushes": int(row[5]),
                "profit_units": _num(row[6]),
                "roi_pct": _num(row[7]),
                "avg_edge": _num(row[8]),
                "best_bet": _opt_str(row[9]),
                "worst_bet": _opt_str(row[10]),
            }
            for row in fetchall(sql, params)
        ]

        total_bets = sum(week["total_bets"] for week in weeks)
        total_wins = sum(week["wins"] for week in weeks)
        total_losses = sum(week["losses"] for week in weeks)
        total_profit = sum(week["profit_units"] for week in weeks)
        decided = total_wins + total_losses

        return {
            "total_bets": total_bets,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "total_profit": round(total_profit, 2),
            "overall_roi": round(total_profit / decided * 100, 1) if decided else 0.0,
            "win_rate": round(total_wins / decided * 100, 1) if decided else 0.0,
            "weeks": weeks,
        }

    @app.get("/api/weekly-summary", tags=["nfl"])
    def get_weekly_summary(weeks: int = Query(4, ge=1, le=52)) -> dict[str, Any]:
        """The most recent graded weeks, newest first."""
        rows = fetchall(
            "SELECT season, week, total_bets, wins, losses, pushes, profit_units, "
            "roi_pct, avg_edge FROM weekly_performance "
            "ORDER BY season DESC, week DESC LIMIT ?",
            (weeks,),
        )
        return {
            "weeks": [
                {
                    "season": int(row[0]),
                    "week": int(row[1]),
                    "total_bets": int(row[2]),
                    "wins": int(row[3]),
                    "losses": int(row[4]),
                    "pushes": int(row[5]),
                    "profit_units": _num(row[6]),
                    "roi_pct": _num(row[7]),
                    "avg_edge": _num(row[8]),
                }
                for row in rows
            ]
        }

    @app.get("/api/outcomes", tags=["nfl"])
    def get_outcomes(season: int = Query(...), week: int = Query(...)) -> list[dict[str, Any]]:
        """Individual graded bets for a week."""
        rows = fetchall(
            "SELECT bet_id, player_name, market, line, actual_result, result, "
            "profit_units, confidence_tier FROM bet_outcomes "
            "WHERE season = ? AND week = ? ORDER BY profit_units DESC",
            (season, week),
        )
        return [
            {
                "bet_id": str(row[0]),
                "player_name": _opt_str(row[1]),
                "market": str(row[2]),
                "line": _num(row[3]),
                "actual_result": _opt_float(row[4]),
                "result": _opt_str(row[5]),
                "profit_units": _opt_float(row[6]),
                "confidence_tier": _opt_str(row[7]),
            }
            for row in rows
        ]

    @app.get("/api/health", tags=["operations"])
    def get_health(
        season: Optional[int] = Query(None), week: Optional[int] = Query(None)
    ) -> dict[str, Any]:
        """Feed freshness. Not a startup probe — use /livez and /readyz."""
        sql = "SELECT feed, season, week, as_of FROM feed_freshness"
        clauses: list[str] = []
        params: list[Any] = []
        if season is not None:
            clauses.append("season = ?")
            params.append(season)
        if week is not None:
            clauses.append("week = ?")
            params.append(week)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY feed"

        feeds = [
            {
                "feed": str(row[0]),
                "season": _opt_int(row[1]),
                "week": _opt_int(row[2]),
                "as_of": _opt_str(row[3]),
            }
            for row in fetchall(sql, tuple(params))
        ]

        if not feeds:
            overall = "unknown"
        else:
            now = datetime.now(timezone.utc)
            ages: list[float] = []
            for feed in feeds:
                stamp = pd.to_datetime(feed["as_of"], utc=True, errors="coerce")
                if pd.isna(stamp):
                    continue
                ages.append((now - stamp.to_pydatetime()).total_seconds() / 3600.0)
            overall = "healthy" if ages and max(ages) < 24 else "degraded"

        return {"feeds": feeds, "overall_status": overall}

    # ── Analytics ───────────────────────────────────────────────────────────

    @app.get("/api/analytics/edge-distribution", tags=["nfl"])
    def edge_distribution(
        season: int = Query(...), week: int = Query(...), bins: int = Query(8, ge=1, le=50)
    ) -> dict[str, Any]:
        """Histogram of edge percentages over the public card."""
        df = load_value_rows(season, week)
        if df.empty:
            return {"bins": [], "counts": []}

        edges = df["edge_percentage"].astype(float)
        upper = max(float(edges.max()), 0.01)
        width = upper / bins
        labels: list[str] = []
        counts: list[int] = []
        for index in range(bins):
            low = index * width
            high = upper if index == bins - 1 else (index + 1) * width
            in_bin = (edges >= low) & (edges <= high if index == bins - 1 else edges < high)
            labels.append(f"{low:.4f}-{high:.4f}")
            counts.append(int(in_bin.sum()))
        return {"bins": labels, "counts": counts}

    def group_stats(season: int, week: int, column: str) -> list[dict[str, Any]]:
        df = load_value_rows(season, week)
        if df.empty or column not in df.columns:
            return []
        frame = df.dropna(subset=[column])
        if frame.empty:
            return []
        grouped = frame.groupby(frame[column].astype(str), dropna=False)
        rows = [
            {
                column: str(key),
                "bet_count": int(len(chunk)),
                "avg_edge": round(_num(chunk["edge_percentage"].astype(float).mean()), 6),
                "avg_roi": round(_num(chunk["expected_roi"].astype(float).mean()), 6),
                "avg_p_win": round(_num(chunk["p_win"].astype(float).mean()), 6),
            }
            for key, chunk in grouped
        ]
        return sorted(rows, key=lambda row: row["bet_count"], reverse=True)

    @app.get("/api/analytics/by-position", tags=["nfl"])
    def analytics_by_position(season: int = Query(...), week: int = Query(...)) -> dict[str, Any]:
        """Card composition by player position."""
        return {"by_position": group_stats(season, week, "position")}

    @app.get("/api/analytics/by-market", tags=["nfl"])
    def analytics_by_market(season: int = Query(...), week: int = Query(...)) -> dict[str, Any]:
        """Card composition by prop market."""
        return {"by_market": group_stats(season, week, "market")}

    @app.get("/api/analytics/correlation", tags=["nfl"])
    def analytics_correlation(season: int = Query(...), week: int = Query(...)) -> dict[str, Any]:
        """Correlated prop groups and same-team stacks on the public card."""
        df = load_value_rows(season, week)
        if df.empty:
            return {"correlation_groups": [], "team_stacks": []}

        frame = df.reset_index(drop=True)
        tagged = detect_correlations(frame)

        groups: list[dict[str, Any]] = []
        for label, chunk in tagged.dropna(subset=["correlation_group"]).groupby(
            "correlation_group"
        ):
            groups.append(
                {
                    "group": str(label),
                    "type": str(label).rsplit("_", 1)[0],
                    "players": [
                        {
                            "player_id": _opt_str(row.get("player_id")) or "",
                            "player_name": _opt_str(row.get("player_name")),
                            "market": _opt_str(row.get("market")) or "",
                            "team": _opt_str(row.get("team")),
                        }
                        for row in chunk.to_dict("records")
                    ],
                    "combined_stake": round(_num(chunk["stake"].astype(float).sum()), 2),
                }
            )

        stacks = [
            {
                "team": team,
                "count": len(indices),
                "player_ids": [
                    _opt_str(frame.loc[index, "player_id"]) or "" for index in indices
                ],
            }
            for team, indices in detect_team_stacks(frame).items()
        ]
        stacks.sort(key=lambda item: item["count"], reverse=True)

        return {"correlation_groups": groups, "team_stacks": stacks}

    @app.get("/api/analytics/risk-summary", tags=["nfl"])
    def analytics_risk_summary(season: int = Query(...), week: int = Query(...)) -> dict[str, Any]:
        """Bankroll exposure by team and game, with guardrail breaches."""
        bankroll = float(config.betting.bankroll)
        guardrails = {
            "max_team_exposure": float(config.risk.max_team_exposure),
            "max_game_exposure": float(config.risk.max_game_exposure),
            "max_player_exposure": float(config.risk.max_player_exposure),
        }

        df = load_value_rows(season, week)
        if df.empty:
            return {
                "total_stake": 0.0,
                "bankroll": bankroll,
                "team_exposure": [],
                "game_exposure": [],
                "guardrails": guardrails,
                "warnings": [],
            }

        frame = df.reset_index(drop=True)
        frame["stake"] = frame["stake"].astype(float)

        def exposure(column: str, key: str) -> list[dict[str, Any]]:
            if column not in frame.columns:
                return []
            totals = frame.dropna(subset=[column]).groupby(frame[column].astype(str))["stake"].sum()
            rows = [
                {
                    key: str(name),
                    "stake": round(float(total), 2),
                    "fraction": round(float(total) / bankroll, 4) if bankroll else 0.0,
                }
                for name, total in totals.items()
            ]
            return sorted(rows, key=lambda row: row["stake"], reverse=True)

        flagged = compute_exposure(frame, bankroll)
        warnings = [
            {
                "player_id": _opt_str(row.get("player_id")) or "",
                "player_name": _opt_str(row.get("player_name")),
                "warning": str(row.get("exposure_warning")),
            }
            for row in flagged.to_dict("records")
            if row.get("exposure_warning")
        ]

        return {
            "total_stake": round(float(frame["stake"].sum()), 2),
            "bankroll": bankroll,
            "team_exposure": exposure("team", "team"),
            "game_exposure": exposure("event_id", "game"),
            "guardrails": guardrails,
            "warnings": warnings,
        }

    # ── Exports ─────────────────────────────────────────────────────────────

    def filtered_card(season: int, week: int, min_edge: float) -> pd.DataFrame:
        df = load_value_rows(season, week)
        if df.empty:
            return df
        return df[df["edge_percentage"].astype(float) >= min_edge]

    @app.get("/api/export/csv", tags=["nfl"])
    def export_csv(
        season: int = Query(...), week: int = Query(...), min_edge: float = Query(0.0)
    ) -> Response:
        """Download the public card as CSV."""
        df = filtered_card(season, week, min_edge)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in df.to_dict("records"):
            payload = bet_payload(row)
            payload["season"] = season
            payload["week"] = week
            writer.writerow({column: payload.get(column, "") for column in CSV_COLUMNS})

        filename = f"value_bets_{season}_w{week:02d}.csv"
        return Response(
            content=buffer.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/export/bundle", tags=["nfl"])
    def export_bundle(
        season: int = Query(...), week: int = Query(...), min_edge: float = Query(0.0)
    ) -> dict[str, Any]:
        """Download the public card plus the run that published it."""
        df = filtered_card(season, week, min_edge)
        records = df.to_dict("records")
        bets = [bet_payload(row) for row in records]

        # Prefer the run that actually published these rows over the newest run
        # for the week: an export should describe the card it contains.
        publishers = [
            str(row["published_run_id"])
            for row in records
            if row.get("published_run_id") not in (None, "")
            and not pd.isna(row.get("published_run_id"))
        ]
        run_row = None
        if publishers:
            run_row = fetchone(
                "SELECT run_id, season, week, status, stages_requested, stages_completed, "
                "started_at, finished_at FROM pipeline_runs WHERE run_id = ?",
                (max(set(publishers), key=publishers.count),),
            )
        if run_row is None:
            run_row = fetchone(
                "SELECT run_id, season, week, status, stages_requested, stages_completed, "
                "started_at, finished_at FROM pipeline_runs "
                "WHERE season = ? AND week = ? ORDER BY started_at DESC LIMIT 1",
                (season, week),
            )

        pipeline_run = None
        if run_row is not None:
            pipeline_run = {
                "run_id": str(run_row[0]),
                "season": int(run_row[1]),
                "week": int(run_row[2]),
                "status": str(run_row[3]),
                "stages_requested": _opt_int(run_row[4]),
                "stages_completed": _opt_int(run_row[5]),
                "started_at": _opt_str(run_row[6]),
                "finished_at": _opt_str(run_row[7]),
            }

        return {
            "season": season,
            "week": week,
            "total_bets": len(bets),
            "bets": bets,
            "pipeline_run": pipeline_run,
            "exported_at": _now(),
        }

    # ── Agent review ────────────────────────────────────────────────────────

    def load_run(run_id: str) -> tuple[Any, ...]:
        row = fetchone(
            "SELECT run_id, season, week, status FROM pipeline_runs WHERE run_id = ?",
            (run_id,),
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Pipeline run not found")
        return row

    def reviewable_bet_count(run_id: str, season: int, week: int) -> int:
        visibility, visibility_params = value_visibility_scope()
        # Demo mode reviews fixture rows explicitly, which carry no
        # published_run_id, so the run linkage only applies in production.
        run_clause = "1 = 1" if config.api.demo_mode else "v.published_run_id = ?"
        run_params: tuple[Any, ...] = () if config.api.demo_mode else (run_id,)
        row = fetchone(
            f"""
            SELECT COUNT(*)
            FROM materialized_value_view v
            WHERE {visibility}
              AND {run_clause}
              AND v.season = ? AND v.week = ?
            """,
            (*visibility_params, *run_params, season, week),
        )
        return int(row[0]) if row else 0

    @app.post("/api/run/{run_id}/review", tags=["nfl"])
    def request_agent_review(
        run_id: str,
        background_tasks: BackgroundTasks,
        season: int = Query(...),
        week: int = Query(...),
        operator: str = Depends(require_pipeline_operator),
    ) -> dict[str, Any]:
        """Queue an agent review of a published card."""
        load_run(run_id)

        if run_id in ns["_review_runs_in_flight"]:
            raise HTTPException(
                status_code=409, detail="Agent review is already running for this run"
            )
        if reviewable_bet_count(run_id, season, week) == 0:
            raise HTTPException(status_code=409, detail="Run has no reviewable published bets")

        ns["_review_runs_in_flight"].add(run_id)

        def invoke() -> None:
            try:
                ns["_run_agent_review_background"](run_id, season, week)
            except Exception:
                logger.exception(
                    "Agent review failed",
                    extra={"event": "api.agent_review_failed", "run_id": run_id},
                )
            finally:
                ns["_review_runs_in_flight"].discard(run_id)

        background_tasks.add_task(invoke)
        return {
            "run_id": run_id,
            "season": season,
            "week": week,
            "review_status": "started",
            "message": "Agent review started",
        }

    @app.get("/api/run/{run_id}/review-status", tags=["nfl"])
    def agent_review_status(
        run_id: str,
        season: int = Query(...),
        week: int = Query(...),
        reader: str = Depends(require_pipeline_reader),
    ) -> dict[str, Any]:
        """Report whether agents have recorded decisions for a week."""
        load_run(run_id)
        row = fetchone(
            "SELECT COUNT(*), MAX(decided_at) FROM agent_decisions WHERE season = ? AND week = ?",
            (season, week),
        )
        count = int(row[0]) if row else 0
        return {
            "run_id": run_id,
            "reviewed": count > 0,
            "reviewed_at": _opt_str(row[1]) if row else None,
            "decision_count": count,
        }

    # ── Authentication ──────────────────────────────────────────────────────

    @app.post("/api/auth/register", tags=["auth"])
    def register(payload: UserCreate) -> dict[str, Any]:
        """Create an account and open a session for it."""
        try:
            create_user(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        session = authenticate_user(UserLogin(email=payload.email, password=payload.password))
        if session is None:
            raise HTTPException(status_code=500, detail="Account created but session failed")
        return session

    @app.post("/api/auth/login", tags=["auth"])
    def login(payload: UserLogin) -> dict[str, Any]:
        """Exchange credentials for a session token."""
        session = authenticate_user(payload)
        if session is None:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        return session

    @app.post("/api/auth/logout", tags=["auth"])
    def logout(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    ) -> dict[str, Any]:
        """Invalidate the presented session token."""
        if credentials is None or not credentials.credentials:
            raise HTTPException(status_code=401, detail="Not authenticated")
        logout_user(credentials.credentials)
        return {"message": "Logged out"}

    @app.get("/api/auth/me", tags=["auth"])
    def whoami(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
        """Return the signed-in user."""
        return current_user

    # ── User bet slip ───────────────────────────────────────────────────────

    @app.post("/api/user/bets", tags=["user"])
    def record_bet(
        payload: BetCreate,
        current_user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Record a bet on the signed-in user's slip."""
        bet_id = f"bet_{secrets.token_hex(12)}"
        execute(
            """
            INSERT INTO user_bets (
                id, user_id, season, week, player_id, player_name, market, sportsbook,
                side, line, price, stake_units, stake_dollars, model_edge,
                confidence_tier, placed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                bet_id,
                current_user.user_id,
                payload.season,
                payload.week,
                payload.player_id,
                payload.player_name,
                payload.market,
                payload.sportsbook,
                payload.side,
                payload.line,
                payload.price,
                payload.stake_units,
                payload.stake_dollars,
                payload.model_edge,
                payload.confidence_tier,
                _now(),
            ),
        )
        return {"bet_id": bet_id, "message": "Bet recorded"}

    @app.get("/api/user/bets", tags=["user"])
    def list_user_bets(
        season: Optional[int] = Query(None),
        week: Optional[int] = Query(None),
        limit: int = Query(200, ge=1, le=1000),
        current_user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """List the signed-in user's recorded bets, newest first."""
        sql = (
            "SELECT id, season, week, player_id, player_name, market, sportsbook, side, "
            "line, price, stake_units, model_edge, placed_at, actual_result, outcome, "
            "profit_units, graded_at FROM user_bets WHERE user_id = ?"
        )
        params: list[Any] = [current_user.user_id]
        if season is not None:
            sql += " AND season = ?"
            params.append(season)
        if week is not None:
            sql += " AND week = ?"
            params.append(week)
        sql += " ORDER BY placed_at DESC LIMIT ?"
        params.append(limit)

        bets = [
            {
                "bet_id": str(row[0]),
                "season": _opt_int(row[1]),
                "week": _opt_int(row[2]),
                "player_id": str(row[3]),
                "player_name": _opt_str(row[4]),
                "market": str(row[5]),
                "sportsbook": str(row[6]),
                "side": _opt_str(row[7]),
                "line": _num(row[8]),
                "price": _opt_int(row[9]) or 0,
                "stake": _num(row[10]),
                "edge_at_placement": _opt_float(row[11]),
                "placed_at": _opt_str(row[12]) or "",
                "actual_result": _opt_float(row[13]),
                "result": _opt_str(row[14]),
                "profit_units": _opt_float(row[15]),
                "settled_at": _opt_str(row[16]),
            }
            for row in fetchall(sql, tuple(params))
        ]
        return {"bets": bets, "total": len(bets)}

    @app.get("/api/user/stats", tags=["user"])
    def user_stats(current_user: UserResponse = Depends(get_current_user)) -> dict[str, Any]:
        """Aggregate results across the signed-in user's slip."""
        row = fetchone(
            """
            SELECT COUNT(*),
                   SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN outcome = 'push' THEN 1 ELSE 0 END),
                   SUM(COALESCE(profit_units, 0)),
                   AVG(model_edge),
                   SUM(COALESCE(stake_units, 0))
            FROM user_bets WHERE user_id = ?
            """,
            (current_user.user_id,),
        )
        total = int(row[0]) if row and row[0] else 0
        wins = int(row[1] or 0) if row else 0
        losses = int(row[2] or 0) if row else 0
        pushes = int(row[3] or 0) if row else 0
        profit = _num(row[4]) if row else 0.0
        avg_edge = _opt_float(row[5]) if row else None
        staked = _num(row[6]) if row else 0.0

        return {
            "total_bets": total,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "total_profit": round(profit, 2),
            "avg_edge": round(avg_edge, 6) if avg_edge is not None else 0.0,
            "roi": round(profit / staked * 100, 2) if staked else 0.0,
        }

    return app


__all__ = ["BetCreate", "get_current_user", "install"]
