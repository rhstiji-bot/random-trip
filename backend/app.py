from __future__ import annotations

import json
import os
import csv
import random
import hashlib
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple, Set

import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# =========================
# App
# =========================

app = FastAPI(title="Random Trip API (MVP)", version="1.0.0")

# CORS (dev)
app.add_middleware(
    CORSMiddleware,
    
    allow_origins=["*"],

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.getenv("APP_DB_PATH", "app.db")
SEED_PLACES_CSV = os.getenv("SEED_PLACES_CSV", "seed_places.csv")


# =========================
# DB helpers
# =========================

def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    schema = """
    PRAGMA journal_mode=WAL;

    CREATE TABLE IF NOT EXISTS places (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      type TEXT NOT NULL,            -- STATION / AREA / SPOT
      name TEXT NOT NULL,
      region_code TEXT,
      lat REAL,
      lon REAL,
      tags_json TEXT NOT NULL DEFAULT '[]'
    );

    CREATE TABLE IF NOT EXISTS trips (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      origin_place_id INTEGER NOT NULL,
      start_date TEXT NOT NULL,
      end_date TEXT NOT NULL,
      budget_yen INTEGER NOT NULL,
      seed INTEGER
    );

    CREATE TABLE IF NOT EXISTS trip_candidates (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      trip_id INTEGER NOT NULL,
      destination_place_id INTEGER NOT NULL,
      total_cost_yen INTEGER NOT NULL,
      score REAL NOT NULL,
      reason_json TEXT NOT NULL DEFAULT '{}'
    );

    CREATE INDEX IF NOT EXISTS idx_trip_candidates_trip ON trip_candidates(trip_id);

    CREATE TABLE IF NOT EXISTS routes (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      trip_candidate_id INTEGER NOT NULL,
      direction TEXT NOT NULL, -- OUTBOUND / INBOUND
      provider TEXT NOT NULL,
      duration_min INTEGER NOT NULL,
      price_yen INTEGER NOT NULL,
      price_after_discounts_yen INTEGER NOT NULL,
      details_json TEXT NOT NULL DEFAULT '{}'
    );

    CREATE INDEX IF NOT EXISTS idx_routes_candidate ON routes(trip_candidate_id);

    CREATE TABLE IF NOT EXISTS lodgings (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      trip_candidate_id INTEGER NOT NULL,
      provider TEXT NOT NULL,
      name TEXT NOT NULL,
      price_per_night_yen INTEGER NOT NULL,
      bargain_flag INTEGER NOT NULL DEFAULT 0,
      url TEXT,
      details_json TEXT NOT NULL DEFAULT '{}'
    );

    CREATE INDEX IF NOT EXISTS idx_lodgings_candidate ON lodgings(trip_candidate_id);

    CREATE TABLE IF NOT EXISTS reroll_history (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      trip_id INTEGER NOT NULL,
      prev_seed INTEGER,
      new_seed INTEGER NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      snapshot_json TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_reroll_history_trip ON reroll_history(trip_id);

    -- Cache: generic
    CREATE TABLE IF NOT EXISTS cache_api_results (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      provider TEXT NOT NULL,
      request_hash TEXT NOT NULL,
      response_json TEXT NOT NULL,
      fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
      expires_at TEXT NOT NULL
    );

    CREATE UNIQUE INDEX IF NOT EXISTS uq_cache_provider_hash
    ON cache_api_results(provider, request_hash);

    CREATE INDEX IF NOT EXISTS idx_cache_expires
    ON cache_api_results(expires_at);

    -- Distance cache
    CREATE TABLE IF NOT EXISTS distance_cache (
      origin_place_id INTEGER NOT NULL,
      dest_place_id INTEGER NOT NULL,
      distance_km REAL NOT NULL,
      computed_at TEXT NOT NULL DEFAULT (datetime('now')),
      PRIMARY KEY (origin_place_id, dest_place_id)
    );

    CREATE INDEX IF NOT EXISTS idx_distance_cache_origin ON distance_cache(origin_place_id);
    CREATE INDEX IF NOT EXISTS idx_distance_cache_dest ON distance_cache(dest_place_id);

    -- Selections
    CREATE TABLE IF NOT EXISTS trip_selections (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      trip_id INTEGER NOT NULL,
      destination_place_id INTEGER NOT NULL,
      selected_at TEXT NOT NULL DEFAULT (datetime('now')),
      note TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_trip_selections_trip ON trip_selections(trip_id);

    CREATE UNIQUE INDEX IF NOT EXISTS uq_trip_selections_trip_dest
    ON trip_selections(trip_id, destination_place_id);
    """
    with connect() as conn:
        conn.executescript(schema)


def seed_places_if_empty() -> None:
    with connect() as conn:
        n = conn.execute("SELECT COUNT(*) AS c FROM places").fetchone()["c"]
        if int(n) > 0:
            return

    if not os.path.exists(SEED_PLACES_CSV):
        # No seed file. Leave empty.
        return

    with open(SEED_PLACES_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    def ffloat(x: Any) -> Optional[float]:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None

    insert_rows = []
    for r in rows:
        insert_rows.append((
            (r.get("type") or "").strip(),
            (r.get("name") or "").strip(),
            (r.get("region_code") or None),
            ffloat(r.get("lat")),
            ffloat(r.get("lon")),
            (r.get("tags_json") or "[]")
        ))

    with connect() as conn:
        conn.executemany(
            "INSERT INTO places(type, name, region_code, lat, lon, tags_json) VALUES(?,?,?,?,?,?)",
            insert_rows
        )


@app.on_event("startup")
def _startup() -> None:
    init_db()
    seed_places_if_empty()
    cache_purge_expired()


# =========================
# Models
# =========================

@dataclass
class Place:
    id: int
    type: str
    name: str
    region_code: Optional[str]
    lat: float
    lon: float
    tags: List[str]


class RouteOut(BaseModel):
    direction: str
    provider: str
    duration_min: int
    price_yen: int
    price_after_discounts_yen: int
    details: Dict[str, Any] = Field(default_factory=dict)


class LodgingOut(BaseModel):
    provider: str
    name: str
    price_per_night_yen: int
    bargain_flag: bool
    url: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class CandidateOut(BaseModel):
    destination: Dict[str, Any]
    total_cost_yen: int
    score: float
    reason: Dict[str, Any] = Field(default_factory=dict)
    routes: List[RouteOut]
    lodging: LodgingOut


class GenerateTripRequest(BaseModel):
    origin_place_id: int
    start_date: date
    end_date: date
    budget_yen: int = Field(ge=0)

    # constraints
    max_total_travel_minutes: Optional[int] = None
    min_destination_distance_km: Optional[float] = 0

    # tags
    preferred_tags: Optional[List[str]] = None
    excluded_tags: Optional[List[str]] = None

    # distance band params
    near_km: Optional[float] = 80
    mid_km: Optional[float] = 250

    # seed
    seed: Optional[int] = None

    # reroll-only behavior
    avoid_selected: Optional[bool] = True


class GenerateTripResponse(BaseModel):
    trip_id: int
    seed: Optional[int]
    candidates: List[CandidateOut]
    cache_hit: Optional[bool] = Field(False, description="生成キャッシュにヒットしたか")


class SelectCandidateRequest(BaseModel):
    destination_place_id: int
    note: Optional[str] = None


# =========================
# Generic cache
# =========================

def _hash_request(provider: str, payload: dict) -> str:
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256((provider + "|" + s).encode("utf-8")).hexdigest()


def cache_get(provider: str, request_hash: str) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute(
            "SELECT response_json FROM cache_api_results "
            "WHERE provider = ? AND request_hash = ? AND expires_at > datetime('now')",
            (provider, request_hash),
        ).fetchone()
    if not row:
        return None
    return json.loads(row["response_json"])


def cache_set(provider: str, request_hash: str, response: dict, ttl_seconds: int = 3600) -> None:
    expires_expr = f"datetime('now', '+{int(ttl_seconds)} seconds')"
    with connect() as conn:
        conn.execute(
            "INSERT INTO cache_api_results(provider, request_hash, response_json, expires_at) "
            f"VALUES(?,?,?,{expires_expr}) "
            "ON CONFLICT(provider, request_hash) DO UPDATE SET "
            "response_json=excluded.response_json, fetched_at=datetime('now'), expires_at=excluded.expires_at",
            (provider, request_hash, json.dumps(response, ensure_ascii=False)),
        )


def cache_purge_expired() -> None:
    with connect() as conn:
        conn.execute("DELETE FROM cache_api_results WHERE expires_at <= datetime('now')")


def cache_stats() -> dict:
    with connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM cache_api_results").fetchone()["c"]
        valid = conn.execute(
            "SELECT COUNT(*) AS c FROM cache_api_results WHERE expires_at > datetime('now')"
        ).fetchone()["c"]
        expired = conn.execute(
            "SELECT COUNT(*) AS c FROM cache_api_results WHERE expires_at <= datetime('now')"
        ).fetchone()["c"]
        by_provider = conn.execute(
            "SELECT provider, COUNT(*) AS c FROM cache_api_results GROUP BY provider ORDER BY c DESC"
        ).fetchall()

    return {
        "total": int(total),
        "valid": int(valid),
        "expired": int(expired),
        "by_provider": [{"provider": r["provider"], "count": int(r["c"])} for r in by_provider],
    }


# =========================
# Distance
# =========================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # lightweight haversine
    from math import radians, sin, cos, asin, sqrt
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def cached_distance_km(origin: Place, dest: Place) -> float:
    with connect() as conn:
        row = conn.execute(
            "SELECT distance_km FROM distance_cache WHERE origin_place_id = ? AND dest_place_id = ?",
            (origin.id, dest.id),
        ).fetchone()
        if row:
            return float(row["distance_km"])

    km = haversine_km(origin.lat, origin.lon, dest.lat, dest.lon)

    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO distance_cache(origin_place_id, dest_place_id, distance_km) VALUES(?,?,?)",
            (origin.id, dest.id, float(km)),
        )

    return float(km)


# =========================
# Places
# =========================

def get_place(place_id: int) -> Place:
    with connect() as conn:
        row = conn.execute("SELECT * FROM places WHERE id = ?", (place_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"place_id {place_id} not found")

    tags = json.loads(row["tags_json"]) if row["tags_json"] else []
    return Place(
        id=int(row["id"]),
        type=row["type"],
        name=row["name"],
        region_code=row["region_code"],
        lat=float(row["lat"] or 0.0),
        lon=float(row["lon"] or 0.0),
        tags=tags,
    )


def list_destination_places(exclude_id: int) -> List[Place]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM places WHERE type IN ('AREA','SPOT') AND id != ?",
            (exclude_id,),
        ).fetchall()

    out: List[Place] = []
    for r in rows:
        out.append(
            Place(
                id=int(r["id"]),
                type=r["type"],
                name=r["name"],
                region_code=r["region_code"],
                lat=float(r["lat"] or 0.0),
                lon=float(r["lon"] or 0.0),
                tags=json.loads(r["tags_json"]) if r["tags_json"] else [],
            )
        )
    return out


# =========================
# Dummy providers (route / lodging)
# =========================

def dummy_route(origin: Place, dest: Place, direction: str) -> Dict[str, Any]:
    provider = "DUMMY_ROUTE"
    payload = {"origin_id": origin.id, "dest_id": dest.id, "direction": direction}
    h = _hash_request(provider, payload)
    cached = cache_get(provider, h)
    if cached is not None:
        return cached

    km = cached_distance_km(origin, dest)
    duration_h = km / 70.0
    duration_min = int(duration_h * 60 + 30)
    price = int(km * 25 + 800)

    discount_rate = random.choice([0.0, 0.05, 0.1, 0.15])
    after = int(price * (1.0 - discount_rate))

    res = {
        "direction": direction,
        "provider": provider,
        "duration_min": max(duration_min, 20),
        "price_yen": max(price, 500),
        "price_after_discounts_yen": max(after, 400),
        "details": {
            "distance_km": round(km, 1),
            "discount_rate": discount_rate,
            "note": "概算（外部経路API未接続）",
        },
    }
    cache_set(provider, h, res, ttl_seconds=24 * 3600)
    return res


def dummy_lodging(dest: Place, nights: int, seed: int) -> Dict[str, Any]:
    provider = "DUMMY_LODGING"
    payload = {"dest_id": dest.id, "nights": nights, "seed": seed}
    h = _hash_request(provider, payload)
    cached = cache_get(provider, h)
    if cached is not None:
        return cached

    base = random.choice([7000, 11000, 16000, 22000])
    bargain = random.random() < 0.25
    price = int(base * (0.85 if bargain else 1.0))

    res = {
        "provider": provider,
        "name": f"{dest.name} ホテル（概算）",
        "price_per_night_yen": price,
        "bargain_flag": bargain,
        "url": None,
        "details": {"nights": nights, "note": "概算（宿API未接続）"},
    }
    cache_set(provider, h, res, ttl_seconds=24 * 3600)
    return res


# =========================
# Scoring & selection
# =========================

def dynamic_sample_size(req: GenerateTripRequest, max_n: int = 30, base: int = 10) -> int:
    n = base
    if req.preferred_tags:
        n += 10
    if req.max_total_travel_minutes is not None and req.max_total_travel_minutes <= 600:
        n += 10
    elif req.max_total_travel_minutes is not None and req.max_total_travel_minutes <= 720:
        n += 5
    if req.min_destination_distance_km is not None and req.min_destination_distance_km >= 80:
        n += 5
    return min(n, max_n)


def distance_band_quota(max_total_travel_minutes: Optional[int]) -> Tuple[int, int, int]:
    if max_total_travel_minutes is None:
        return (3, 4, 3)
    m = int(max_total_travel_minutes)
    if m <= 480:
        return (5, 4, 1)
    if m <= 600:
        return (4, 4, 2)
    if m <= 720:
        return (3, 4, 3)
    return (2, 4, 4)


def sample_destinations_by_distance_band(
    origin: Place,
    destinations: List[Place],
    rnd: random.Random,
    total_n: int,
    near_km: float = 80.0,
    mid_km: float = 250.0,
    quota_near: int = 3,
    quota_mid: int = 4,
    quota_far: int = 3,
) -> List[Place]:
    near, mid, far = [], [], []
    for d in destinations:
        km = cached_distance_km(origin, d)
        if km < near_km:
            near.append(d)
        elif km < mid_km:
            mid.append(d)
        else:
            far.append(d)

    rnd.shuffle(near)
    rnd.shuffle(mid)
    rnd.shuffle(far)

    picked: List[Place] = []

    def take(src: List[Place], n: int):
        nonlocal picked
        if n <= 0 or not src:
            return
        picked.extend(src[:n])
        del src[:n]

    take(near, min(quota_near, len(near)))
    take(mid, min(quota_mid, len(mid)))
    take(far, min(quota_far, len(far)))

    remaining = total_n - len(picked)
    if remaining > 0:
        pool = near + mid + far
        rnd.shuffle(pool)
        picked.extend(pool[:remaining])

    return picked[:total_n]


def diversity_penalty(selected: List[Place], candidate: Place) -> float:
    if not selected:
        return 0.0
    cand_tags = set(candidate.tags or [])
    if not cand_tags:
        return 0.0

    penalty = 0.0
    for p in selected:
        overlap = cand_tags.intersection(set(p.tags or []))
        if overlap:
            penalty += 5.0 * len(overlap)
    return min(penalty, 20.0)


def far_enough(selected: List[Place], candidate: Place, min_km: float) -> bool:
    if not selected or min_km <= 0:
        return True
    for p in selected:
        km = cached_distance_km(p, candidate)
        if km < min_km:
            return False
    return True


def score_candidate(
    req: GenerateTripRequest,
    total_cost_yen: int,
    total_travel_minutes: int,
    discount_yen: int,
) -> float:
    # Higher is better
    score = 100.0

    # budget fit
    if total_cost_yen <= req.budget_yen:
        score += 20.0
    else:
        over = total_cost_yen - req.budget_yen
        score -= min(50.0, over / 500.0)  # -1 per 500 yen over (capped)

    # travel time
    if req.max_total_travel_minutes is not None:
        if total_travel_minutes <= req.max_total_travel_minutes:
            score += 10.0
        else:
            score -= min(60.0, (total_travel_minutes - req.max_total_travel_minutes) / 5.0)

    # discount
    score += min(20.0, discount_yen / 500.0)

    return float(score)


def select_top_k_with_diversity(
    evaluated: List[tuple],
    k: int,
    min_km: float
) -> List[tuple]:
    if not evaluated:
        return []

    evaluated_sorted = sorted(evaluated, key=lambda x: x[0], reverse=True)

    selected_places: List[Place] = []
    selected_items: List[tuple] = []

    # pass1: enforce distance
    for item in evaluated_sorted:
        base_score, dest, out_r, in_r, lodging, total, reason = item

        if not far_enough(selected_places, dest, min_km):
            continue

        pen = diversity_penalty(selected_places, dest)
        adjusted = base_score - pen
        reason["diversity_penalty"] = pen
        reason["min_distance_km"] = min_km

        selected_places.append(dest)
        selected_items.append((adjusted, dest, out_r, in_r, lodging, total, reason))
        if len(selected_items) >= k:
            break

    # pass2: fill ignoring distance
    if len(selected_items) < k:
        for item in evaluated_sorted:
            base_score, dest, out_r, in_r, lodging, total, reason = item
            if any(dest.id == p.id for p in selected_places):
                continue
            pen = diversity_penalty(selected_places, dest)
            adjusted = base_score - pen
            reason["diversity_penalty"] = pen
            reason["min_distance_km"] = 0

            selected_places.append(dest)
            selected_items.append((adjusted, dest, out_r, in_r, lodging, total, reason))
            if len(selected_items) >= k:
                break

    return sorted(selected_items, key=lambda x: x[0], reverse=True)[:k]


# =========================
# Reroll history & persistence
# =========================

def clear_trip_candidates(trip_id: int) -> None:
    with connect() as conn:
        cand_rows = conn.execute(
            "SELECT id FROM trip_candidates WHERE trip_id = ?",
            (trip_id,),
        ).fetchall()
        cand_ids = [int(r["id"]) for r in cand_rows]
        if cand_ids:
            q = ",".join(["?"] * len(cand_ids))
            conn.execute(f"DELETE FROM routes WHERE trip_candidate_id IN ({q})", tuple(cand_ids))
            conn.execute(f"DELETE FROM lodgings WHERE trip_candidate_id IN ({q})", tuple(cand_ids))
        conn.execute("DELETE FROM trip_candidates WHERE trip_id = ?", (trip_id,))


def persist_candidates(trip_id: int, candidates: List[dict]) -> None:
    clear_trip_candidates(trip_id)
    with connect() as conn:
        for c in candidates:
            dest_id = int((c["destination"] or {})["id"])
            total = int(c["total_cost_yen"])
            score = float(c["score"])
            reason = c.get("reason", {})

            cand_id = conn.execute(
                "INSERT INTO trip_candidates(trip_id, destination_place_id, total_cost_yen, score, reason_json) "
                "VALUES(?,?,?,?,?)",
                (trip_id, dest_id, total, score, json.dumps(reason, ensure_ascii=False)),
            ).lastrowid

            for r in c.get("routes", []):
                conn.execute(
                    "INSERT INTO routes(trip_candidate_id, direction, provider, duration_min, price_yen, "
                    "price_after_discounts_yen, details_json) VALUES(?,?,?,?,?,?,?)",
                    (
                        cand_id,
                        r["direction"],
                        r["provider"],
                        int(r["duration_min"]),
                        int(r["price_yen"]),
                        int(r["price_after_discounts_yen"]),
                        json.dumps(r.get("details", {}), ensure_ascii=False),
                    ),
                )

            l = c.get("lodging", {})
            conn.execute(
                "INSERT INTO lodgings(trip_candidate_id, provider, name, price_per_night_yen, bargain_flag, url, details_json) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    cand_id,
                    l.get("provider", "UNKNOWN"),
                    l.get("name", "UNKNOWN"),
                    int(l.get("price_per_night_yen", 0)),
                    1 if l.get("bargain_flag") else 0,
                    l.get("url"),
                    json.dumps(l.get("details", {}), ensure_ascii=False),
                ),
            )


def insert_reroll_history(trip_id: int, prev_seed: Optional[int], new_seed: int, candidates: List[dict]) -> None:
    snap = {"candidates": candidates}
    with connect() as conn:
        conn.execute(
            "INSERT INTO reroll_history(trip_id, prev_seed, new_seed, snapshot_json) VALUES(?,?,?,?)",
            (trip_id, prev_seed, new_seed, json.dumps(snap, ensure_ascii=False)),
        )


def selected_destination_ids(trip_id: int) -> Set[int]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT destination_place_id FROM trip_selections WHERE trip_id = ?",
            (trip_id,),
        ).fetchall()
    return {int(r["destination_place_id"]) for r in rows}


# =========================
# Trip generate cache payload
# =========================

def trip_generate_cache_payload(req: GenerateTripRequest, selected_ids: Optional[List[int]] = None) -> dict:
    return {
        "origin_place_id": req.origin_place_id,
        "start_date": req.start_date.isoformat(),
        "end_date": req.end_date.isoformat(),
        "budget_yen": int(req.budget_yen),
        "seed": int(req.seed) if req.seed is not None else None,
        "max_total_travel_minutes": req.max_total_travel_minutes,
        "min_destination_distance_km": float(req.min_destination_distance_km or 0),
        "near_km": float(req.near_km or 80),
        "mid_km": float(req.mid_km or 250),
        "preferred_tags": sorted(req.preferred_tags) if req.preferred_tags else [],
        "excluded_tags": sorted(req.excluded_tags) if req.excluded_tags else [],
        "avoid_selected": bool(req.avoid_selected if req.avoid_selected is not None else True),
        "selected_destination_ids": sorted(selected_ids) if selected_ids else [],
    }


# =========================
# Core generate functions
# =========================

def _filter_by_tags(req: GenerateTripRequest, places: List[Place]) -> List[Place]:
    pref = req.preferred_tags or []
    excl = req.excluded_tags or []
    pref_set = set(pref)
    excl_set = set(excl)

    def ok(p: Place) -> bool:
        pt = set(p.tags or [])
        if excl_set and pt.intersection(excl_set):
            return False
        if pref_set:
            return bool(pt.intersection(pref_set))
        return True

    return [p for p in places if ok(p)]


def _build_candidates_for_dest(req: GenerateTripRequest, origin: Place, dest: Place, seed: int) -> CandidateOut:
    n_nights = max(0, (req.end_date - req.start_date).days)
    out_r = dummy_route(origin, dest, "OUTBOUND")
    in_r = dummy_route(origin, dest, "INBOUND")
    lodging = dummy_lodging(dest, n_nights, seed)

    travel_minutes = int(out_r["duration_min"]) + int(in_r["duration_min"])
    travel_cost = int(out_r["price_after_discounts_yen"]) + int(in_r["price_after_discounts_yen"])
    lodging_cost = int(lodging["price_per_night_yen"]) * n_nights
    total = travel_cost + lodging_cost

    discount_yen = (int(out_r["price_yen"]) - int(out_r["price_after_discounts_yen"])) + \
                   (int(in_r["price_yen"]) - int(in_r["price_after_discounts_yen"]))

    score = score_candidate(req, total, travel_minutes, discount_yen)

    reason = {
        "travel_minutes": travel_minutes,
        "discount_yen": discount_yen,
        "n_nights": n_nights,
        "sample_size": None,  # filled by caller if needed
    }

    return CandidateOut(
        destination={"id": dest.id, "name": dest.name, "region_code": dest.region_code},
        total_cost_yen=total,
        score=score,
        reason=reason,
        routes=[
            RouteOut(**out_r),
            RouteOut(**in_r),
        ],
        lodging=LodgingOut(**lodging),
    )


def generate_trip(req: GenerateTripRequest) -> GenerateTripResponse:
    origin = get_place(req.origin_place_id)

    # Seed
    seed = int(req.seed) if req.seed is not None else random.randint(1, 2_147_483_647)
    rnd = random.Random(seed)
    random.seed(seed)

    # Cache check (generate)
    provider = "TRIP_GENERATE"
    req_for_hash = req.model_copy(update={"seed": seed})
    payload = trip_generate_cache_payload(req_for_hash)
    h = _hash_request(provider, payload)
    cached = cache_get(provider, h)

    # Create trip row always (to get trip_id)
    with connect() as conn:
        trip_id = conn.execute(
            "INSERT INTO trips(origin_place_id, start_date, end_date, budget_yen, seed) VALUES(?,?,?,?,?)",
            (req.origin_place_id, req.start_date.isoformat(), req.end_date.isoformat(), int(req.budget_yen), seed),
        ).lastrowid

    if cached is not None:
        # restore DB + history
        persist_candidates(int(trip_id), cached["candidates"])
        insert_reroll_history(int(trip_id), None, seed, cached["candidates"])
        return GenerateTripResponse(trip_id=int(trip_id), seed=seed, candidates=cached["candidates"], cache_hit=True)

    all_dests = list_destination_places(exclude_id=origin.id)
    all_dests = _filter_by_tags(req, all_dests)
    if not all_dests:
        raise HTTPException(400, "No destinations match tag constraints. Relax tags or add more places.")

    desired = dynamic_sample_size(req)
    total_n = min(desired, len(all_dests))

    q_near, q_mid, q_far = distance_band_quota(req.max_total_travel_minutes)
    dests = sample_destinations_by_distance_band(
        origin=origin,
        destinations=all_dests,
        rnd=rnd,
        total_n=total_n,
        near_km=float(req.near_km or 80),
        mid_km=float(req.mid_km or 250),
        quota_near=q_near,
        quota_mid=q_mid,
        quota_far=q_far,
    )

    evaluated = []
    for d in dests:
        c = _build_candidates_for_dest(req, origin, d, seed)
        c.reason["sample_size"] = total_n
        c.reason["distance_band_quota"] = {"near": q_near, "mid": q_mid, "far": q_far}
        c.reason["distance_band_km"] = {"near_km": float(req.near_km or 80), "mid_km": float(req.mid_km or 250)}
        evaluated.append((c.score, d, c.routes[0].model_dump(), c.routes[1].model_dump(),
                          c.lodging.model_dump(), c.total_cost_yen, c.reason))

    evaluated = select_top_k_with_diversity(
        evaluated,
        k=3,
        min_km=float(req.min_destination_distance_km or 0),
    )

    candidates_out: List[CandidateOut] = []
    for score, dest, out_r, in_r, lodging, total, reason in evaluated:
        candidates_out.append(
            CandidateOut(
                destination={"id": dest.id, "name": dest.name, "region_code": dest.region_code},
                total_cost_yen=int(total),
                score=float(score),
                reason=reason,
                routes=[RouteOut(**out_r), RouteOut(**in_r)],
                lodging=LodgingOut(**lodging),
            )
        )

    # Persist + history
    persist_candidates(int(trip_id), [c.model_dump() for c in candidates_out])
    insert_reroll_history(int(trip_id), None, seed, [c.model_dump() for c in candidates_out])

    # Cache store
    cache_set(provider, h, {"candidates": [c.model_dump() for c in candidates_out]}, ttl_seconds=6 * 3600)

    return GenerateTripResponse(trip_id=int(trip_id), seed=seed, candidates=candidates_out, cache_hit=False)


def generate_candidates_into_trip(trip_id: int, req: GenerateTripRequest) -> GenerateTripResponse:
    # Load trip context
    with connect() as conn:
        trip = conn.execute("SELECT * FROM trips WHERE id = ?", (trip_id,)).fetchone()
    if not trip:
        raise HTTPException(404, "trip not found")

    origin = get_place(int(trip["origin_place_id"]))
    # use trip's dates/budget as base if caller omitted (UI always sends, but safe)
    base_req = req.model_copy(update={
        "origin_place_id": int(trip["origin_place_id"]),
        "start_date": date.fromisoformat(trip["start_date"]),
        "end_date": date.fromisoformat(trip["end_date"]),
        "budget_yen": int(trip["budget_yen"]),
    })

    # Seed
    seed = int(base_req.seed) if base_req.seed is not None else random.randint(1, 2_147_483_647)
    rnd = random.Random(seed)
    random.seed(seed)

    # Prev seed (for history)
    prev_seed = int(trip["seed"]) if trip["seed"] is not None else None

    # Selected IDs (reroll behavior + cache key)
    selected_ids = selected_destination_ids(trip_id)
    selected_ids_list = sorted(list(selected_ids))

    # Cache check (reroll) - include selected_ids in payload
    provider = "TRIP_REROLL"
    req_for_hash = base_req.model_copy(update={"seed": seed})
    payload = trip_generate_cache_payload(req_for_hash, selected_ids=selected_ids_list)
    h = _hash_request(provider, payload)
    cached = cache_get(provider, h)
    if cached is not None:
        # update trip seed + restore DB + history
        with connect() as conn:
            conn.execute("UPDATE trips SET seed = ? WHERE id = ?", (seed, trip_id))
        persist_candidates(trip_id, cached["candidates"])
        insert_reroll_history(trip_id, prev_seed, seed, cached["candidates"])
        return GenerateTripResponse(trip_id=trip_id, seed=seed, candidates=cached["candidates"], cache_hit=True)

    all_dests = list_destination_places(exclude_id=origin.id)
    all_dests = _filter_by_tags(base_req, all_dests)

    # Avoid selected (but revive if pool too small)
    avoid_enabled = bool(base_req.avoid_selected if base_req.avoid_selected is not None else True)
    selected_map = {p.id: p for p in all_dests if p.id in selected_ids}
    unselected = [p for p in all_dests if p.id not in selected_ids]
    all_dests_effective = unselected if avoid_enabled else all_dests

    min_pool = 10
    revived_ids: Set[int] = set()
    if avoid_enabled and len(all_dests_effective) < min_pool and selected_map:
        need = min_pool - len(all_dests_effective)
        revive = list(selected_map.values())
        rnd.shuffle(revive)
        revive = revive[:max(0, need)]
        revived_ids = {p.id for p in revive}
        all_dests_effective = all_dests_effective + revive

    all_dests = all_dests_effective
    if not all_dests:
        raise HTTPException(400, "No destinations available. Relax constraints or add more places.")

    desired = dynamic_sample_size(base_req)
    total_n = min(desired, len(all_dests))

    q_near, q_mid, q_far = distance_band_quota(base_req.max_total_travel_minutes)
    dests = sample_destinations_by_distance_band(
        origin=origin,
        destinations=all_dests,
        rnd=rnd,
        total_n=total_n,
        near_km=float(base_req.near_km or 80),
        mid_km=float(base_req.mid_km or 250),
        quota_near=q_near,
        quota_mid=q_mid,
        quota_far=q_far,
    )

    evaluated = []
    for d in dests:
        c = _build_candidates_for_dest(base_req, origin, d, seed)

        # Selected penalty (only matters when revived / avoid disabled)
        pen_sel = 15.0 if (d.id in selected_ids) else 0.0
        c.score = float(c.score - pen_sel)

        c.reason["sample_size"] = total_n
        c.reason["distance_band_quota"] = {"near": q_near, "mid": q_mid, "far": q_far}
        c.reason["distance_band_km"] = {"near_km": float(base_req.near_km or 80), "mid_km": float(base_req.mid_km or 250)}
        c.reason["selected_avoid_enabled"] = avoid_enabled
        c.reason["selected_penalty"] = pen_sel
        c.reason["selected_revived"] = bool(d.id in revived_ids)

        evaluated.append((c.score, d, c.routes[0].model_dump(), c.routes[1].model_dump(),
                          c.lodging.model_dump(), c.total_cost_yen, c.reason))

    evaluated = select_top_k_with_diversity(
        evaluated,
        k=3,
        min_km=float(base_req.min_destination_distance_km or 0),
    )

    candidates_out: List[CandidateOut] = []
    for score, dest, out_r, in_r, lodging, total, reason in evaluated:
        candidates_out.append(
            CandidateOut(
                destination={"id": dest.id, "name": dest.name, "region_code": dest.region_code},
                total_cost_yen=int(total),
                score=float(score),
                reason=reason,
                routes=[RouteOut(**out_r), RouteOut(**in_r)],
                lodging=LodgingOut(**lodging),
            )
        )

    # Persist: update trip seed + candidates + history
    with connect() as conn:
        conn.execute("UPDATE trips SET seed = ? WHERE id = ?", (seed, trip_id))
    persist_candidates(trip_id, [c.model_dump() for c in candidates_out])
    insert_reroll_history(trip_id, prev_seed, seed, [c.model_dump() for c in candidates_out])

    # Cache store (reroll)
    cache_set(provider, h, {"candidates": [c.model_dump() for c in candidates_out]}, ttl_seconds=6 * 3600)

    return GenerateTripResponse(trip_id=trip_id, seed=seed, candidates=candidates_out, cache_hit=False)


# =========================
# Load trip from DB (for UI)
# =========================

def load_trip_response(trip_id: int) -> GenerateTripResponse:
    with connect() as conn:
        trip = conn.execute("SELECT * FROM trips WHERE id = ?", (trip_id,)).fetchone()
        if not trip:
            raise HTTPException(status_code=404, detail="trip not found")

        cands = conn.execute(
            "SELECT * FROM trip_candidates WHERE trip_id = ? ORDER BY score DESC",
            (trip_id,),
        ).fetchall()

        candidates_out: List[CandidateOut] = []
        for c in cands:
            dest = conn.execute(
                "SELECT id, name, region_code FROM places WHERE id = ?",
                (c["destination_place_id"],),
            ).fetchone()

            routes = conn.execute(
                "SELECT * FROM routes WHERE trip_candidate_id = ?",
                (c["id"],),
            ).fetchall()

            lodging = conn.execute(
                "SELECT * FROM lodgings WHERE trip_candidate_id = ? LIMIT 1",
                (c["id"],),
            ).fetchone()

            routes_out: List[RouteOut] = []
            for r in routes:
                routes_out.append(
                    RouteOut(
                        direction=r["direction"],
                        provider=r["provider"],
                        duration_min=int(r["duration_min"]),
                        price_yen=int(r["price_yen"]),
                        price_after_discounts_yen=int(r["price_after_discounts_yen"]),
                        details=json.loads(r["details_json"]) if r["details_json"] else {},
                    )
                )
            # stable ordering
            routes_out.sort(key=lambda rr: 0 if rr.direction == "OUTBOUND" else 1)

            if lodging:
                lodging_out = LodgingOut(
                    provider=lodging["provider"],
                    name=lodging["name"],
                    price_per_night_yen=int(lodging["price_per_night_yen"]),
                    bargain_flag=bool(lodging["bargain_flag"]),
                    url=lodging["url"],
                    details=json.loads(lodging["details_json"]) if lodging["details_json"] else {},
                )
            else:
                lodging_out = LodgingOut(
                    provider="UNKNOWN",
                    name="UNKNOWN",
                    price_per_night_yen=0,
                    bargain_flag=False,
                    url=None,
                    details={},
                )

            candidates_out.append(
                CandidateOut(
                    destination={
                        "id": int(dest["id"]) if dest else int(c["destination_place_id"]),
                        "name": dest["name"] if dest else str(c["destination_place_id"]),
                        "region_code": dest["region_code"] if dest else None,
                    },
                    total_cost_yen=int(c["total_cost_yen"]),
                    score=float(c["score"]),
                    reason=json.loads(c["reason_json"]) if c["reason_json"] else {},
                    routes=routes_out,
                    lodging=lodging_out,
                )
            )

    return GenerateTripResponse(
        trip_id=int(trip["id"]),
        seed=int(trip["seed"]) if trip["seed"] is not None else None,
        candidates=candidates_out,
        cache_hit=False,
    )


# =========================
# Diff summary (latest reroll)
# =========================

def _snapshot_candidates(trip_id: int) -> List[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT snapshot_json FROM reroll_history WHERE trip_id = ? ORDER BY id DESC LIMIT 2",
            (trip_id,),
        ).fetchall()
    if len(rows) < 2:
        return []
    latest = json.loads(rows[0]["snapshot_json"])
    prev = json.loads(rows[1]["snapshot_json"])
    return [prev.get("candidates", []), latest.get("candidates", [])]


def _dest_key(c: dict) -> str:
    d = c.get("destination") or {}
    return str(d.get("id"))


@app.get("/v1/trips/{trip_id}/rerolls/compare/latest/summary")
def api_compare_latest_summary(trip_id: int):
    pair = _snapshot_candidates(trip_id)
    if not pair:
        return {"summary": None}

    prev, latest = pair
    prev_ids = [_dest_key(c) for c in prev]
    latest_ids = [_dest_key(c) for c in latest]

    prev_set = set(prev_ids)
    latest_set = set(latest_ids)

    added = [c for c in latest if _dest_key(c) not in prev_set]
    removed = [c for c in prev if _dest_key(c) not in latest_set]

    prev_rank = {pid: i + 1 for i, pid in enumerate(prev_ids)}
    latest_rank = {pid: i + 1 for i, pid in enumerate(latest_ids)}

    rank_changes = []
    for pid in latest_ids:
        if pid in prev_rank:
            dr = prev_rank[pid] - latest_rank[pid]
            if dr != 0:
                # find name
                name = None
                for c in latest:
                    if _dest_key(c) == pid:
                        name = (c.get("destination") or {}).get("name")
                        break
                rank_changes.append({"destination_id": int(pid), "name": name, "delta": dr})

    summary = {
        "added": [{"id": int(_dest_key(c)), "name": (c.get("destination") or {}).get("name")} for c in added],
        "removed": [{"id": int(_dest_key(c)), "name": (c.get("destination") or {}).get("name")} for c in removed],
        "rank_changes": rank_changes,
    }
    return {"summary": summary}


# =========================
# API endpoints
# =========================

@app.get("/v1/cache/stats")
def api_cache_stats():
    return cache_stats()


@app.post("/v1/cache/purge")
def api_cache_purge():
    cache_purge_expired()
    return {"status": "ok"}


@app.get("/v1/places")
def api_list_places(q: str = "", type: str = "", limit: int = 50):
    limit = max(1, min(int(limit), 200))
    q = (q or "").strip()
    type = (type or "").strip().upper()

    where = []
    params = []

    if q:
        where.append("name LIKE ?")
        params.append(f"%{q}%")

    if type in ("STATION", "AREA", "SPOT"):
        where.append("type = ?")
        params.append(type)

    sql = "SELECT id, type, name, region_code, lat, lon, tags_json FROM places"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id ASC LIMIT ?"
    params.append(limit)

    with connect() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()

    out = []
    for r in rows:
        out.append({
            "id": int(r["id"]),
            "type": r["type"],
            "name": r["name"],
            "region_code": r["region_code"],
            "lat": r["lat"],
            "lon": r["lon"],
            "tags": json.loads(r["tags_json"]) if r["tags_json"] else [],
        })
    return out


@app.post("/v1/trips/generate", response_model=GenerateTripResponse)
def api_generate_trip(req: GenerateTripRequest):
    return generate_trip(req)


@app.post("/v1/trips/{trip_id}/reroll", response_model=GenerateTripResponse)
def api_reroll_trip(trip_id: int, req: GenerateTripRequest = GenerateTripRequest(
    origin_place_id=1, start_date=date.today(), end_date=date.today(), budget_yen=0
)):
    # NOTE: UIは必須項目を送る前提だが、FastAPIの必須バリデーション回避のためダミーdefaultを置いている。
    # 実際には trip の値で上書きされる。
    return generate_candidates_into_trip(trip_id, req)


@app.get("/v1/trips")
def api_list_trips(limit: int = 20):
    limit = max(1, min(int(limit), 100))
    with connect() as conn:
        rows = conn.execute(
            "SELECT t.id, t.start_date, t.end_date, t.budget_yen, t.seed, "
            "p.name AS origin_name, p.region_code AS origin_region_code "
            "FROM trips t JOIN places p ON p.id = t.origin_place_id "
            "ORDER BY t.id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/v1/trips/{trip_id}", response_model=GenerateTripResponse)
def api_get_trip(trip_id: int):
    return load_trip_response(trip_id)


@app.post("/v1/trips/{trip_id}/select")
def api_select_candidate(trip_id: int, req: SelectCandidateRequest):
    with connect() as conn:
        t = conn.execute("SELECT id FROM trips WHERE id = ?", (trip_id,)).fetchone()
        if not t:
            raise HTTPException(status_code=404, detail="trip not found")

        c = conn.execute(
            "SELECT id FROM trip_candidates WHERE trip_id = ? AND destination_place_id = ?",
            (trip_id, req.destination_place_id),
        ).fetchone()
        if not c:
            raise HTTPException(status_code=400, detail="destination is not in current candidates")

        try:
            conn.execute(
                "INSERT INTO trip_selections(trip_id, destination_place_id, note) VALUES(?,?,?)",
                (trip_id, req.destination_place_id, req.note),
            )
        except Exception:
            # already selected, ignore
            pass

    return {"status": "ok"}


@app.get("/v1/trips/{trip_id}/selections")
def api_list_selections(trip_id: int):
    with connect() as conn:
        rows = conn.execute(
            "SELECT s.trip_id, s.destination_place_id, s.selected_at, s.note, p.name AS destination_name "
            "FROM trip_selections s JOIN places p ON p.id = s.destination_place_id "
            "WHERE s.trip_id = ? ORDER BY s.id DESC",
            (trip_id,),
        ).fetchall()
    return [dict(r) for r in rows]
