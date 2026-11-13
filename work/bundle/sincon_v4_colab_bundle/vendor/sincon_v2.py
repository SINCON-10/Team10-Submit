"""SINCON v2: fixed-origin forecast built from explicit, checkable components.

log1p(boardings) for a target day is predicted as

    level      mean of the last normal days of the same day type (weekday/Sat/Sun)
  + seasonal   how the same station/hour moved a year earlier between the
               equivalent origin and the equivalent target date
  + weekday    Monday..Friday offsets
  + special    public-holiday / long-weekend offsets learned from past instances

Only records on or before the forecast origin are read: the forecaster receives
an array truncated at the origin, so later observations cannot leak.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "sincon-v2.0"
START = pd.Timestamp("2025-01-01")
TRAIN_END = pd.Timestamp("2026-06-30")
TEST_END = pd.Timestamp("2026-10-31")
PAD = 15
HOURS = 18
YEAR = 364  # 52 weeks: keeps the day of week aligned

# Public holidays fixed before 2026-06-30 (substitute and one-off holidays included).
HOLIDAYS = {
    "2025-01-01": "new_year", "2025-01-27": "temporary",
    "2025-01-28": "seollal", "2025-01-29": "seollal", "2025-01-30": "seollal",
    "2025-03-01": "national", "2025-03-03": "substitute",
    "2025-05-05": "children_buddha", "2025-05-06": "substitute",
    "2025-06-03": "election", "2025-06-06": "memorial",
    "2025-08-15": "national", "2025-10-03": "national",
    "2025-10-05": "chuseok", "2025-10-06": "chuseok", "2025-10-07": "chuseok",
    "2025-10-08": "substitute", "2025-10-09": "national", "2025-12-25": "christmas",
    "2026-01-01": "new_year",
    "2026-02-16": "seollal", "2026-02-17": "seollal", "2026-02-18": "seollal",
    "2026-03-01": "national", "2026-03-02": "substitute",
    "2026-05-01": "labor", "2026-05-05": "children", "2026-05-24": "buddha",
    "2026-05-25": "substitute", "2026-06-03": "election", "2026-06-06": "memorial",
    "2026-07-17": "constitution", "2026-08-15": "national", "2026-08-17": "substitute",
    "2026-09-24": "chuseok", "2026-09-25": "chuseok", "2026-09-26": "chuseok",
    "2026-10-03": "national", "2026-10-05": "substitute", "2026-10-09": "national",
}
SEMI_HOLIDAYS = {"2025-05-01"}  # Labour Day before it became a public holiday

# Day categories. 0-2 are ordinary days; the rest get an offset from past instances.
WK, SAT, SUN = 0, 1, 2
HOL_WK, HOL_SAT, HOL_SUN, LUN_PRE, LUN_DAY, LUN_POST = 3, 4, 5, 6, 7, 8
PRE, POST, BRIDGE, IN_SAT, IN_SUN = 9, 10, 11, 12, 13
EXCLUDED = -1
CATEGORY_NAMES = ["wk", "sat", "sun", "hol_wk", "hol_sat", "hol_sun", "lun_pre", "lun_day",
                  "lun_post", "pre", "post", "bridge", "in_sat", "in_sun"]
FALLBACK = {HOL_SAT: HOL_WK, LUN_PRE: LUN_DAY, LUN_POST: LUN_DAY, LUN_DAY: HOL_WK}

DEFAULTS = dict(
    level_weeks=8,        # recent window for the level
    season_half=14,       # +-days around the same date a year earlier
    season_weight=0.6,    # station-hour share; the rest is the station average over hours
    weekend_level_weeks=12,   # Saturdays/Sundays are few, so they use a wider window
    weekend_season_half=14,
    weekend_season_weight=0.5,
    dow_weeks=26,         # window for Monday..Friday offsets
    dow_scale=1.0,
    ref_half=28,          # local reference window around a past special day
    effect_prior=1.0,     # pseudo-count pulling special-day offsets to the all-station mean
    use_special=True,
    min_prior_days=28,    # prior-year days needed before a station's own seasonal shift is used
    break_step=0.08,      # log jump (vs the median station) that marks a station as hit by a line opening
    use_events=True,
    event_ratio=4.0,      # a day is an event day when some hour exceeds this multiple of the local median
    event_min_days=20,    # event days in the last year needed to treat a station as an event station
    event_half=28,        # +-days used for the local median and for last year's event frequency
    event_season=45,      # +-days around the same date last year whose events shape the profile
    play_prob=0.9,        # chance that a scheduled event takes place (rain-outs are unknown)
    other_prob=0.03,      # chance of an unscheduled event on a day the schedule covers
)


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def build_calendar():
    """Per-day arrays for START..TEST_END (index = days since START)."""
    dates = pd.date_range(START - pd.Timedelta(days=PAD), TEST_END + pd.Timedelta(days=PAD))
    n = len(dates)
    names = pd.Series(dates.strftime("%Y-%m-%d")).map(HOLIDAYS).fillna("").to_numpy()
    semi = pd.Series(dates.strftime("%Y-%m-%d")).isin(SEMI_HOLIDAYS).to_numpy()
    dow = dates.dayofweek.to_numpy()
    hol = names != ""
    off = (dow >= 5) | hol
    length = np.zeros(n, int)
    i = 0
    while i < n:
        if not off[i]:
            i += 1
            continue
        end = i
        while end + 1 < n and off[end + 1]:
            end += 1
        length[i:end + 1] = end - i + 1
        i = end + 1
    to_off, from_off = np.full(n, 9), np.full(n, 9)
    next_len, prev_len = np.zeros(n, int), np.zeros(n, int)
    for i in range(n):
        if off[i]:
            to_off[i] = from_off[i] = 0
            continue
        for j in range(1, 9):
            if i + j < n and off[i + j]:
                to_off[i], next_len[i] = j, length[i + j]
                break
        for j in range(1, 9):
            if i - j >= 0 and off[i - j]:
                from_off[i], prev_len[i] = j, length[i - j]
                break
    lunar = np.isin(names, ["seollal", "chuseok"])
    category = np.full(n, EXCLUDED)
    for i in range(n):
        if semi[i]:
            continue
        if lunar[i]:
            before = i > 0 and lunar[i - 1]
            after = i + 1 < n and lunar[i + 1]
            category[i] = LUN_DAY if before and after else (LUN_POST if before else LUN_PRE)
        elif hol[i]:
            category[i] = HOL_SAT if dow[i] == 5 else (HOL_SUN if dow[i] == 6 else HOL_WK)
        elif off[i]:
            long_block = length[i] >= 3
            category[i] = (IN_SAT if long_block else SAT) if dow[i] == 5 else (IN_SUN if long_block else SUN)
        else:
            pre = to_off[i] == 1 and next_len[i] >= 3
            post = from_off[i] == 1 and prev_len[i] >= 3
            bridge = to_off[i] == 1 and from_off[i] == 1
            category[i] = BRIDGE if bridge else (PRE if pre else (POST if post else WK))
    day_type = np.where(hol | (dow == 6), SUN, np.where(dow == 5, SAT, WK))
    sl = slice(PAD, n - PAD)
    return dict(dates=dates[sl], dow=dow[sl], is_hol=hol[sl].astype(int), day_type=day_type[sl],
                category=category[sl], block_len=length[sl], names=names[sl])


CAL = build_calendar()
N_TRAIN = (TRAIN_END - START).days + 1
N_ALL = (TEST_END - START).days + 1


def day_index(date):
    return (pd.Timestamp(date) - START).days


def load(data_dir):
    """Read the raw files into a (day, station, hour) array of log1p(boardings)."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["date"])
    test = pd.read_csv(data_dir / "test.csv", parse_dates=["date"])
    meta = pd.read_csv(data_dir / "metaData.csv")
    require(len(train) == 688176 and len(test) == 157194 and len(meta) == 71, "Unexpected file sizes")
    require(train.date.between(START, TRAIN_END).all(), "Train dates outside 2025-01-01..2026-06-30")
    require(test.date.between(TRAIN_END + pd.Timedelta(days=1), TEST_END).all(), "Test dates invalid")
    require(train.hour.between(6, 23).all() and test.hour.between(6, 23).all(), "Hour outside 6..23")
    require(not train.duplicated(["date", "station_id", "hour"]).any(), "Duplicate train keys")
    require(not test.duplicated(["date", "station_id", "hour"]).any(), "Duplicate test keys")
    require(test.example_id.is_unique, "Duplicate test ids")
    stations = np.sort(meta.station_id.unique())
    require(set(train.station_id) == set(stations) == set(test.station_id), "Station mismatch")
    index = {s: i for i, s in enumerate(stations)}
    values = train.boardings.to_numpy(dtype=float)
    require(np.isfinite(values).all() and (values >= 0).all(), "Invalid boardings")
    Y = np.full((N_TRAIN, len(stations), HOURS), np.nan, np.float32)
    Y[(train.date - START).dt.days.to_numpy(), train.station_id.map(index).to_numpy(),
      train.hour.to_numpy() - 6] = np.log1p(values)
    lines = meta.set_index("station_id").loc[stations, "canonical_line"].to_numpy()
    return dict(Y=Y, stations=stations, index=index, lines=lines, test=test, meta=meta)


def nan_mean(values, axis=0):
    count = np.isfinite(values).sum(axis=axis)
    total = np.nansum(values, axis=axis)
    out = np.full(np.shape(total), np.nan, np.float32)
    np.divide(total, count, out=out, where=count > 0)
    return out, count


class Forecaster:
    """Forecast from one origin. `history` must end at the origin (no later rows)."""

    def __init__(self, history, schedule=None, **config):
        """schedule: optional {station index: (set of scheduled day indices, last covered day index)}."""
        unknown = set(config) - set(DEFAULTS)
        require(not unknown, f"Unknown options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **config}
        self.raw = history
        self.c = len(history) - 1
        require(0 < self.c < N_TRAIN, "Origin outside the training period")
        self.S = history.shape[1]
        self.cat = CAL["category"]
        self.dow = CAL["dow"]
        self.day_type = CAL["day_type"]
        self.normal = (self.cat >= WK) & (self.cat <= SUN)
        self.schedule = schedule or {}
        self.events = np.zeros(history.shape[:2], bool)
        self.event_stations = np.zeros(self.S, bool)
        if self.cfg["use_events"]:
            self.events = detect_events(history, self.day_type, self.cfg["event_ratio"], self.cfg["event_half"])
            self.event_stations = self.events[max(0, self.c - 364):].sum(axis=0) >= self.cfg["event_min_days"]
            for s in self.schedule:
                self.event_stations[s] = True
        # Ordinary level, seasonal shift and offsets are computed without event days.
        self.Y = history
        if self.event_stations.any():
            self.Y = history.copy()
            hide = self.events & self.event_stations[None, :]
            self.Y[hide] = np.nan
        self._prepare()
        self._event_excess = {int(s): self._excess_by_day(int(s)) for s in np.flatnonzero(self.event_stations)}

    # ---- building blocks -------------------------------------------------
    def _mean(self, lo, hi, day_type, dow=None, min_count=2):
        """Mean over normal days of one day type inside [lo, hi], never past the origin."""
        lo, hi = max(0, lo), min(self.c, hi)
        empty = np.full((self.S, HOURS), np.nan, np.float32)
        if hi < lo:
            return empty
        mask = self.normal[lo:hi + 1] & (self.day_type[lo:hi + 1] == day_type)
        if dow is not None:
            mask &= self.dow[lo:hi + 1] == dow
        if not mask.any():
            return empty
        mean, count = nan_mean(self.Y[lo:hi + 1][mask])
        return np.where(count >= min_count, mean, np.nan)

    def _weeks(self, day_type):
        return self.cfg["level_weeks"] if day_type == WK else self.cfg["weekend_level_weeks"]

    def _level(self, end, day_type, weeks=None):
        weeks = weeks or self._weeks(day_type)
        return self._mean(end - 7 * weeks + 1, end, day_type)

    def _prepare(self):
        cfg, c = self.cfg, self.c
        self.level = {}
        for k in (WK, SAT, SUN):
            level = self._level(c, k)
            for weeks in (16, 52, 80):  # stations with a short record
                level = np.where(np.isfinite(level), level, self._level(c, k, weeks))
            self.level[k] = level
        fallback = nan_mean(np.stack([self.level[k] for k in (WK, SAT, SUN)]))[0]
        overall = nan_mean(self.Y.reshape(-1, HOURS))[0]
        for k in (WK, SAT, SUN):
            level = np.where(np.isfinite(self.level[k]), self.level[k], fallback)
            self.level[k] = np.where(np.isfinite(level), level, overall[None, :])
        # A year earlier, at the equivalent origin.
        self.prior_level = {k: self._level(c - YEAR, k) for k in (WK, SAT, SUN)}
        if c - YEAR >= 0:
            lo = max(0, c - YEAR - 7 * cfg["level_weeks"] + 1)
            observed = np.isfinite(self.Y[lo:c - YEAR + 1]).any(axis=2).sum(axis=0)
        else:
            observed = np.zeros(self.S)
        self.own_season = observed >= cfg["min_prior_days"]
        # A new line opening shifts riders between stations once; that is not a seasonal move.
        self.break_day, self.break_stations = find_break(self.raw, self.Y, self.cat, cfg["break_step"])
        # Monday..Friday offsets.
        lo = c - 7 * cfg["dow_weeks"] + 1
        weekday = self._mean(lo, c, WK)
        self.dow_effect = np.zeros((5, self.S, HOURS), np.float32)
        for j in range(5):
            self.dow_effect[j] = np.nan_to_num(self._mean(lo, c, WK, dow=j) - weekday) * cfg["dow_scale"]
        self.effect = self._special_effects() if cfg["use_special"] else {}

    def _reference(self, q):
        """Normal-day level around a past special day q (same base day type)."""
        k = self.day_type[q]
        ref = None
        for half in (self.cfg["ref_half"], 2 * self.cfg["ref_half"]):
            if k == WK:
                ref = self._mean(q - half, q + half, WK, dow=self.dow[q])
                ref = np.where(np.isfinite(ref), ref, self._mean(q - half, q + half, WK))
            else:
                ref = self._mean(q - half, q + half, k)
            if np.isfinite(ref).any():
                break
        return ref

    def _special_effects(self):
        effects = {}
        prior = self.cfg["effect_prior"]
        for category in range(HOL_WK, IN_SUN + 1):
            days = np.flatnonzero(self.cat[:self.c + 1] == category)
            if len(days) == 0:
                continue
            mean, count = nan_mean(np.stack([self.Y[q] - self._reference(q) for q in days]))
            pooled = np.nan_to_num(nan_mean(mean)[0])        # all-station mean per hour
            mean = np.where(count > 0, mean, pooled[None, :])
            effects[category] = ((count * mean + prior * pooled[None, :]) / (count + prior)).astype(np.float32)
        return effects

    def _effect(self, category):
        seen = set()
        while category not in self.effect and category in FALLBACK and category not in seen:
            seen.add(category)
            category = FALLBACK[category]
        return self.effect.get(category)

    def seasonal(self, d, k):
        """Shift between the equivalent origin and target date one year earlier."""
        cfg = self.cfg
        center = d - YEAR
        empty = np.zeros((self.S, HOURS), np.float32)
        if self.c - YEAR < 0 or center < 0:
            return empty, False
        half = cfg["season_half"] if k == WK else cfg["weekend_season_half"]
        weight = cfg["season_weight"] if k == WK else cfg["weekend_season_weight"]
        window = self._mean(center - half, center + half, k)
        window = np.where(np.isfinite(window), window, self._mean(center - 2 * half, center + 2 * half, k))
        shift = window - self.prior_level[k]
        own = self.own_season
        prior_start = self.c - YEAR - 7 * self._weeks(k)
        if self.break_day is not None and prior_start < self.break_day <= center + half:
            own = own & ~self.break_stations       # the opening falls inside last year's comparison
        shift[~own] = np.nan
        if not np.isfinite(shift).any():
            return empty, False
        station = nan_mean(shift, axis=1)[0][:, None]
        shift = weight * shift + (1 - weight) * station
        pooled = np.nan_to_num(nan_mean(shift)[0])           # stations without a prior year
        shift = np.where(np.isfinite(shift), shift, pooled[None, :])
        return shift.astype(np.float32), True

    # ---- events ----------------------------------------------------------
    def _excess_by_day(self, s):
        """Past event days of station s -> log excess over the ordinary level around that day."""
        excess = {}
        for q in np.flatnonzero(self.events[:, s]):
            ref = self._mean(q - self.cfg["event_half"], q + self.cfg["event_half"], self.day_type[q])[s]
            if np.isfinite(ref).all():
                excess[int(q)] = self.raw[q, s] - ref
        return excess

    def event_profile(self, s, d):
        """Mean excess of comparable past events: same day type, same season or recent."""
        excess = self._event_excess.get(s, {})
        same_type = [q for q in excess if self.day_type[q] == self.day_type[d]]
        near = [q for q in same_type
                if abs(q - (d - YEAR)) <= self.cfg["event_season"] or q > self.c - 7 * self.cfg["level_weeks"]]
        for group in (near, same_type, list(excess)):
            if len(group) >= 3:
                return np.mean([excess[q] for q in group], axis=0)
        return np.zeros(HOURS, np.float32)

    def event_probability(self, s, d):
        """Scheduled -> play_prob; otherwise the share of event days on this weekday in season."""
        if s in self.schedule:
            scheduled, covered_until = self.schedule[s]
            if d <= covered_until:
                return self.cfg["play_prob"] if d in scheduled else self.cfg["other_prob"]
        half = self.cfg["event_half"]
        first = int(np.argmax(np.isfinite(self.raw[:, s]).any(axis=1)))
        center = d - YEAR
        if center - half >= first:
            days = np.arange(center - half, min(self.c, center + half) + 1)
        else:
            days = np.arange(max(first, self.c - 2 * half + 1), self.c + 1)
        days = days[self.dow[days] == self.dow[d]]
        return float(self.events[days, s].mean()) if len(days) else 0.0

    # ---- forecast --------------------------------------------------------
    def predict(self, days, parts=False):
        """log1p predictions, shape (len(days), stations, 18)."""
        days = np.asarray(days)
        require((days > self.c).all() and (days < N_ALL).all(), "Target days must follow the origin")
        out = np.empty((len(days), self.S, HOURS), np.float32)
        names = ("level", "season", "dow", "special", "event")
        detail = {name: np.zeros_like(out) for name in names} if parts else None
        for i, d in enumerate(days):
            k, category = self.day_type[d], self.cat[d]
            season, _ = self.seasonal(d, k)
            dow = self.dow_effect[self.dow[d]] if category in (WK, PRE, POST, BRIDGE) else 0.0
            special = self._effect(category) if category > SUN else None
            special = 0.0 if special is None else special
            event = np.zeros((self.S, HOURS), np.float32)
            for s in np.flatnonzero(self.event_stations):
                event[s] = self.event_probability(int(s), int(d)) * self.event_profile(int(s), int(d))
            out[i] = self.level[k] + season + dow + special + event
            if parts:
                detail["level"][i], detail["season"][i] = self.level[k], season
                detail["dow"][i], detail["special"][i], detail["event"][i] = dow, special, event
        out = np.maximum(out, 0.0)
        return (out, detail) if parts else out


def find_break(raw, history, category, step=0.08, half=14):
    """Stations whose weekday level jumped when new stations opened.

    The opening day is read from `raw` (first day a late-starting station is observed).
    Levels come from `history` (event days already hidden). A station is affected when its
    jump across that day differs from the median station's by more than `step` in log units.
    Returns (day index or None, boolean per station).
    """
    n, stations = raw.shape[:2]
    affected = np.zeros(stations, bool)
    seen = np.isfinite(raw).any(axis=2)
    first = np.where(seen.any(axis=0), seen.argmax(axis=0), n)
    late = first[(first > 0) & (first < n)]
    if len(late) == 0:
        return None, affected
    day = int(late.min())
    ordinary = np.flatnonzero(category[:n] == WK)
    before = ordinary[(ordinary >= day - half) & (ordinary < day)]
    after = ordinary[(ordinary >= day) & (ordinary < day + half)]
    if len(before) < 5 or len(after) < 5:
        return day, affected
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # stations not yet open are all-NaN
        jump = np.nanmean(np.nanmean(history[after], axis=0) - np.nanmean(history[before], axis=0), axis=1)
        affected = np.abs(jump - np.nanmedian(jump)) > step
    return day, affected


def read_schedule(path, data):
    """Optional event schedule announced before 2026-06-30.

    CSV columns: date, station_id, source_url. Each station is treated as covered up to its
    last listed date; later days fall back to last year's event frequency.
    """
    table = pd.read_csv(path, parse_dates=["date"])
    require({"date", "station_id", "source_url"} <= set(table.columns), "Schedule columns missing")
    require(table.station_id.isin(data["stations"]).all(), "Unknown station in schedule")
    require(table.source_url.fillna("").str.match(r"https?://").all(), "Schedule rows need a source link")
    require(table.date.between(TRAIN_END + pd.Timedelta(days=1), TEST_END).all(), "Schedule dates outside the forecast period")
    schedule = {}
    for station, rows in table.groupby("station_id"):
        days = set((rows.date - START).dt.days.tolist())
        schedule[data["index"][station]] = (days, max(days))
    return schedule


def detect_events(history, day_type, ratio=4.0, half=28):
    """Days on which some hour is `ratio` times the local median of the same day type.

    Uses only `history` (rows up to the origin). Returns a (day, station) boolean array.
    """
    n = len(history)
    events = np.zeros(history.shape[:2], bool)
    threshold = np.log(ratio)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # stations not yet open are all-NaN
        for d in range(n):
            lo, hi = max(0, d - half), min(n - 1, d + half)
            block = history[lo:hi + 1][day_type[lo:hi + 1] == day_type[d]]
            median = np.nanmedian(block, axis=0)
            events[d] = np.nanmax(history[d] - median, axis=1) > threshold
    return events


def evaluate(Y, cutoff, horizon=123, schedule=None, **config):
    """Hold-out check: forecast from `cutoff` and score against the following days."""
    c = day_index(cutoff)
    days = np.arange(c + 1, min(N_TRAIN - 1, c + horizon) + 1)
    pred = Forecaster(Y[:c + 1], schedule=schedule, **config).predict(days)
    error = (Y[days] - pred) ** 2
    return days, pred, error


def summarize(days, error, exclude=None):
    """RMSLE overall and by day category."""
    keep = np.ones(error.shape[1], bool)
    if exclude is not None:
        keep &= ~exclude
    rows = {"all": float(np.sqrt(np.nanmean(error))), "kept": float(np.sqrt(np.nanmean(error[:, keep])))}
    cats = CAL["category"][days]
    for label, members in [("wk", [WK]), ("sat", [SAT]), ("sun", [SUN]),
                           ("holiday", [HOL_WK, HOL_SAT, HOL_SUN, LUN_PRE, LUN_DAY, LUN_POST]),
                           ("around", [PRE, POST, BRIDGE, IN_SAT, IN_SUN])]:
        mask = np.isin(cats, members)
        rows[label] = float(np.sqrt(np.nanmean(error[mask][:, keep]))) if mask.any() else np.nan
    return rows


def make_submission(data, path, schedule=None, **config):
    """Forecast 2026-07-01..2026-10-31 from 2026-06-30 and write example_id, boardings."""
    Y, test = data["Y"], data["test"]
    require(len(Y) == N_TRAIN, "History must end on 2026-06-30")
    days = np.arange(N_TRAIN, N_ALL)
    pred = Forecaster(Y, schedule=schedule, **config).predict(days)
    d = (test.date - START).dt.days.to_numpy() - N_TRAIN
    s = test.station_id.map(data["index"]).to_numpy()
    h = test.hour.to_numpy() - 6
    boardings = np.expm1(pred[d, s, h].astype(np.float64))
    require(np.isfinite(boardings).all() and (boardings >= 0).all(), "Invalid predictions")
    out = pd.DataFrame({"example_id": test.example_id.to_numpy(), "boardings": boardings})
    require(len(out) == 157194 and out.example_id.is_unique, "Submission rows invalid")
    out.to_csv(path, index=False)
    return out
