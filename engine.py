"""Engine for the Pressure Performance app: reuses the Phase 3 model (phase3_players.fit/tables) on any filtered slice of the data."""
import numpy as np, pandas as pd
from phase3_players import fit, tables

CLIP = 6.0


def load_raw(src):
    d = pd.read_csv(src, low_memory=False)
    d["year"] = pd.to_datetime(d.match_date).dt.year
    return d


def prepare(raw, hp_thr):
    """Per-ball score and high-pressure flag, computed on the FULL data so the score scale never shifts with filters."""
    d = raw.copy()
    ml = d.loc[d.is_legal, "leverage"].mean()
    d["s"] = (d.wpa / np.maximum(d.leverage, 0.3 * ml)).clip(-CLIP, CLIP)
    d = d[~((d.team_wkt > 0) & (~d.bowler_wkt))].copy()
    d["hp"] = (d.pressure_index >= hp_thr).astype(float)
    d["order_band"] = d.batting_order.clip(upper=7).astype(int)
    d["pace_or_spin"] = d.pace_or_spin.fillna("unk")
    unit = d[(d.hp > 0) & d.is_legal].leverage.mean() * 100 * 100   # win-prob points per 100 high-pressure balls
    return d, unit


def subset(d, years, comps, excluded=(), fit_teams=()):
    m = d.year.isin(years) & d.comp_group.isin(comps) & ~d.match_id.isin(list(excluded))
    if fit_teams:
        ids = d.loc[d.batting_team_id.isin(fit_teams) | d.bowling_team_id.isin(fit_teams), "match_id"].unique()
        m &= d.match_id.isin(ids)
    return d[m]


def _col(g, k, flag):
    return g[(k, flag)] if (k, flag) in g.columns else pd.Series(dtype=float)


def rate(d, unit, tau, team_names, reestimate=False, min_hp=100, min_matches=20, gate=True):
    m = fit(d, tau0=tuple(tau), fixed_tau=None if reestimate else tuple(tau))
    bat, bow = tables(d, m)
    for t in (bat, bow):
        t["rating"] = t.pressure_level * unit; t["rating_sd"] = t.sd_level * unit; t["usual_rating"] = t.usual * unit
    f = d[d.batter_faced].assign(out=lambda x: ((x.team_wkt > 0) & (x.dismissed_player_id == x.striker_id)).astype(int))
    g = f.groupby(["striker_id", f.hp > 0]).agg(b=("bat_runs", "size"), r=("bat_runs", "sum"), o=("out", "sum")).unstack()
    mp = bat.player_id
    bat["sr_hp"] = mp.map(100 * _col(g, "r", True) / _col(g, "b", True)).round(1)
    bat["sr_usual"] = mp.map(100 * _col(g, "r", False) / _col(g, "b", False)).round(1)
    bat["bpd_hp"] = mp.map(_col(g, "b", True) / _col(g, "o", True).replace(0, np.nan)).round(1)
    bat["bpd_usual"] = mp.map(_col(g, "b", False) / _col(g, "o", False).replace(0, np.nan)).round(1)
    h = d.groupby(["bowler_id", d.hp > 0]).agg(l=("is_legal", "sum"), r=("bowler_runs", "sum"), w=("bowler_wkt", "sum")).unstack()
    mp = bow.player_id
    bow["econ_hp"] = mp.map(6 * _col(h, "r", True) / _col(h, "l", True)).round(2)
    bow["econ_usual"] = mp.map(6 * _col(h, "r", False) / _col(h, "l", False)).round(2)
    bow["bpw_hp"] = mp.map(_col(h, "l", True) / _col(h, "w", True).replace(0, np.nan)).round(1)
    bow["bpw_usual"] = mp.map(_col(h, "l", False) / _col(h, "w", False).replace(0, np.nan)).round(1)
    tn = lambda ids: ", ".join(team_names.get(i, str(i)) for i in ids[:3])
    for t, tab, key, col in ((bat, f, "striker_id", "batting_team_id"), (bow, d, "bowler_id", "bowling_team_id")):
        order = tab.groupby(key)[col].agg(lambda s: s.value_counts().index.tolist())
        t["team_ids"] = t.player_id.map(order); t["team"] = t.team_ids.map(lambda ids: tn(ids) if isinstance(ids, list) else "")
    for t, hpcol in ((bat, "hp_balls"), (bow, "hp_legal")):
        t["sample_ok"] = (t[hpcol] >= min_hp) & (t.matches >= min_matches)
        t["elig"] = t.sample_ok & ((t.hp_wpa > 0) if gate else True)
    bat = bat.rename(columns={"hp_balls": "hp_n"}); bow = bow.rename(columns={"hp_legal": "hp_n"})
    return bat, bow, m


def bootstrap(d, tau, bat_ids, bow_ids, reps=20, seed=11, progress=None):
    """Resample whole matches; rank among eligible players; returns p_top10 and 5th/95th percentile rank."""
    rng = np.random.RandomState(seed); mids = d.match_id.unique(); gm = {k: v for k, v in d.groupby("match_id")}
    rb = {k: [] for k in bat_ids}; rw = {k: [] for k in bow_ids}
    for i in range(reps):
        ds = pd.concat([gm[k] for k in rng.choice(mids, len(mids), replace=True)], ignore_index=True)
        bb, ww = tables(ds, fit(ds, fixed_tau=tuple(tau)))
        for tab, ids, store in ((bb, bat_ids, rb), (ww, bow_ids, rw)):
            e = tab[tab.player_id.isin(ids)].copy(); e["rk"] = e.pressure_level.rank(ascending=False)
            for pid, rk in zip(e.player_id, e.rk): store[pid].append(rk)
        if progress: progress((i + 1) / reps)
    def summ(store):
        return pd.DataFrame({"player_id": list(store), "p_top10": [np.mean(np.array(v) <= 10) if v else np.nan for v in store.values()],
                             "rank_lo": [np.percentile(v, 5) if v else np.nan for v in store.values()],
                             "rank_hi": [np.percentile(v, 95) if v else np.nan for v in store.values()]})
    return summ(rb), summ(rw)


def season_profile(d, role, pid):
    if role == "Batters":
        x = d[(d.striker_id == pid) & d.batter_faced]
        out = x.groupby(["year", x.hp > 0]).agg(balls=("bat_runs", "size"), runs=("bat_runs", "sum")).unstack(fill_value=0)
        res = pd.DataFrame({"pressure balls": out.get(("balls", True), 0), "pressure strike rate": (100 * out.get(("runs", True), 0) / out.get(("balls", True), 0).replace(0, np.nan)).round(1),
                            "usual balls": out.get(("balls", False), 0), "usual strike rate": (100 * out.get(("runs", False), 0) / out.get(("balls", False), 0).replace(0, np.nan)).round(1)})
    else:
        x = d[d.bowler_id == pid]
        out = x.groupby(["year", x.hp > 0]).agg(l=("is_legal", "sum"), r=("bowler_runs", "sum")).unstack(fill_value=0)
        res = pd.DataFrame({"pressure balls": out.get(("l", True), 0), "pressure economy": (6 * out.get(("r", True), 0) / out.get(("l", True), 0).replace(0, np.nan)).round(2),
                            "usual balls": out.get(("l", False), 0), "usual economy": (6 * out.get(("r", False), 0) / out.get(("l", False), 0).replace(0, np.nan)).round(2)})
    res.index.name = "year"
    return res.reset_index()
