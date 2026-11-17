#!/usr/bin/env python3
"""Phase 3: player performance under pressure.

Usage: python phase3_players.py <ball_pressure.csv.gz> <out_dir> [--hp 1.5]

Per-ball outcome score (leverage-neutral):  s = WPA / max(leverage, 0.3 * mean leverage), winsorised at +-6.
  WPA is the batting side's win-probability change on the delivery. Dividing by leverage puts a boundary in a dead
  over and a boundary in a tight finish on the same scale ("how good was the outcome for the situation").

Joint crossed random-effects model, fitted by EM (exact posterior, dense normal equations):
  s_ball = x'beta + b_batter + d_batter * HP  -  w_bowler  -  e_bowler * HP  + noise
  HP = delivery with Pressure Index >= threshold.
  b = usual level, d = change under pressure (the "pressure delta"), w/e the same for bowlers.
  x = fixed context: phase x innings, batting-order band, competition, pace/spin, and the same context interacted
      with HP (so a finisher is compared with finishers, a chase with chases).
  Because bowler terms are in the batter model (and vice versa) every player is compared against the quality of
  opposition actually faced. All random effects are shrunk by estimated variance components (empirical Bayes).
"""
import argparse, json, os
import numpy as np, pandas as pd
import scipy.sparse as sp

CLIP = 6.0


def prep(path, hp_thr, continuous=False):
    d = pd.read_csv(path, low_memory=False)
    ml = d.loc[d.is_legal, "leverage"].mean()
    d["s"] = (d.wpa / np.maximum(d.leverage, 0.3 * ml)).clip(-CLIP, CLIP)
    # drop run-outs / non-bowler dismissals: neither a bowler wicket nor a clean batting outcome
    d = d[~((d.team_wkt > 0) & (~d.bowler_wkt))].copy()
    d["hp"] = (d.pressure_index >= hp_thr).astype(float)
    if continuous:
        d["hp"] = np.log(np.maximum(d.pressure_index, 0.3)).clip(-1.5, 2.5)
        d["hp"] = d.hp - d.hp.mean()
    d["order_band"] = d.batting_order.clip(upper=7).astype(int)
    d["pace_or_spin"] = d.pace_or_spin.fillna("unk")
    return d


def fixed_design(d):
    cols = {}
    hp = d.hp.values
    ph = pd.get_dummies(d.phase + "_" + d.innings.astype(str)).astype(float)
    ph = ph.drop(columns=ph.columns[0])
    ob = pd.get_dummies(d.order_band.astype(str), prefix="ob").astype(float); ob = ob.drop(columns=ob.columns[0])
    cg = pd.get_dummies(d.comp_group, prefix="cg").astype(float); cg = cg.drop(columns=cg.columns[0])
    ps = pd.get_dummies(d.pace_or_spin, prefix="ps").astype(float); ps = ps.drop(columns=ps.columns[0])
    base = pd.concat([ph, ob, cg, ps], axis=1)
    inter = base.mul(hp, axis=0).add_suffix("_xHP")
    F = pd.concat([pd.Series(1.0, index=d.index, name="const"), pd.Series(hp, index=d.index, name="HP"), base, inter], axis=1)
    return F


def fit(d, iters=120, tau0=(0.07, 0.06, 0.07, 0.06), fixed_tau=None, verbose=False):
    """Returns dict with posterior means/sd for batter b,d and bowler w,e."""
    F = fixed_design(d)
    bat_ids = np.sort(d.loc[d.batter_faced, "striker_id"].unique()); bow_ids = np.sort(d.bowler_id.unique())
    bi = pd.Series(np.arange(len(bat_ids)), index=bat_ids); wi = pd.Series(np.arange(len(bow_ids)), index=bow_ids)
    n = len(d); nb, nw = len(bat_ids), len(bow_ids); nf = F.shape[1]
    hp = d.hp.values
    rows = np.arange(n)
    isb = d.batter_faced.values
    bcol = bi.reindex(d.striker_id.values).values
    # column layout: fixed | b | d | w | e
    Zb = sp.csr_matrix((np.ones(isb.sum()), (rows[isb], bcol[isb].astype(int))), shape=(n, nb))
    Zd = sp.csr_matrix((hp[isb], (rows[isb], bcol[isb].astype(int))), shape=(n, nb))
    wcol = wi.reindex(d.bowler_id.values).values.astype(int)
    Zw = sp.csr_matrix((-np.ones(n), (rows, wcol)), shape=(n, nw))
    Ze = sp.csr_matrix((-hp, (rows, wcol)), shape=(n, nw))
    W = sp.hstack([sp.csr_matrix(F.values), Zb, Zd, Zw, Ze]).tocsr()
    y = d.s.values
    G = (W.T @ W).toarray(); r = W.T @ y
    groups = [slice(nf, nf + nb), slice(nf + nb, nf + 2 * nb), slice(nf + 2 * nb, nf + 2 * nb + nw), slice(nf + 2 * nb + nw, nf + 2 * nb + 2 * nw)]
    sizes = [nb, nb, nw, nw]
    from scipy.linalg import cho_factor, cho_solve
    from scipy.optimize import minimize
    yy = y @ y
    sig2 = y.var()
    p = G.shape[0]

    def prec_vec(tau, sig2):
        prec = np.full(p, 1e-6)
        for g, t in zip(groups, tau): prec[g] = sig2 / t ** 2
        return prec

    def nll(logtau, sig2):
        tau = np.exp(logtau)
        c = cho_factor(G + np.diag(prec_vec(tau, sig2)))
        logdetA = 2 * np.log(np.diag(c[0])).sum()
        quad = (yy - r @ cho_solve(c, r)) / sig2
        # log|A| - sum log(prec) over random effects (fixed-effect terms are constant in tau)
        lp = sum(sz * np.log(sig2 / t ** 2) for sz, t in zip(sizes, tau))
        return 0.5 * (n * np.log(sig2) + logdetA - lp + quad)

    tau = np.array(tau0, float) if fixed_tau is None else np.array(fixed_tau, float)
    if fixed_tau is None:
        for rnd in range(2):  # alternate: variance components <-> residual variance
            res = minimize(nll, np.log(tau), args=(sig2,), method="Nelder-Mead", options={"xatol": 0.02, "fatol": 0.05, "maxiter": 120})
            tau = np.exp(res.x)
            Ainv = np.linalg.inv(G + np.diag(prec_vec(tau, sig2))); beta = Ainv @ r
            resid = y - W @ beta
            sig2 = (resid @ resid + sig2 * np.einsum("ij,ji->", G, Ainv)) / n
            if verbose: print("round", rnd, tau.round(4), round(sig2, 4), "nll", round(res.fun, 2))
    Ainv = np.linalg.inv(G + np.diag(prec_vec(tau, sig2))); beta = Ainv @ r
    cov = sig2 * Ainv
    out = {"bat_ids": bat_ids, "bow_ids": bow_ids, "tau": tau, "sig2": sig2, "nf": nf, "F": list(F.columns),
           "beta_fixed": beta[:nf]}
    sl = groups
    out["b"], out["d"] = beta[sl[0]], beta[sl[1]]
    out["w"], out["e"] = beta[sl[2]], beta[sl[3]]
    out["sd_b"], out["sd_d"] = np.sqrt(np.diag(cov)[sl[0]]), np.sqrt(np.diag(cov)[sl[1]])
    out["sd_w"], out["sd_e"] = np.sqrt(np.diag(cov)[sl[2]]), np.sqrt(np.diag(cov)[sl[3]])
    # sd of the pressure level b + d and w + e (needs covariance)
    out["sd_bd"] = np.sqrt(np.maximum(np.diag(cov)[sl[0]] + np.diag(cov)[sl[1]] + 2 * np.diag(cov[sl[0], sl[1]]), 0))
    out["sd_we"] = np.sqrt(np.maximum(np.diag(cov)[sl[2]] + np.diag(cov)[sl[3]] + 2 * np.diag(cov[sl[2], sl[3]]), 0))
    return out


def tables(d, m):
    hp = d[d.hp > 0] if d.hp.max() <= 1 else d
    bat = pd.DataFrame({"player_id": m["bat_ids"], "usual": m["b"], "delta": m["d"], "sd_usual": m["sd_b"], "sd_delta": m["sd_d"],
                        "pressure_level": m["b"] + m["d"], "sd_level": m["sd_bd"]})
    nm = d.drop_duplicates("striker_id").set_index("striker_id").striker_name
    bat["name"] = bat.player_id.map(nm)
    d = d.assign(bat_out=((d.team_wkt > 0) & (d.dismissed_player_id == d.striker_id)).astype(int))
    f = d[d.batter_faced]
    g = f.groupby("striker_id")
    bat["balls"] = bat.player_id.map(g.size())
    fh = f[f.hp > 0].groupby("striker_id")
    bat["hp_balls"] = bat.player_id.map(fh.size()).fillna(0).astype(int)
    bat["hp_runs"] = bat.player_id.map(fh.bat_runs.sum()).fillna(0)
    bat["hp_sr"] = (100 * bat.hp_runs / bat.hp_balls.replace(0, np.nan)).round(1)
    bat["hp_dismissals"] = bat.player_id.map(fh.bat_out.sum()).fillna(0).astype(int)
    bat["hp_wpa"] = bat.player_id.map(fh.wpa.sum()).fillna(0)
    bat["matches"] = bat.player_id.map(g.match_id.nunique())
    bow = pd.DataFrame({"player_id": m["bow_ids"], "usual": m["w"], "delta": m["e"], "sd_usual": m["sd_w"], "sd_delta": m["sd_e"],
                        "pressure_level": m["w"] + m["e"], "sd_level": m["sd_we"]})
    nm = d.drop_duplicates("bowler_id").set_index("bowler_id").bowler_name
    bow["name"] = bow.player_id.map(nm)
    g = d.groupby("bowler_id")
    bow["deliveries"] = bow.player_id.map(g.size())
    dh = d[d.hp > 0].groupby("bowler_id")
    bow["hp_deliveries"] = bow.player_id.map(dh.size()).fillna(0).astype(int)
    bow["hp_legal"] = bow.player_id.map(dh.is_legal.sum()).fillna(0).astype(int)
    bow["hp_runs"] = bow.player_id.map(dh.bowler_runs.sum()).fillna(0)
    bow["hp_econ"] = (6 * bow.hp_runs / bow.hp_legal.replace(0, np.nan)).round(2)
    bow["hp_wkts"] = bow.player_id.map(dh.bowler_wkt.sum()).fillna(0).astype(int)
    bow["hp_wpa"] = bow.player_id.map((-dh.wpa.sum())).fillna(0)
    bow["matches"] = bow.player_id.map(g.match_id.nunique())
    return bat, bow


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("ball_pressure"); ap.add_argument("out_dir"); ap.add_argument("--hp", type=float, default=1.5)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    d = prep(a.ball_pressure, a.hp)
    m = fit(d, verbose=True)
    bat, bow = tables(d, m)
    bat.to_csv(f"{a.out_dir}/batters_all.csv", index=False); bow.to_csv(f"{a.out_dir}/bowlers_all.csv", index=False)
    json.dump({"tau_b_d_w_e": m["tau"].tolist(), "sigma2": m["sig2"], "hp_threshold": a.hp,
               "fixed": dict(zip(m["F"], map(float, m["beta_fixed"])))}, open(f"{a.out_dir}/model_params.json", "w"), indent=1)
    print(bat[bat.hp_balls >= 40].sort_values("pressure_level", ascending=False).head(15).round(3).to_string())
    print(bow[bow.hp_legal >= 40].sort_values("pressure_level", ascending=False).head(15).round(3).to_string())
