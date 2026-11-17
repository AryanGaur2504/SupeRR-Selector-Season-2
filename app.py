import json, os, hashlib
import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
import engine

HERE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="Pressure Performance Rankings", page_icon="🏏", layout="wide")

# ---------- data ----------
@st.cache_data(show_spinner="Loading data…")
def load(src_bytes_key, src):
    return engine.load_raw(src)

@st.cache_data(show_spinner=False)
def prep(raw_key, _raw, hp):
    return engine.prepare(_raw, hp)

@st.cache_data(show_spinner="Fitting the model on this selection…")
def run(raw_key, _d, unit, key, tau, team_names, reest, min_hp, min_m, gate):
    bat, bow, m = engine.rate(_d, unit, list(tau), team_names, reest, min_hp, min_m, gate)
    return bat, bow, m["tau"].tolist()

TAU0 = json.load(open(f"{HERE}/data/model_params.json"))["tau_b_d_w_e"]
TEAMS = dict(pd.read_csv(f"{HERE}/data/teams.csv").values)

st.sidebar.header("Data")
up = st.sidebar.file_uploader("Use your own ball_pressure file (optional)", type=["csv", "gz"],
                              help="Same columns as the Phase 2 output ball_pressure.csv.gz. Leave empty to use the bundled data.")
if up is not None:
    raw_key = f"up:{up.name}:{up.size}"; raw = load(raw_key, up)
else:
    raw_key = "bundled"; raw = load(raw_key, f"{HERE}/data/ball_pressure.csv.gz")

st.sidebar.header("Filters")
years_all = sorted(raw.year.unique()); comps_all = sorted(raw.comp_group.unique())
years = st.sidebar.multiselect("Years", years_all, default=years_all)
comps = st.sidebar.multiselect("Competitions", comps_all, default=comps_all,
                               help="IPL, T20I (internationals) and T20 (other leagues), as grouped in the data.")
team_ids_all = sorted(set(raw.batting_team_id) | set(raw.bowling_team_id), key=lambda i: TEAMS.get(i, str(i)))
sel_teams = st.sidebar.multiselect("Teams", team_ids_all, format_func=lambda i: TEAMS.get(i, str(i)))
team_mode = st.sidebar.radio("Team filter works as", ["Show only these teams' players", "Fit the model only on their matches"],
                             help="Showing keeps the full opposition picture and just filters the list. Fitting uses only matches these teams played.",
                             disabled=not sel_teams)

with st.sidebar.expander("Model settings", expanded=False):
    hp_thr = st.slider("High-pressure cut-off (Pressure Index)", 1.0, 3.0, 1.5, 0.25)
    min_hp = st.number_input("Minimum high-pressure balls", 10, 500, 100, 10)
    min_m = st.number_input("Minimum matches", 1, 100, 20, 1)
    gate = st.checkbox("Require positive net win-probability under pressure", True)
    reest = st.checkbox("Re-estimate shrinkage from this selection (slower)", False,
                        help="Off: uses the variance components fitted on all data (about 2 seconds). On: refits them, which can take minutes.")
    top_n = st.slider("Players to show", 5, 50, 10)

excluded = st.session_state.setdefault("excluded", set())
if st.sidebar.button("Clear match exclusions", disabled=not excluded):
    excluded.clear(); st.rerun()

if not years or not comps:
    st.warning("Select at least one year and one competition."); st.stop()

d_all, unit = prep(raw_key, raw, hp_thr)
fit_teams = tuple(sel_teams) if (sel_teams and team_mode.startswith("Fit")) else ()
d = engine.subset(d_all, years, comps, excluded, fit_teams)
if d.match_id.nunique() < 15:
    st.warning(f"Only {d.match_id.nunique()} matches match these filters. Widen them to get a stable fit."); st.stop()

key = hashlib.md5(json.dumps([sorted(map(int, years)), sorted(comps), sorted(map(int, excluded)), sorted(map(int, fit_teams)), hp_thr]).encode()).hexdigest()
bat, bow, tau_used = run(raw_key, d, unit, key, tuple(TAU0), TEAMS, reest, int(min_hp), int(min_m), gate)

# ---------- header ----------
st.title("Pressure performance rankings")
st.caption("Who performs best when the match is tight, after adjusting for context, opposition and sample size. Change the filters on the left and the model refits on that slice.")
c = st.columns(5)
c[0].metric("Matches", d.match_id.nunique()); c[1].metric("Deliveries", f"{len(d):,}")
c[2].metric("High-pressure deliveries", f"{int(d.hp.sum()):,}")
c[3].metric("Eligible batters", int(bat.elig.sum())); c[4].metric("Eligible bowlers", int(bow.elig.sum()))
st.info("In the full data, no batter or bowler shows a reliable personal “clutch” effect, so ratings show who is best when the match is tight, not who rises. "
        "Treat small filtered slices with extra caution: fewer matches means wider uncertainty.")

# ---------- rankings ----------
def view(tab, role):
    t = tab[tab.elig].sort_values("pressure_level", ascending=False).copy()
    t.insert(0, "overall_rank", range(1, len(t) + 1))
    if sel_teams and not fit_teams:
        keep = t.team_ids.map(lambda ids: isinstance(ids, list) and any(i in ids for i in sel_teams))
        t = t[keep]
    return t

def boot_block(role, tab):
    k = (key, role, reest, int(min_hp), int(min_m), gate)
    store = st.session_state.setdefault("boot", {})
    with st.expander("Rank stability (bootstrap)"):
        reps = st.slider("Resamples", 5, 60, 20, 5, key=f"reps{role}", help="Each resample takes about as long as one fit. 20 is a quick look, 60 matches the report.")
        if st.button("Run bootstrap for batters and bowlers", key=f"bb{role}"):
            bar = st.progress(0.0)
            store[k[:1] + k[2:]] = engine.bootstrap(d, tau_used, bat.player_id[bat.elig].tolist(), bow.player_id[bow.elig].tolist(), reps, progress=bar.progress)
            bar.empty()
        if k[:1] + k[2:] in store: st.success("Showing top-10 chance and rank range in the table below.")
    return store.get(k[:1] + k[2:])

boot = None
tabs = st.tabs(["Batters", "Bowlers", "Player detail", "Matches (edit data)", "How it works"])
for tb, (role, tab) in zip(tabs[:2], [("Batters", bat), ("Bowlers", bow)]):
    with tb:
        if role == "Batters": boot = boot_block(role, tab)
        t = view(tab, role)
        if boot is not None:
            t = t.merge(boot[0 if role == "Batters" else 1], on="player_id", how="left")
        if t.empty:
            st.warning("No eligible players. Lower the minimum balls or matches, or widen the filters."); continue
        if len(t) < top_n:
            st.caption(f"Only {len(t)} players meet the minimums on this selection. Lower the minimum balls or matches in Model settings to see more.")
        t = t.reset_index(drop=True); t.insert(0, "rank", range(1, len(t) + 1))
        show = t.head(top_n)
        fig = go.Figure(go.Bar(x=show.rating[::-1], y=show.name[::-1], orientation="h", error_x=dict(type="data", array=show.rating_sd[::-1]),
                               marker_color="#0b6e8e", hovertemplate="%{y}: %{x:.1f}<extra></extra>"))
        fig.update_layout(height=max(280, 34 * len(show) + 80), margin=dict(l=0, r=10, t=10, b=30), xaxis_title="Rating (win-probability points per 100 high-pressure balls, ±1 SD)")
        st.plotly_chart(fig, width="stretch")
        if role == "Batters":
            cols = {"rank": "Rank", "name": "Batter", "team": "Team", "rating": "Rating", "hp_n": "Pressure balls", "matches": "Matches", "sr_hp": "SR pressure", "sr_usual": "SR usual", "bpd_hp": "Balls/dismissal pressure", "bpd_usual": "Balls/dismissal usual"}
        else:
            cols = {"rank": "Rank", "name": "Bowler", "team": "Team", "rating": "Rating", "hp_n": "Pressure balls", "matches": "Matches", "econ_hp": "Economy pressure", "econ_usual": "Economy usual", "bpw_hp": "Balls/wicket pressure", "bpw_usual": "Balls/wicket usual"}
        if "p_top10" in show: cols.update({"p_top10": "Top-10 chance", "rank_lo": "Rank 5%", "rank_hi": "Rank 95%"})
        st.dataframe(show[list(cols)].rename(columns=cols), hide_index=True, width="stretch",
                     column_config={"Rating": st.column_config.NumberColumn(format="%.1f"), "Top-10 chance": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=1)} if "p_top10" in show else {"Rating": st.column_config.NumberColumn(format="%.1f")})
        st.download_button(f"Download all eligible {role.lower()} (CSV)", t.drop(columns=["team_ids"]).to_csv(index=False), f"{role.lower()}_pressure_rankings.csv", "text/csv")

# ---------- player detail ----------
with tabs[2]:
    role = st.radio("Role", ["Batters", "Bowlers"], horizontal=True)
    tab = bat if role == "Batters" else bow
    pool = tab[tab.hp_n > 0].sort_values("pressure_level", ascending=False)
    pid = st.selectbox("Player", pool.player_id, format_func=lambda i: pool.set_index("player_id").name[i])
    r = tab.set_index("player_id").loc[pid]
    a, b2, c2, e = st.columns(4)
    a.metric("Rating under pressure", f"{r.rating:.1f}"); b2.metric("Usual level", f"{r.usual_rating:.1f}")
    b2.caption("Same scale, all situations"); c2.metric("Pressure change", f"{r.delta * unit:+.2f}")
    e.metric("Eligible", "Yes" if r.elig else "No")
    if not r.elig:
        st.caption("Not eligible: " + ("fewer than the minimum balls or matches" if not r.sample_ok else "negative net win-probability under pressure"))
    st.dataframe(engine.season_profile(d, role, pid), hide_index=True, width="stretch")

# ---------- edit data ----------
with tabs[3]:
    st.write("Untick a match to leave it out of every calculation (for example a rain-affected game), then apply.")
    base = d_all[d_all.year.isin(years) & d_all.comp_group.isin(comps)]
    mt = base[base.innings == 1].drop_duplicates("match_id")[["match_id", "match_date", "comp_group", "batting_team_id", "bowling_team_id"]].copy()
    mt["match"] = mt.batting_team_id.map(lambda i: TEAMS.get(i, i)).astype(str) + " v " + mt.bowling_team_id.map(lambda i: TEAMS.get(i, i)).astype(str)
    mt["include"] = ~mt.match_id.isin(list(excluded))
    mt = mt.sort_values("match_date", ascending=False)[["include", "match_date", "comp_group", "match", "match_id"]]
    with st.form("matches"):
        ed = st.data_editor(mt, hide_index=True, width="stretch", height=380, disabled=["match_date", "comp_group", "match", "match_id"])
        if st.form_submit_button("Apply match selection"):
            shown = set(ed.match_id); off = set(ed.match_id[~ed.include])
            st.session_state["excluded"] = (excluded - shown) | off; st.rerun()
    st.caption(f"{len(excluded)} matches currently excluded.")

with tabs[4]:
    st.markdown("""
**Pressure** is a continuous index built from leverage (how much a delivery can swing the match). A delivery counts as high pressure at or above the cut-off.
**Per-ball score** is win-probability change divided by leverage, so a boundary in a dead over and one in a tight finish are on the same scale.
**Model**: batters and bowlers are fitted together, adjusting for phase, innings, batting position, competition, pace or spin and the opposition faced, with shrinkage toward the league average.
**Eligibility**: minimum high-pressure balls and matches, and (optionally) a positive net win-probability contribution under pressure.

**What changes when you filter.** The model refits on the selected matches, so ratings and ranks are relative to that slice. The settings in the sidebar by default reproduce the report.
Small slices give noisier ratings, so lower the minimums only if you accept that.
""")
