"""
RBO Corrected Experiment
========================
Implements all 5 corrections from the critical analysis:

FIX 1: New baseline "SL-Only" — Kelly + 12% stop-loss, NO behavioral layer
        → Isolates the contribution of the risk score z_t vs. pure stop-loss
FIX 2: Edge threshold raised from 4% to 8%
        → Reduces value bets to a realistic 15-25% of matches
FIX 3: Positive-edge model using closing/max odds as "true" probability proxy
        → Shows RBO behavior when the model actually has an edge
FIX 4: Escalation CONSEQUENCES metric (escalation × stake size) 
        → Correctly measures harm from escalation, not just count
FIX 5: Explicit negative/positive ROI analysis
        → Separate tables for Elo model (negative EV) and closing-odds model (positive EV)

Dataset: MATCHES-top5-15-25.csv (18,010 matches, 2015-2025)
"""

import numpy as np
import pandas as pd
import json
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================
df = pd.read_csv('/Users/renemanassegalekwa/Documents/PHD research/DATA/MATCHES-top5-15-25.csv')
df['MatchDate'] = pd.to_datetime(df['MatchDate'])
df = df.sort_values('MatchDate').reset_index(drop=True)
df = df.dropna(subset=['OddHome','OddDraw','OddAway','HomeElo','AwayElo','FTResult'])

# Fair probabilities from consensus odds (removing margin)
ip_h = 1/df['OddHome']; ip_a = 1/df['OddAway']; ip_d = 1/df['OddDraw']
margin = ip_h + ip_a + ip_d
df['fair_h'] = ip_h / margin
df['fair_a'] = ip_a / margin
df['fair_d'] = ip_d / margin
df['margin'] = margin - 1

# "Sharp" probabilities from MAX odds (best available price → closest to true prob)
ip_mh = 1/df['MaxHome']; ip_ma = 1/df['MaxAway']; ip_md = 1/df['MaxDraw']
margin_max = ip_mh + ip_ma + ip_md
df['sharp_h'] = ip_mh / margin_max
df['sharp_a'] = ip_ma / margin_max
df['sharp_d'] = ip_md / margin_max

LEAGUE_MAP = {
    'D1':  {'name': 'Bundesliga',    'color': '#e41a1c', 'short': 'BUN'},
    'E0':  {'name': 'Premier League','color': '#377eb8', 'short': 'EPL'},
    'F1':  {'name': 'Ligue 1',       'color': '#4daf4a', 'short': 'L1'},
    'SP1': {'name': 'La Liga',       'color': '#ff7f00', 'short': 'LIG'},
    'I1':  {'name': 'Serie A',       'color': '#984ea3', 'short': 'SER'},
}

STRATEGY_COLORS = {
    'Flat':       '#1f77b4',
    'Frac. Kelly':'#ff7f0e',
    'SL-Only':    '#9467bd',   # FIX 1: new baseline
    'CVaR-Kelly': '#2ca02c',
    'RBO':        '#d62728',
}

# ============================================================
# 2. ELO MODEL CALIBRATION
# ============================================================
def build_elo_model(train_df):
    elo_diff = (train_df['HomeElo'] - train_df['AwayElo']).values
    y_h = (train_df['FTResult'] == 'H').astype(float).values
    y_a = (train_df['FTResult'] == 'A').astype(float).values
    ir_h = IsotonicRegression(out_of_bounds='clip'); ir_h.fit(elo_diff, y_h)
    ir_a = IsotonicRegression(out_of_bounds='clip'); ir_a.fit(-elo_diff, y_a)
    return ir_h, ir_a

def apply_elo_model(df_sub, ir_h, ir_a, noise_std=0.035):
    elo_diff = (df_sub['HomeElo'] - df_sub['AwayElo']).values
    ph = ir_h.predict(elo_diff)
    pa = ir_a.predict(-elo_diff)
    pd_p = np.clip(1 - ph - pa, 0.05, 0.40)
    total = ph + pa + pd_p; ph /= total; pa /= total; pd_p /= total
    ph = np.clip(ph + np.random.normal(0, noise_std, len(ph)), 0.01, 0.99)
    pa = np.clip(pa + np.random.normal(0, noise_std, len(pa)), 0.01, 0.99)
    pd_p = np.clip(pd_p + np.random.normal(0, noise_std, len(pd_p)), 0.01, 0.99)
    total = ph + pa + pd_p
    return ph/total, pd_p/total, pa/total

def apply_sharp_model(df_sub, noise_std=0.02):
    """FIX 3: Use max-odds implied probabilities as 'sharp' model.
    These approximate the true probability and give a genuine edge 
    over consensus odds, simulating a skilled bettor."""
    ph = df_sub['sharp_h'].values.copy()
    pa = df_sub['sharp_a'].values.copy()
    pd_p = df_sub['sharp_d'].values.copy()
    # Small noise to avoid perfect hindsight
    ph = np.clip(ph + np.random.normal(0, noise_std, len(ph)), 0.01, 0.99)
    pa = np.clip(pa + np.random.normal(0, noise_std, len(pa)), 0.01, 0.99)
    pd_p = np.clip(pd_p + np.random.normal(0, noise_std, len(pd_p)), 0.01, 0.99)
    total = ph + pa + pd_p
    return ph/total, pd_p/total, pa/total

def identify_value_bets(test_df, edge_threshold=0.08, prob_col_h='model_ph',
                         prob_col_a='model_pa'):
    """FIX 2: edge_threshold default raised to 0.08 (8%)."""
    bets = []
    for _, row in test_df.iterrows():
        edge_h = row[prob_col_h] - row['fair_h']
        edge_a = row[prob_col_a] - row['fair_a']
        if edge_h > edge_threshold and edge_h >= edge_a:
            bets.append({
                'date': row['MatchDate'], 'league': row['Division'],
                'match': f"{row['HomeTeam']} v {row['AwayTeam']}",
                'side': 'H', 'model_prob': row[prob_col_h],
                'fair_prob': row['fair_h'],
                'odds': row['OddHome'], 'max_odds': row['MaxHome'],
                'edge': edge_h, 'won': row['FTResult'] == 'H'
            })
        elif edge_a > edge_threshold:
            bets.append({
                'date': row['MatchDate'], 'league': row['Division'],
                'match': f"{row['HomeTeam']} v {row['AwayTeam']}",
                'side': 'A', 'model_prob': row[prob_col_a],
                'fair_prob': row['fair_a'],
                'odds': row['OddAway'], 'max_odds': row['MaxAway'],
                'edge': edge_a, 'won': row['FTResult'] == 'A'
            })
    return pd.DataFrame(bets)

# ============================================================
# 3. STAKING STRATEGIES (including FIX 1: SL-Only baseline)
# ============================================================
def sim_flat(won, odds, T, B0, frac=0.02):
    bk = np.full(T+1, B0); st = np.zeros(T)
    for t in range(T):
        s = frac * bk[t]; st[t] = frac
        bk[t+1] = bk[t] + s*(odds[t]-1) if won[t] else bk[t] - s
        if bk[t+1] <= 0: bk[t+1:] = 0; break
    return bk, st

def sim_fk(won, odds, prob, T, B0, frac=0.25, cap=0.10):
    bk = np.full(T+1, B0); st = np.zeros(T); esc = 0; prev = []
    esc_cost = 0.0  # FIX 4
    for t in range(T):
        if bk[t] <= 0: bk[t+1:] = 0; break
        b = odds[t]-1
        k = max((prob[t]*b - (1-prob[t]))/b, 0) if b > 0 else 0
        f = min(frac*k, cap); st[t] = f
        prev.append(f)
        if len(prev) > 10: prev.pop(0)
        if len(prev) >= 3 and prev[-1] > prev[-2] > prev[-3] and all(x > 0 for x in prev[-3:]):
            esc += 1
            esc_cost += f * bk[t]  # FIX 4: dollar amount at risk during escalation
        bk[t+1] = bk[t] + f*bk[t]*(odds[t]-1) if won[t] else bk[t] - f*bk[t]
        if bk[t+1] <= 0: bk[t+1:] = 0; break
    return bk, st, esc, esc_cost

def sim_sl_only(won, odds, prob, T, B0, frac=0.25, cap=0.10, sl=0.12):
    """FIX 1: Kelly + pure stop-loss at 12%. No behavioral scoring, no cooldown,
    no loss-chasing suppression. Isolates the stop-loss contribution."""
    bk = np.full(T+1, B0); st = np.zeros(T); esc = 0; prev = []; peak = B0
    esc_cost = 0.0
    for t in range(T):
        if bk[t] <= 0: bk[t+1:] = 0; break
        peak = max(peak, bk[t]); dd = 1 - bk[t]/peak
        if dd >= sl:
            # Pure stop-loss: just stop. No cooldown, no scaling.
            st[t] = 0; bk[t+1] = bk[t]; continue
        b = odds[t]-1
        k = max((prob[t]*b - (1-prob[t]))/b, 0) if b > 0 else 0
        f = min(frac*k, cap); st[t] = f
        prev.append(f)
        if len(prev) > 10: prev.pop(0)
        if len(prev) >= 3 and prev[-1] > prev[-2] > prev[-3] and all(x > 0 for x in prev[-3:]):
            esc += 1
            esc_cost += f * bk[t]
        bk[t+1] = bk[t] + f*bk[t]*(odds[t]-1) if won[t] else bk[t] - f*bk[t]
        if bk[t+1] <= 0: bk[t+1:] = 0; break
    return bk, st, esc, esc_cost

def sim_ck(won, odds, prob, T, B0, frac=0.25, cap=0.10, sl=0.15):
    bk = np.full(T+1, B0); st = np.zeros(T); esc = 0; prev = []; peak = B0
    esc_cost = 0.0
    for t in range(T):
        if bk[t] <= 0: bk[t+1:] = 0; break
        peak = max(peak, bk[t]); dd = 1 - bk[t]/peak
        if dd >= sl: st[t] = 0; bk[t+1] = bk[t]; continue
        b = odds[t]-1
        k = max((prob[t]*b - (1-prob[t]))/b, 0) if b > 0 else 0
        ds = max(0.1, 1 - dd/sl)
        f = min(frac*k*ds, cap); st[t] = f
        prev.append(f)
        if len(prev) > 10: prev.pop(0)
        if len(prev) >= 3 and prev[-1] > prev[-2] > prev[-3] and all(x > 0 for x in prev[-3:]):
            esc += 1
            esc_cost += f * bk[t]
        bk[t+1] = bk[t] + f*bk[t]*(odds[t]-1) if won[t] else bk[t] - f*bk[t]
        if bk[t+1] <= 0: bk[t+1:] = 0; break
    return bk, st, esc, esc_cost

def sim_rbo(won, odds, prob, T, B0, frac=0.25, cap=0.10, sl=0.12, cd_after=5, cd_len=3):
    bk = np.full(T+1, B0); st = np.zeros(T); rs = np.zeros(T)
    esc = 0; prev = []; peak = B0; lstreak = 0; cd = 0
    esc_cost = 0.0  # FIX 4
    for t in range(T):
        if bk[t] <= 0: bk[t+1:] = 0; break
        if cd > 0: cd -= 1; st[t] = 0; bk[t+1] = bk[t]; continue
        peak = max(peak, bk[t]); dd = 1 - bk[t]/peak
        if dd >= sl: st[t] = 0; bk[t+1] = bk[t]; cd = cd_len; continue
        ls_c = min(lstreak/5, 1)*0.4
        sv_c = min(np.std(prev[-10:])*100 if len(prev) >= 3 else 0, 1)*0.3
        esc_c = min(max(0, prev[-1] - np.mean(prev[-5:]))*20 if len(prev) >= 2 else 0, 1)*0.3
        z = ls_c + sv_c + esc_c; rs[t] = z
        b = odds[t]-1
        k = max((prob[t]*b - (1-prob[t]))/b, 0) if b > 0 else 0
        ds = max(0.1, 1 - dd/sl)
        hs = max(0.1, 1 - z)
        dcap = cap * (1 - 0.5*z)
        f = min(frac*k*ds*hs, dcap)
        if lstreak >= 3 and len(prev) >= 3: f = min(f, np.mean(prev[-5:]))
        st[t] = f
        prev.append(f)
        if len(prev) > 20: prev.pop(0)
        if len(prev) >= 3 and f > 0 and prev[-2] > 0 and prev[-3] > 0:
            if prev[-1] > prev[-2] > prev[-3]:
                esc += 1
                esc_cost += f * bk[t]  # FIX 4
        bk[t+1] = bk[t] + f*bk[t]*(odds[t]-1) if won[t] else bk[t] - f*bk[t]
        if bk[t+1] <= 0: bk[t+1:] = 0; break
        if won[t]: lstreak = 0
        else:
            lstreak += 1
            if lstreak >= cd_after: cd = cd_len; lstreak = 0
    return bk, st, esc, rs, esc_cost

# ============================================================
# 4. SIMULATION ENGINE
# ============================================================
N_RUNS = 500
T = 1500
B0 = 1000.0
STRATEGIES = ['Flat', 'Frac. Kelly', 'SL-Only', 'CVaR-Kelly', 'RBO']

def run_experiment(bets_df, label=""):
    """Run all strategies on a given bet pool. Returns results dict and metrics."""
    all_won = bets_df['won'].values.astype(float)
    all_odds = bets_df['odds'].values
    all_prob = bets_df['model_prob'].values
    n_avail = len(all_won)
    T_run = min(T, n_avail * 3)

    R = {n: {'bk': [], 'dd': [], 'roi': [], 'st': [], 'esc': [], 'esc_cost': []}
         for n in STRATEGIES}

    for run in range(N_RUNS):
        idx = np.random.choice(n_avail, T_run, replace=True)
        w, o, p = all_won[idx], all_odds[idx], all_prob[idx]

        bk, st = sim_flat(w, o, T_run, B0)
        pk = np.maximum.accumulate(bk); dd = (1 - bk/np.maximum(pk,1e-8)).max()
        R['Flat']['bk'].append(bk); R['Flat']['dd'].append(dd)
        R['Flat']['roi'].append((bk[-1]-B0)/B0); R['Flat']['st'].append(st)
        R['Flat']['esc'].append(0); R['Flat']['esc_cost'].append(0)

        bk, st, e, ec = sim_fk(w, o, p, T_run, B0)
        pk = np.maximum.accumulate(bk); dd = (1 - bk/np.maximum(pk,1e-8)).max()
        R['Frac. Kelly']['bk'].append(bk); R['Frac. Kelly']['dd'].append(dd)
        R['Frac. Kelly']['roi'].append((bk[-1]-B0)/B0); R['Frac. Kelly']['st'].append(st)
        R['Frac. Kelly']['esc'].append(e); R['Frac. Kelly']['esc_cost'].append(ec)

        # FIX 1: SL-Only baseline
        bk, st, e, ec = sim_sl_only(w, o, p, T_run, B0)
        pk = np.maximum.accumulate(bk); dd = (1 - bk/np.maximum(pk,1e-8)).max()
        R['SL-Only']['bk'].append(bk); R['SL-Only']['dd'].append(dd)
        R['SL-Only']['roi'].append((bk[-1]-B0)/B0); R['SL-Only']['st'].append(st)
        R['SL-Only']['esc'].append(e); R['SL-Only']['esc_cost'].append(ec)

        bk, st, e, ec = sim_ck(w, o, p, T_run, B0)
        pk = np.maximum.accumulate(bk); dd = (1 - bk/np.maximum(pk,1e-8)).max()
        R['CVaR-Kelly']['bk'].append(bk); R['CVaR-Kelly']['dd'].append(dd)
        R['CVaR-Kelly']['roi'].append((bk[-1]-B0)/B0); R['CVaR-Kelly']['st'].append(st)
        R['CVaR-Kelly']['esc'].append(e); R['CVaR-Kelly']['esc_cost'].append(ec)

        bk, st, e, rs, ec = sim_rbo(w, o, p, T_run, B0)
        pk = np.maximum.accumulate(bk); dd = (1 - bk/np.maximum(pk,1e-8)).max()
        R['RBO']['bk'].append(bk); R['RBO']['dd'].append(dd)
        R['RBO']['roi'].append((bk[-1]-B0)/B0); R['RBO']['st'].append(st)
        R['RBO']['esc'].append(e); R['RBO']['esc_cost'].append(ec)

    # Compute metrics
    metrics = {}
    for name in STRATEGIES:
        r = R[name]
        rois = np.array(r['roi'])
        dds = np.array(r['dd'])
        escs = np.array(r['esc'])
        esc_costs = np.array(r['esc_cost'])
        svs = [np.std(s[s>0])*100 if np.any(s>0) else 0 for s in r['st']]
        s95 = np.sort(dds); ci = int(0.95*len(s95)); cvar = s95[ci:].mean()
        metrics[name] = {
            'roi_mean': round(rois.mean()*100, 1),
            'roi_std': round(rois.std()*100, 1),
            'max_dd': round(dds.mean()*100, 1),
            'cvar95': round(cvar, 3),
            'esc': round(escs.mean(), 1),
            'esc_cost': round(esc_costs.mean(), 1),  # FIX 4
            'svol': round(np.mean(svs), 2),
        }
    return R, metrics

# ============================================================
# 5. EXPERIMENT A: ELO MODEL (negative EV) — with FIX 2 (8% threshold)
# ============================================================
print("=" * 70)
print("  EXPERIMENT A: ELO MODEL (edge threshold = 8%)")
print("=" * 70)

# Aggregate experiment (all leagues pooled)
train_all = df[df['MatchDate'] < '2020-07-01']
test_all = df[df['MatchDate'] >= '2020-07-01'].copy()

ir_h_all, ir_a_all = build_elo_model(train_all)
ph, pd_p, pa = apply_elo_model(test_all, ir_h_all, ir_a_all)
test_all['model_ph'] = ph; test_all['model_pd'] = pd_p; test_all['model_pa'] = pa

bets_elo = identify_value_bets(test_all, edge_threshold=0.08)  # FIX 2
print(f"\nAggregate: {len(bets_elo)} value bets ({len(bets_elo)/len(test_all):.1%} of matches)")
print(f"  Win rate: {bets_elo['won'].mean():.3f} | Avg odds: {bets_elo['odds'].mean():.2f}")
print(f"  Avg edge: {bets_elo['edge'].mean()*100:.2f}%")

R_elo_agg, M_elo_agg = run_experiment(bets_elo, "Elo-Aggregate")
for s in STRATEGIES:
    m = M_elo_agg[s]
    print(f"  {s:15s}: ROI={m['roi_mean']:+6.1f}±{m['roi_std']:5.1f}% | DD={m['max_dd']:5.1f}% | "
          f"CVaR={m['cvar95']:.3f} | Esc={m['esc']:5.1f} | EscCost=${m['esc_cost']:8.1f} | SVol={m['svol']:.2f}%")

# Per-league Elo experiment
elo_league_R = {}
elo_league_M = {}
for div_code, li in LEAGUE_MAP.items():
    print(f"\n--- {li['name']} ({div_code}) ---")
    lg = df[df['Division'] == div_code].copy()
    train_lg = lg[lg['MatchDate'] < '2020-07-01']
    test_lg = lg[lg['MatchDate'] >= '2020-07-01'].copy()
    ir_h, ir_a = build_elo_model(train_lg)
    ph, pd_p, pa = apply_elo_model(test_lg, ir_h, ir_a)
    test_lg['model_ph'] = ph; test_lg['model_pd'] = pd_p; test_lg['model_pa'] = pa
    bets_lg = identify_value_bets(test_lg, edge_threshold=0.08)  # FIX 2
    if len(bets_lg) == 0:
        print("  No value bets found!")
        continue
    print(f"  Bets: {len(bets_lg)} ({len(bets_lg)/len(test_lg):.1%}) | WR={bets_lg['won'].mean():.3f} | Odds={bets_lg['odds'].mean():.2f}")
    R, M = run_experiment(bets_lg)
    elo_league_R[div_code] = R
    elo_league_M[div_code] = M
    # Store data stats
    M['data'] = {'n_bets': len(bets_lg), 'win_rate': round(bets_lg['won'].mean(), 3),
                 'avg_odds': round(bets_lg['odds'].mean(), 2), 'avg_edge': round(bets_lg['edge'].mean()*100, 2),
                 'pct': round(len(bets_lg)/len(test_lg)*100, 1)}
    for s in ['SL-Only', 'CVaR-Kelly', 'RBO']:
        m = M[s]
        print(f"    {s:12s}: ROI={m['roi_mean']:+6.1f}% DD={m['max_dd']:5.1f}% CVaR={m['cvar95']:.3f} "
              f"Esc={m['esc']:5.1f} EscCost=${m['esc_cost']:7.1f} SVol={m['svol']:.2f}%")

# ============================================================
# 6. EXPERIMENT B: SHARP/CLOSING-ODDS MODEL (positive EV) — FIX 3
# ============================================================
print("\n" + "=" * 70)
print("  EXPERIMENT B: SHARP MODEL (closing/max odds → positive EV)")
print("=" * 70)

test_all_b = df[df['MatchDate'] >= '2020-07-01'].copy()
ph_s, pd_s, pa_s = apply_sharp_model(test_all_b, noise_std=0.02)
test_all_b['model_ph'] = ph_s; test_all_b['model_pd'] = pd_s; test_all_b['model_pa'] = pa_s

bets_sharp = identify_value_bets(test_all_b, edge_threshold=0.03)  # lower threshold ok for sharp model
print(f"\nAggregate: {len(bets_sharp)} value bets ({len(bets_sharp)/len(test_all_b):.1%} of matches)")
print(f"  Win rate: {bets_sharp['won'].mean():.3f} | Avg odds: {bets_sharp['odds'].mean():.2f}")
print(f"  Avg edge: {bets_sharp['edge'].mean()*100:.2f}%")

R_sharp_agg, M_sharp_agg = run_experiment(bets_sharp, "Sharp-Aggregate")
for s in STRATEGIES:
    m = M_sharp_agg[s]
    print(f"  {s:15s}: ROI={m['roi_mean']:+6.1f}±{m['roi_std']:5.1f}% | DD={m['max_dd']:5.1f}% | "
          f"CVaR={m['cvar95']:.3f} | Esc={m['esc']:5.1f} | EscCost=${m['esc_cost']:8.1f} | SVol={m['svol']:.2f}%")

# Per-league sharp experiment
sharp_league_R = {}
sharp_league_M = {}
for div_code, li in LEAGUE_MAP.items():
    print(f"\n--- {li['name']} ({div_code}) ---")
    lg = df[df['Division'] == div_code].copy()
    test_lg = lg[lg['MatchDate'] >= '2020-07-01'].copy()
    ph_s, pd_s, pa_s = apply_sharp_model(test_lg, noise_std=0.02)
    test_lg['model_ph'] = ph_s; test_lg['model_pd'] = pd_s; test_lg['model_pa'] = pa_s
    bets_lg = identify_value_bets(test_lg, edge_threshold=0.03)
    if len(bets_lg) == 0:
        print("  No value bets found!")
        continue
    print(f"  Bets: {len(bets_lg)} ({len(bets_lg)/len(test_lg):.1%}) | WR={bets_lg['won'].mean():.3f} | Odds={bets_lg['odds'].mean():.2f}")
    R, M = run_experiment(bets_lg)
    sharp_league_R[div_code] = R
    sharp_league_M[div_code] = M
    M['data'] = {'n_bets': len(bets_lg), 'win_rate': round(bets_lg['won'].mean(), 3),
                 'avg_odds': round(bets_lg['odds'].mean(), 2), 'avg_edge': round(bets_lg['edge'].mean()*100, 2),
                 'pct': round(len(bets_lg)/len(test_lg)*100, 1)}
    for s in ['SL-Only', 'CVaR-Kelly', 'RBO']:
        m = M[s]
        print(f"    {s:12s}: ROI={m['roi_mean']:+6.1f}% DD={m['max_dd']:5.1f}% CVaR={m['cvar95']:.3f} "
              f"Esc={m['esc']:5.1f} EscCost=${m['esc_cost']:7.1f} SVol={m['svol']:.2f}%")

# ============================================================
# 7. SAVE ALL METRICS
# ============================================================
all_metrics = {
    'elo_aggregate': M_elo_agg,
    'elo_aggregate_data': {
        'n_bets': len(bets_elo), 'win_rate': round(bets_elo['won'].mean(), 3),
        'avg_odds': round(bets_elo['odds'].mean(), 2),
        'pct_matches': round(len(bets_elo)/len(test_all)*100, 1),
    },
    'sharp_aggregate': M_sharp_agg,
    'sharp_aggregate_data': {
        'n_bets': len(bets_sharp), 'win_rate': round(bets_sharp['won'].mean(), 3),
        'avg_odds': round(bets_sharp['odds'].mean(), 2),
        'pct_matches': round(len(bets_sharp)/len(test_all_b)*100, 1),
    },
    'elo_by_league': elo_league_M,
    'sharp_by_league': sharp_league_M,
}
with open('figures/metrics_corrected.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)

# ============================================================
# 8. GENERATE FIGURES
# ============================================================
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'figure.dpi': 300,
    'axes.linewidth': 0.5, 'grid.linewidth': 0.3,
})

league_order = ['E0', 'D1', 'SP1', 'I1', 'F1']

# --- FIG 7: AGGREGATE BANKROLL - ELO vs SHARP (2-panel) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)

for sname in STRATEGIES:
    bks = np.array(R_elo_agg[sname]['bk'])
    med = np.median(bks, axis=0)
    ax1.plot(np.arange(len(med)), med, label=sname, color=STRATEGY_COLORS[sname], lw=1.2)
ax1.axhline(B0, ls='--', color='gray', lw=0.4, alpha=0.5)
ax1.set_title('(a) Elo Model (Negative EV)', fontsize=10, fontweight='bold')
ax1.set_xlabel('Bet Index $t$'); ax1.set_ylabel('Bankroll ($)')
ax1.legend(fontsize=7, loc='center left'); ax1.grid(True, alpha=0.2)

for sname in STRATEGIES:
    bks = np.array(R_sharp_agg[sname]['bk'])
    med = np.median(bks, axis=0)
    ax2.plot(np.arange(len(med)), med, label=sname, color=STRATEGY_COLORS[sname], lw=1.2)
ax2.axhline(B0, ls='--', color='gray', lw=0.4, alpha=0.5)
ax2.set_title('(b) Sharp Model (Positive EV)', fontsize=10, fontweight='bold')
ax2.set_xlabel('Bet Index $t$')
ax2.legend(fontsize=7, loc='upper left'); ax2.grid(True, alpha=0.2)

fig.suptitle('Median Bankroll Trajectories: Negative vs. Positive Expected Value', fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig('figures/fig7c_bankroll_dual.pdf', bbox_inches='tight')
plt.close()
print("Fig 7c saved!")

# --- FIG 8c: SL-ONLY vs RBO ISOLATION (key FIX 1 figure) ---
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

# Panel A: Drawdown distribution comparison (SL-Only vs RBO)
strats_comp = ['SL-Only', 'CVaR-Kelly', 'RBO']
colors_comp = [STRATEGY_COLORS[s] for s in strats_comp]
dd_data_elo = [np.array(R_elo_agg[s]['dd'])*100 for s in strats_comp]
dd_data_sharp = [np.array(R_sharp_agg[s]['dd'])*100 for s in strats_comp]

bp = axes[0].boxplot(dd_data_elo, labels=strats_comp, patch_artist=True, widths=0.5)
for patch, c in zip(bp['boxes'], colors_comp): patch.set_facecolor(c); patch.set_alpha(0.6)
axes[0].set_ylabel('Max Drawdown (%)', fontsize=8)
axes[0].set_title('(a) Drawdown: Elo Model', fontsize=9)
axes[0].grid(True, alpha=0.2, axis='y')

# Panel B: Same for sharp model
bp2 = axes[1].boxplot(dd_data_sharp, labels=strats_comp, patch_artist=True, widths=0.5)
for patch, c in zip(bp2['boxes'], colors_comp): patch.set_facecolor(c); patch.set_alpha(0.6)
axes[1].set_ylabel('Max Drawdown (%)', fontsize=8)
axes[1].set_title('(b) Drawdown: Sharp Model', fontsize=9)
axes[1].grid(True, alpha=0.2, axis='y')

# Panel C: Escalation COST (FIX 4)
x = np.arange(3)
width = 0.35
elo_esc_cost = [M_elo_agg[s]['esc_cost'] for s in strats_comp]
sharp_esc_cost = [M_sharp_agg[s]['esc_cost'] for s in strats_comp]
axes[2].bar(x - width/2, elo_esc_cost, width, label='Elo Model', color='#4a90d9', alpha=0.8)
axes[2].bar(x + width/2, sharp_esc_cost, width, label='Sharp Model', color='#e8913a', alpha=0.8)
axes[2].set_xticks(x); axes[2].set_xticklabels(strats_comp, fontsize=8)
axes[2].set_ylabel('Escalation Cost ($)', fontsize=8)
axes[2].set_title('(c) Escalation Exposure ($)', fontsize=9)
axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.2, axis='y')

fig.suptitle('Isolating the Behavioral Layer: SL-Only vs. CVaR-Kelly vs. RBO', fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig('figures/fig8c_sl_isolation.pdf', bbox_inches='tight')
plt.close()
print("Fig 8c saved!")

# --- FIG 9c: Per-League Trajectories (Elo model, 5 strategies) ---
fig, axes = plt.subplots(1, 5, figsize=(13, 2.8), sharey=True)
for ax, lcode in zip(axes, league_order):
    if lcode not in elo_league_R: continue
    R = elo_league_R[lcode]
    for sname in STRATEGIES:
        bks = np.array(R[sname]['bk'])
        med = np.median(bks, axis=0)
        ax.plot(np.arange(len(med)), med, label=sname, color=STRATEGY_COLORS[sname], lw=1)
    ax.axhline(B0, ls='--', color='gray', lw=0.4, alpha=0.5)
    ax.set_title(LEAGUE_MAP[lcode]['name'], fontsize=9, fontweight='bold')
    ax.set_xlabel('Bet $t$', fontsize=7); ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    if ax == axes[0]: ax.set_ylabel('Bankroll ($)', fontsize=8)
axes[-1].legend(fontsize=5.5, loc='center right', bbox_to_anchor=(1.6, 0.5))
fig.suptitle('Elo Model: Median Bankroll by League (edge threshold = 8%, 500 trials)', fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig('figures/fig9c_league_bankrolls_elo.pdf', bbox_inches='tight')
plt.close()
print("Fig 9c saved!")

# --- FIG 10c: Per-League Trajectories (Sharp model, 5 strategies) ---
fig, axes = plt.subplots(1, 5, figsize=(13, 2.8), sharey=True)
for ax, lcode in zip(axes, league_order):
    if lcode not in sharp_league_R: continue
    R = sharp_league_R[lcode]
    for sname in STRATEGIES:
        bks = np.array(R[sname]['bk'])
        med = np.median(bks, axis=0)
        ax.plot(np.arange(len(med)), med, label=sname, color=STRATEGY_COLORS[sname], lw=1)
    ax.axhline(B0, ls='--', color='gray', lw=0.4, alpha=0.5)
    ax.set_title(LEAGUE_MAP[lcode]['name'], fontsize=9, fontweight='bold')
    ax.set_xlabel('Bet $t$', fontsize=7); ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    if ax == axes[0]: ax.set_ylabel('Bankroll ($)', fontsize=8)
axes[-1].legend(fontsize=5.5, loc='center right', bbox_to_anchor=(1.6, 0.5))
fig.suptitle('Sharp Model: Median Bankroll by League (positive EV, 500 trials)', fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig('figures/fig10c_league_bankrolls_sharp.pdf', bbox_inches='tight')
plt.close()
print("Fig 10c saved!")

# --- FIG 11c: Harm metrics grouped bars - SL-Only vs CVaR-Kelly vs RBO ---
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
x = np.arange(len(league_order))
width = 0.25
strats3 = ['SL-Only', 'CVaR-Kelly', 'RBO']
cols3 = [STRATEGY_COLORS[s] for s in strats3]

# Panel A: Stake Volatility by league
for i, sname in enumerate(strats3):
    vals = [elo_league_M[l][sname]['svol'] for l in league_order if l in elo_league_M]
    axes[0].bar(x[:len(vals)] + i*width, vals, width, label=sname, color=cols3[i], alpha=0.8)
axes[0].set_xticks(x + width); axes[0].set_xticklabels([LEAGUE_MAP[l]['short'] for l in league_order], fontsize=8)
axes[0].set_ylabel('Stake Volatility (%)', fontsize=8)
axes[0].set_title('(a) Stake Volatility (Elo)', fontsize=9)
axes[0].legend(fontsize=6); axes[0].grid(True, alpha=0.2, axis='y')

# Panel B: Escalation COST by league (FIX 4)
for i, sname in enumerate(strats3):
    vals = [elo_league_M[l][sname]['esc_cost'] for l in league_order if l in elo_league_M]
    axes[1].bar(x[:len(vals)] + i*width, vals, width, label=sname, color=cols3[i], alpha=0.8)
axes[1].set_xticks(x + width); axes[1].set_xticklabels([LEAGUE_MAP[l]['short'] for l in league_order], fontsize=8)
axes[1].set_ylabel('Escalation Cost ($)', fontsize=8)
axes[1].set_title('(b) Escalation Exposure (Elo)', fontsize=9)
axes[1].legend(fontsize=6); axes[1].grid(True, alpha=0.2, axis='y')

# Panel C: CVaR by league
for i, sname in enumerate(strats3):
    vals = [elo_league_M[l][sname]['cvar95'] for l in league_order if l in elo_league_M]
    axes[2].bar(x[:len(vals)] + i*width, vals, width, label=sname, color=cols3[i], alpha=0.8)
axes[2].set_xticks(x + width); axes[2].set_xticklabels([LEAGUE_MAP[l]['short'] for l in league_order], fontsize=8)
axes[2].set_ylabel(r'CVaR$_{0.95}$', fontsize=8)
axes[2].set_title('(c) Tail Risk (Elo)', fontsize=9)
axes[2].legend(fontsize=6); axes[2].grid(True, alpha=0.2, axis='y')

fig.suptitle('Harm Metrics: SL-Only vs. CVaR-Kelly vs. RBO (Elo Model, edge ≥ 8%)', fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig('figures/fig11c_harm_league.pdf', bbox_inches='tight')
plt.close()
print("Fig 11c saved!")

# --- FIG 12c: Heatmap - RBO performance Elo vs Sharp side by side ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))

cmap = LinearSegmentedColormap.from_list('rg', ['#ff6b6b', '#ffd93d', '#6bcb77'])
metrics_names = ['ROI (%)', 'Max DD (%)', r'CVaR$_{0.95}$', 'Esc Cost ($)', 'SVol (%)']

for ax, league_M, title in [(ax1, elo_league_M, 'RBO — Elo Model (neg. EV)'),
                              (ax2, sharp_league_M, 'RBO — Sharp Model (pos. EV)')]:
    valid_leagues = [l for l in league_order if l in league_M]
    data_matrix = []
    for lcode in valid_leagues:
        m = league_M[lcode]['RBO']
        data_matrix.append([m['roi_mean'], m['max_dd'], m['cvar95'], m['esc_cost'], m['svol']])
    data_matrix = np.array(data_matrix)

    norm_data = np.zeros_like(data_matrix)
    for j in range(data_matrix.shape[1]):
        col = data_matrix[:, j]
        rng = col.max() - col.min()
        if rng < 1e-8: norm_data[:, j] = 0.5
        elif j == 0: norm_data[:, j] = (col - col.min()) / rng
        else: norm_data[:, j] = 1 - (col - col.min()) / rng

    im = ax.imshow(norm_data, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(metrics_names)))
    ax.set_yticks(np.arange(len(valid_leagues)))
    ax.set_xticklabels(metrics_names, fontsize=7)
    ax.set_yticklabels([LEAGUE_MAP[l]['name'] for l in valid_leagues], fontsize=8)
    for i in range(len(valid_leagues)):
        for j in range(len(metrics_names)):
            val = data_matrix[i, j]
            fmt = f"{val:.1f}" if j not in [2] else f"{val:.3f}"
            ax.text(j, i, fmt, ha='center', va='center', fontsize=8, fontweight='bold',
                    color='black' if norm_data[i,j] > 0.3 else 'white')
    ax.set_title(title, fontsize=9, pad=8)

fig.suptitle('RBO Performance by League: Negative vs. Positive EV Markets', fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig('figures/fig12c_heatmap_dual.pdf', bbox_inches='tight')
plt.close()
print("Fig 12c saved!")

# --- FIG 13c: ROI Distribution - sharp model (shows RBO preserves gains) ---
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for ax, sname in zip(axes, ['SL-Only', 'CVaR-Kelly', 'RBO']):
    rois = np.array(R_sharp_agg[sname]['roi']) * 100
    ax.hist(rois, bins=40, color=STRATEGY_COLORS[sname], alpha=0.7, edgecolor='white', lw=0.3)
    ax.axvline(0, ls='--', color='black', lw=0.5)
    ax.axvline(rois.mean(), ls='-', color='red', lw=1, label=f'Mean={rois.mean():.1f}%')
    ax.set_xlabel('ROI (%)', fontsize=8); ax.set_title(sname, fontsize=10, fontweight='bold')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2); ax.tick_params(labelsize=7)
    ax.set_ylabel('Count', fontsize=8)
fig.suptitle('ROI Distributions Under Positive-EV Sharp Model (500 Trials)', fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig('figures/fig13c_roi_dist_sharp.pdf', bbox_inches='tight')
plt.close()
print("Fig 13c saved!")

# ============================================================
# 10. NEW FIGURES: TEMPORAL ROBUSTNESS & MONITORING
# ============================================================

# --- FIG 14: Rolling ROI (Temporal Robustness) ---
# Window = 200 bets. Using R_sharp_agg (Positive EV) to checking consistency.
window = 200
fig, ax = plt.subplots(figsize=(10, 4))
colors = STRATEGY_COLORS

for sname in STRATEGIES:
    # Use the median bankroll trajectory to compute a representative rolling ROI
    bks = np.array(R_sharp_agg[sname]['bk'])
    med_bk = np.median(bks, axis=0) # shape (T+1,)
    
    # Calculate rolling ROI: (B_t - B_{t-w}) / B_{t-w}
    # We'll calculate it for t from window to T
    rolling_roi = []
    x_axis = []
    
    for t in range(window, len(med_bk)):
        start_bk = med_bk[t-window]
        curr_bk = med_bk[t]
        if start_bk > 1: # Avoid division by zero or tiny numbers
             roi = (curr_bk - start_bk) / start_bk
             rolling_roi.append(roi)
             x_axis.append(t)
    
    if rolling_roi:
        ax.plot(x_axis, np.array(rolling_roi)*100, label=sname, color=colors[sname], lw=1.2)

ax.axhline(0, ls='--', color='black', lw=0.8)
ax.set_ylabel(f'Rolling ROI (Window={window} bets) [%]')
ax.set_xlabel('Bet Index $t$')
ax.set_title('Figure 14: Temporal Robustness - Rolling ROI Trajectories (Sharp Model)', fontsize=10, fontweight='bold')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('figures/fig14_rolling_roi.pdf', bbox_inches='tight')
plt.close()
print("Fig 14 saved!")

# --- FIG 15: Behavioral Monitoring (Single Run) ---
# We run ONE simulation with RBO to get traces of Z_t, S_t
print("Generating Single-Run Visualization...")
# Use Sharp model data
idxs = np.arange(len(bets_sharp))
# Pick a random successful sequence or fixed seed for reproducibility
np.random.seed(42) 
# Create a sub-slice of 1000 bets
sample_bets = bets_sharp.iloc[:1000].reset_index(drop=True)
w = sample_bets['won'].values.astype(float)
o = sample_bets['odds'].values
p = sample_bets['model_prob'].values
T_sim = len(w)

# Run RBO sim
bk, st, esc, rs, esc_cost = sim_rbo(w, o, p, T_sim, B0)

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Top: Bankroll
axes[0].plot(bk, color=STRATEGY_COLORS['RBO'], lw=1.5)
axes[0].set_ylabel('Bankroll ($)')
axes[0].set_title('(a) Bankroll Evolution', fontsize=10)
axes[0].grid(True, alpha=0.3)

# Middle: Stake Size
axes[1].bar(np.arange(len(st)), st*100, color='black', alpha=0.7, width=1.0) # as % of bankroll? No, st is fraction. 
# st is fraction of current bankroll. Let's plot actual $ or fraction. 
# Code uses st[t] = fraction. Let's plot fraction %.
axes[1].set_ylabel('Stake Size (%)')
axes[1].set_title('(b) Stake Sizing Decisions', fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, max(st)*100 * 1.2)

# Bottom: Risk Score Z_t
axes[2].plot(rs, color='#d62728', lw=1.2, label='$Z_t$ (Risk Score)')
axes[2].axhline(0.5, ls='--', color='orange', lw=1, label='Warning (0.5)')
axes[2].axhline(0.8, ls='--', color='red', lw=1, label='Critical (0.8)')
axes[2].fill_between(np.arange(len(rs)), 0, rs, color='#d62728', alpha=0.1)
axes[2].set_ylabel('Risk Score $Z_t$')
axes[2].set_xlabel('Bet Index $t$')
axes[2].set_title('(c) Behavioral Risk Monitoring', fontsize=10)
axes[2].legend(loc='upper right', fontsize=8)
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(0, 1.1)

fig.suptitle('Figure 15: Single-Run RBO Monitoring Dashboard', fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig('figures/fig15_behavioral_monitoring.pdf', bbox_inches='tight')
plt.close()
print("Fig 15 saved!")


# ============================================================
# 9. SUMMARY TABLES
# ============================================================
print("\n" + "=" * 90)
print("TABLE II: AGGREGATE RESULTS — ELO MODEL (Negative EV, edge ≥ 8%)")
print("=" * 90)
print(f"Value bets: {len(bets_elo)} ({len(bets_elo)/len(test_all):.1%} of matches) — WR={bets_elo['won'].mean():.3f}")
print(f"{'Strategy':<15} {'ROI%':>8} {'±':>5} {'MaxDD%':>8} {'CVaR':>7} {'Esc':>6} {'EscCost$':>10} {'SVol%':>7}")
print("-" * 70)
for s in STRATEGIES:
    m = M_elo_agg[s]
    print(f"{s:<15} {m['roi_mean']:>+8.1f} {m['roi_std']:>5.1f} {m['max_dd']:>8.1f} "
          f"{m['cvar95']:>7.3f} {m['esc']:>6.1f} {m['esc_cost']:>10.1f} {m['svol']:>7.2f}")

print(f"\n{'='*90}")
print("TABLE III: AGGREGATE RESULTS — SHARP MODEL (Positive EV, edge ≥ 3%)")
print("=" * 90)
print(f"Value bets: {len(bets_sharp)} ({len(bets_sharp)/len(test_all_b):.1%} of matches) — WR={bets_sharp['won'].mean():.3f}")
print(f"{'Strategy':<15} {'ROI%':>8} {'±':>5} {'MaxDD%':>8} {'CVaR':>7} {'Esc':>6} {'EscCost$':>10} {'SVol%':>7}")
print("-" * 70)
for s in STRATEGIES:
    m = M_sharp_agg[s]
    print(f"{s:<15} {m['roi_mean']:>+8.1f} {m['roi_std']:>5.1f} {m['max_dd']:>8.1f} "
          f"{m['cvar95']:>7.3f} {m['esc']:>6.1f} {m['esc_cost']:>10.1f} {m['svol']:>7.2f}")

print("\n\n✅ All corrected experiments complete!")
