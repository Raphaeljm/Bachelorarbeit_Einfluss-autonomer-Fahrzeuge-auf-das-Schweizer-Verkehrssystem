# -*- coding: utf-8 -*-
"""
04_model_pipeline

Ziel
----
- Binäres Logit-Modell (MIV vs. ÖV) schätzen
- Wert der Reisezeitersparnis (VTTS) berechnen
- Parameter-Transfer auf MNL mit drei Alternativen (MIV, ÖV, AV)
- Modalsplit berechnen
- Direkte Elastizitäten (Preis & Reisezeit) ermitteln
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from datetime import datetime

# ────────────────────────────────────────────────
# 1. Pfade und Verzeichnisse
# ────────────────────────────────────────────────
base_path = r'c:/Users/rapha/Documents/Datascience/Bachelorarbeit_git/Bachelorarbeit/projekt'
data_path = os.path.join(base_path, 'output', 'model_ready_long_final.csv')
results_dir = os.path.join(base_path, 'results')

os.makedirs(results_dir, exist_ok=True)
print(f"Ergebnisse werden gespeichert in: {results_dir}\n")

# ────────────────────────────────────────────────
# 2. Daten laden
# ────────────────────────────────────────────────
df = pd.read_csv(data_path, sep=';')
print(f"Daten geladen: {len(df):,} Zeilen  |  {df['WEGID'].nunique():,} eindeutige Wege\n")

# ────────────────────────────────────────────────
# 3. Binäres Logit-Modell: MIV vs. ÖV
# ────────────────────────────────────────────────
print("Schätzung binäres Logit-Modell (MIV = 1, ÖV = 0) ...")

data_binary = []
for wegid, group in df.groupby('WEGID'):
    miv = group[group['mode'] == 'miv']
    oev = group[group['mode'] == 'oev']

    if miv.empty or oev.empty:
        continue

    miv_row = miv.iloc[0]
    oev_row = oev.iloc[0]

    if miv_row['choice'] == 0 and oev_row['choice'] == 0:
        continue

    y = 1 if miv_row['choice'] == 1 else 0
    income = miv_row['income']

    ttt_miv = miv_row['access_time'] + miv_row['duration_min']
    ttt_oev = oev_row['access_time'] + oev_row['duration_min']
    delta_ttt = ttt_miv - ttt_oev

    delta_cost = miv_row['cost'] - oev_row['cost']

    data_binary.append({
        'y': y,
        'income': income,
        'delta_ttt': delta_ttt,
        'delta_cost': delta_cost
    })

df_binary = pd.DataFrame(data_binary)
print(f"→ {len(df_binary):,} gültige Beobachtungen für binäres Modell\n")

# Modell schätzen
X_bin = df_binary[['income', 'delta_ttt', 'delta_cost']]
X_bin = sm.add_constant(X_bin)
y_bin = df_binary['y']

model_bin = sm.Logit(y_bin, X_bin).fit(disp=0)
print(model_bin.summary())
print()

# ────────────────────────────────────────────────
# 4. Parameter extrahieren & VTTS berechnen
# ────────────────────────────────────────────────
beta_ttt = model_bin.params['delta_ttt']
beta_cost = model_bin.params['delta_cost']

# VTTS = Wert der Reisezeitersparnis pro Stunde (annualisiert: 360 Tage/Jahr)
vtts_chf_per_hour = abs(beta_ttt / beta_cost) * 60 / 360

def chf_format(value, decimals=2):
    return f"{value:,.{decimals}f}".replace(',', 'X').replace('.', ',').replace('X', '.')

print(f"Wert der Reisezeitersparnis (VTTS): {chf_format(vtts_chf_per_hour)} CHF/Stunde\n")

# ────────────────────────────────────────────────
# 5. MNL-Prognose mit drei Alternativen (MIV, ÖV, AV)
# ────────────────────────────────────────────────
print("MNL-Prognose mit AV (Parameter-Transfer) ...")

data_mnl = []
for wegid, group in df.groupby('WEGID'):
    miv = group[group['mode'] == 'miv'].iloc[0] if not group[group['mode'] == 'miv'].empty else None
    oev = group[group['mode'] == 'oev'].iloc[0] if not group[group['mode'] == 'oev'].empty else None
    av  = group[group['mode'] == 'av' ].iloc[0] if not group[group['mode'] == 'av' ].empty else None

    if miv is None or oev is None or av is None:
        continue

    income = miv['income']

    ttt_miv = miv['access_time'] + miv['duration_min']
    ttt_oev = oev['access_time'] + oev['duration_min']
    ttt_av  = av ['access_time'] + av ['duration_min']

    cost_miv = miv['cost']
    cost_oev = oev['cost']
    cost_av  = av ['cost']

    # Nutzen
    U_oev = beta_ttt * ttt_oev + beta_cost * cost_oev
    U_miv = model_bin.params['const'] + model_bin.params['income'] * income + beta_ttt * ttt_miv + beta_cost * cost_miv
    U_av  = beta_ttt * ttt_av  + beta_cost * cost_av

    # Softmax
    exp_U = np.exp([U_oev, U_miv, U_av])
    sum_exp = exp_U.sum()

    P_oev = exp_U[0] / sum_exp
    P_miv = exp_U[1] / sum_exp
    P_av  = exp_U[2] / sum_exp

    data_mnl.append({
        'WEGID': wegid,
        'P_MIV': P_miv,
        'P_OEV': P_oev,
        'P_AV':  P_av,
        'TTT_MIV': ttt_miv,
        'TTT_OEV': ttt_oev,
        'TTT_AV':  ttt_av,
        'cost_MIV': cost_miv,
        'cost_OEV': cost_oev,
        'cost_AV':  cost_av
    })

df_prognose = pd.DataFrame(data_mnl)
print(f"→ Prognose für {len(df_prognose):,} Wege möglich\n")

# Modalsplit
modalsplit = {
    'MIV': df_prognose['P_MIV'].mean(),
    'ÖV':  df_prognose['P_OEV'].mean(),
    'AV':  df_prognose['P_AV'].mean()
}

print("Modalsplit (Mittelwerte):")
print(f"  MIV: {modalsplit['MIV']:6.1%}")
print(f"   ÖV: {modalsplit['ÖV']:6.1%}")
print(f"   AV: {modalsplit['AV']:6.1%}\n")

# ────────────────────────────────────────────────
# 6. Direkte Elastizitäten
# ────────────────────────────────────────────────
print("Berechnung direkter Elastizitäten ...")

# Gewichtete Mittelwerte der Attribute
w_P = df_prognose['P_MIV'] + df_prognose['P_OEV'] + df_prognose['P_AV']

mean_TTT_MIV = (df_prognose['P_MIV'] * df_prognose['TTT_MIV']).sum() / w_P.sum()
mean_TTT_OEV = (df_prognose['P_OEV'] * df_prognose['TTT_OEV']).sum() / w_P.sum()
mean_TTT_AV  = (df_prognose['P_AV']  * df_prognose['TTT_AV' ]).sum() / w_P.sum()

mean_cost_MIV = (df_prognose['P_MIV'] * df_prognose['cost_MIV']).sum() / w_P.sum()
mean_cost_OEV = (df_prognose['P_OEV'] * df_prognose['cost_OEV']).sum() / w_P.sum()
mean_cost_AV  = (df_prognose['P_AV']  * df_prognose['cost_AV' ]).sum() / w_P.sum()

# Direkte Elastizitäten (MNL-Formel)
e_price_MIV = beta_cost * mean_cost_MIV * (1 - modalsplit['MIV'])
e_time_MIV  = beta_ttt  * mean_TTT_MIV  * (1 - modalsplit['MIV'])

e_price_OEV = beta_cost * mean_cost_OEV * (1 - modalsplit['ÖV'])
e_time_OEV  = beta_ttt  * mean_TTT_OEV  * (1 - modalsplit['ÖV'])

e_price_AV  = beta_cost * mean_cost_AV  * (1 - modalsplit['AV'])
e_time_AV   = beta_ttt  * mean_TTT_AV   * (1 - modalsplit['AV'])

print("Direkte Elastizitäten:")
print(f"  MIV   Preis: {e_price_MIV:>6.3f}    Zeit: {e_time_MIV:>6.3f}")
print(f"  ÖV    Preis: {e_price_OEV:>6.3f}    Zeit: {e_time_OEV:>6.3f}")
print(f"  AV    Preis: {e_price_AV:>6.3f}    Zeit: {e_time_AV:>6.3f}")
print()
print(f"Für AV ist die Reisezeit ca. {abs(e_time_AV / e_price_AV):.1f} mal wichtiger als der Preis.\n")

# ────────────────────────────────────────────────
# 7. Ergebnisse speichern
# ────────────────────────────────────────────────
# Modalsplit
summary_df = pd.DataFrame({
    'Verkehrsmittel': ['MIV', 'ÖV', 'AV'],
    'Marktanteil': [f"{v:.1%}" for v in modalsplit.values()],
    'VTTS_CHF_h': [chf_format(vtts_chf_per_hour), '', '']
})
summary_df.to_csv(os.path.join(results_dir, 'modalsplit_summary.csv'), index=False, sep=';')

# Detaillierte Prognose
df_prognose.to_csv(os.path.join(results_dir, 'prognose_detailed.csv'), index=False, sep=';')

# Modellparameter
params_df = pd.DataFrame({
    'Parameter': ['const (MIV)', 'income', 'delta_ttt', 'delta_cost', 'VTTS (CHF/h)'],
    'Wert': [model_bin.params['const'], model_bin.params['income'], beta_ttt, beta_cost, chf_format(vtts_chf_per_hour)],
    'Einheit': ['', 'pro CHF', 'pro Minute', 'pro CHF', 'CHF/Stunde']
})
params_df.to_csv(os.path.join(results_dir, 'model_parameters.csv'), index=False, sep=';')

# Elastizitäten
elasticity_df = pd.DataFrame({
    'Verkehrsmittel': ['MIV', 'MIV', 'ÖV', 'ÖV', 'AV', 'AV'],
    'Attribut':       ['Preis', 'Zeit', 'Preis', 'Zeit', 'Preis', 'Zeit'],
    'Elastizität':    [e_price_MIV, e_time_MIV, e_price_OEV, e_time_OEV, e_price_AV, e_time_AV],
    'Marktanteil':    [modalsplit['MIV']] * 2 + [modalsplit['ÖV']] * 2 + [modalsplit['AV']] * 2
})
elasticity_df.to_csv(os.path.join(results_dir, 'elastizitaeten_direkt.csv'), index=False, sep=';')

# Laufzeit-Log
log_text = f"""PIPELINE AUSGEFÜHRT: {datetime.now().strftime('%d.%m.%Y %H:%M')}
- Binäres Modell:       {len(df_binary):,} Beobachtungen
- Wert der Reisezeitersparnis (VTTS): {chf_format(vtts_chf_per_hour)} CHF/Stunde
- Modalsplit:           MIV {modalsplit['MIV']:5.1%} | ÖV {modalsplit['ÖV']:5.1%} | AV {modalsplit['AV']:5.1%}
- Elastizitäten (AV):   Preis {e_price_AV:.3f}   |   Zeit {e_time_AV:.3f}
"""
with open(os.path.join(results_dir, 'pipeline_log.txt'), 'w', encoding='utf-8') as f:
    f.write(log_text)

print(f"Analyse abgeschlossen – Ergebnisse gespeichert in:\n  {results_dir}")