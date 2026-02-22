"""
03_data_finalisation.py

Ziel
----

Komplet bereinigts und fehlerfreies Dataframe für das Modell.

- Bereinigt alle na_values
- überprüft die Ordnung und weitere Unreinheiten der Daten
"""

import pandas as pd
import numpy as np

# -------------------------------------------------
# 1. CSV einlesen (robust)
# -------------------------------------------------
df = pd.read_csv(
    "c:/Users/rapha/Documents/Datascience/Bachelorarbeit_git/Bachelorarbeit/projekt/output/model_ready_long.csv",
    sep=";",
    na_values=["", " ", "nan", "NaN", "None", "-", "--"]
)

print(f"Daten geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten\n")
print("Erste 5 Zeilen:")
print(df.head(), "\n")

# -------------------------------------------------
# 2. Prüfung auf fehlende / fehlerhafte Werte
# -------------------------------------------------
print("=== Prüfung auf fehlende/fehlerhafte Werte ===")

# NaN-Werte
nan_counts = df.isna().sum()
if nan_counts.sum() > 0:
    print("NaN-Werte gefunden:")
    print(nan_counts[nan_counts > 0])
else:
    print("Keine NaN-Werte gefunden.")

# leere Strings (nach Strip, nur zur Sicherheit)
empty_strings = df.apply(lambda x: x.astype(str).str.strip().eq("")).sum()
if empty_strings.sum() > 0:
    print("\nLeere Strings gefunden:")
    print(empty_strings[empty_strings > 0])
else:
    print("Keine leeren Strings gefunden.")

# unendliche Werte
inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
if inf_counts.sum() > 0:
    print("\nUnendliche Werte (inf/-inf) gefunden:")
    print(inf_counts[inf_counts > 0])
else:
    print("Keine unendlichen Werte gefunden.")

print("\n")

# -------------------------------------------------
# 3. Logische Prüfungen
# -------------------------------------------------
print("=== Logische Prüfungen ===")

# choice nur 0/1
if 'choice' in df.columns:
    invalid_choice = df[~df['choice'].isin([0, 1])]
    if len(invalid_choice) > 0:
        print(f"Warnung: {len(invalid_choice)} Zeilen mit ungültigem 'choice'")
    else:
        print("'choice' enthält nur 0 und 1.")

# negative Werte in Zeit/Kosten
for col in ['access_time', 'duration_min', 'cost']:
    if col in df.columns:
        neg = (pd.to_numeric(df[col], errors='coerce') < 0).sum()
        if neg > 0:
            print(f"Warnung: {neg} negative Werte in '{col}'")

# choice=1 aber duration_min=0
if {'duration_min', 'choice'}.issubset(df.columns):
    zero_dur_chosen = df[(df['duration_min'] == 0) & (df['choice'] == 1)]
    if len(zero_dur_chosen) > 0:
        print(f"Warnung: {len(zero_dur_chosen)} Fälle: choice=1, duration_min=0")

# access_time > duration_min
if {'access_time', 'duration_min'}.issubset(df.columns):
    df_num = df[['access_time', 'duration_min']].apply(pd.to_numeric, errors='coerce')
    bad_time = (df_num['access_time'] > df_num['duration_min']) & df_num['duration_min'].notna()
    if bad_time.sum() > 0:
        print(f"Warnung: {bad_time.sum()} Fälle: access_time > duration_min")

print("\n")

# -------------------------------------------------
# 4. Entfernen von WEG-Gruppen mit fehlenden oder leeren Werten
# -------------------------------------------------
print("=== Entfernen von WEG-Gruppen mit fehlenden oder leeren Werten ===")

# Einheitliches Format für WEGID
df['WEGID'] = df['WEGID'].astype(str).str.strip()

# Sicherstellen, dass alle "leeren" Felder echte NaN sind
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

# Mask für Zeilen mit NaN
missing_mask = df.isna().any(axis=1)

if missing_mask.any():
    bad_wegids = df.loc[missing_mask, 'WEGID'].unique()
    print(f"{len(bad_wegids)} WEG-Gruppen enthalten fehlende Werte "
          f"(insgesamt {missing_mask.sum()} betroffene Zeilen).")
    print(f"Diese WEGIDs werden komplett entfernt:\n{bad_wegids}\n")

    before = df.shape[0]
    df_clean = df[~df['WEGID'].isin(bad_wegids)].copy()
    after = df_clean.shape[0]

    print(f"Entfernte Zeilen: {before - after} (von {before} → {after})")
else:
    print("Keine fehlenden Werte → nichts zu entfernen.")
    df_clean = df.copy()

print("\n")

# -------------------------------------------------
# 5. Nachprüfung des bereinigten Datensatzes
# -------------------------------------------------
print("=== Kurze Nachprüfung des bereinigten Datensatzes ===")
if df_clean.isna().any().any() or df_clean.apply(lambda x: x.astype(str).str.strip().eq("")).any().any():
    print("Warnung: Noch immer NaN oder leere Strings vorhanden!")
else:
    print("Bereinigter Datensatz ist frei von NaN und leeren Strings.")

# -------------------------------------------------
# 5b. Prüfung auf Choice-Konsistenz und Duplikate
# -------------------------------------------------
print("=== Prüfung auf Choice-Konsistenz und Duplikate ===")

# 1. Jede WEGID muss genau eine gewählte Alternative haben
choice_check = df_clean.groupby('WEGID')['choice'].sum().reset_index(name='n_chosen')
invalid_choice_sets = choice_check[choice_check['n_chosen'] != 1]
if len(invalid_choice_sets) > 0:
    print(f"→ {len(invalid_choice_sets)} WEG-Gruppen mit ungültiger Choice-Verteilung (≠ 1 gewählt).")
    df_clean = df_clean[df_clean['WEGID'].isin(choice_check.loc[choice_check['n_chosen'] == 1, 'WEGID'])]
else:
    print("Alle WEGIDs haben genau eine gewählte Alternative.")

# 2. Jede WEGID muss genau drei Alternativen haben (oev, miv, av)
mode_check = df_clean.groupby('WEGID')['mode'].nunique().reset_index(name='n_modes')
incomplete = mode_check[mode_check['n_modes'] != 3]
if len(incomplete) > 0:
    print(f"→ {len(incomplete)} WEG-Gruppen mit unvollständigem Choice-Set entfernt.")
    df_clean = df_clean[df_clean['WEGID'].isin(mode_check.loc[mode_check['n_modes'] == 3, 'WEGID'])]
else:
    print("Alle WEGIDs enthalten genau drei Alternativen (oev, miv, av).")

# 3. Doppelte Modi pro WEGID entfernen
dup_check = df_clean.duplicated(subset=['WEGID', 'mode'], keep=False)
if dup_check.any():
    print(f"→ {dup_check.sum()} doppelte Alternativen gefunden → nur erste behalten.")
    df_clean = (
        df_clean.sort_values(by=['WEGID', 'mode', 'duration_min'])
                .drop_duplicates(subset=['WEGID', 'mode'], keep='first')
    )
else:
    print("Keine doppelten Alternativen pro WEGID gefunden.")

# -------------------------------------------------
# 6. Speichern des final bereinigten Datensatzes
# -------------------------------------------------
output_path = "c:/Users/rapha/Documents/Datascience/Bachelorarbeit_git/Bachelorarbeit/projekt/output/model_ready_long_final.csv"
df_clean.to_csv(output_path, sep=";", index=False)
print(f"\nFinal bereinigter Datensatz gespeichert unter: {output_path}")
