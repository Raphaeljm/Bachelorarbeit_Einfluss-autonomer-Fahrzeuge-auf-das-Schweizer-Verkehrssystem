# -*- coding: utf-8 -*-
"""
02_data_preparation.py

Ziel
----
Erstellung eines einheitlichen, modellfähigen Datensatzes im Long-Format mit
drei Verkehrsmittel-Alternativen pro Weg (ÖV, MIV, AV).

Schritte:
- Laden der Rohdaten (MiD/MiD-wege, Haushaltsdaten) und API-Ergebnisse
- Berechnung von Reisezeiten, Zugangszeiten und Kosten für alle drei Modi
- Zusammenführung zu einem Wide-Datensatz
- Umwandlung in Long-Format (eine Zeile pro Alternative und Weg)

Ausgabe: model_ready_long.csv (für diskrete Wahlmodelle, z. B. Logit)
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd


# ────────────────────────────────────────────────
#  Konfiguration – alle wichtigen Parameter zentral
# ────────────────────────────────────────────────
CONFIG = {
    "data_dir": "c:/Users/rapha/Documents/Datascience/Bachelorarbeit_git/Bachelorarbeit/projekt/data",
    "output_dir": "c:/Users/rapha/Documents/Datascience/Bachelorarbeit_git/Bachelorarbeit/projekt/output",
    "file_wege": "wege.csv",
    "file_haushalte": "haushalte.csv",
    "file_miv_api": "distance_duration_results.csv",
    "file_oev_parts": [
        "miv_wege_sample_part1_oev.csv",
        "miv_wege_sample_part2_oev.csv",
        "miv_wege_sample_part3_oev.csv",
        "miv_wege_sample_part4_oev.csv",
        "miv_wege_sample_part5_oev.csv",
    ],
    "sep": ";",
    "encoding": "iso8859-1",
    "miv_cost_per_km": 0.76,         # CHF/km (inkl. Betrieb + Abschreibung)
    "oev_cost_per_km": 0.50,         # CHF/km (angenommener Durchschnitt)
    "av_cost_per_km": 0.52,          # CHF/km (Annahme für autonomes Fahren)
    "annualization": 360,            # Tage pro Jahr für Jahreskosten
    "ga_cost_cap": 3995.0,           # Maximalpreis GA (Deckelung ÖV-Kosten)
    "final_model_file": "model_ready_long.csv",
}


# ────────────────────────────────────────────────
#  Logging – einheitliche Protokollierung
# ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("data_preparation")


# ────────────────────────────────────────────────
#  Hilfsfunktionen – Datei-IO und Schlüsselgenerierung
# ────────────────────────────────────────────────

def read_csv(path: Path, sep: str = ";", encoding: str = "utf-8") -> pd.DataFrame:
    """Liest eine CSV-Datei und protokolliert den Vorgang."""
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")
    
    logger.info(f"Lese Datei: {path}")
    df = pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False)
    logger.info(f"  → {len(df):,} Zeilen | Spalten: {list(df.columns)}")
    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Speichert DataFrame als CSV und protokolliert."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=";", encoding="utf-8")
    logger.info(f"Geschrieben: {path}  ({len(df):,} Zeilen)")


def enrich_with_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Erzeugt WEGID (HHNR_WEGNR) falls nicht vorhanden."""
    if 'WEGID' not in df.columns and 'HHNR' in df.columns and 'WEGNR' in df.columns:
        df['WEGID'] = df['HHNR'].astype(str) + '_' + df['WEGNR'].astype(str)
    return df


# ────────────────────────────────────────────────
#  Berechnungsfunktionen – Zeit und Kosten
# ────────────────────────────────────────────────

# Einkommensstufen → jährliches Haushaltseinkommen (Mittelwert der Klasse)
INCOME_MAP = {
    1: 12000, 2: 36000, 3: 60000, 4: 84000,
    5: 108000, 6: 132000, 7: 156000, 8: 180000, 9: 204000
}


def map_income(stage: pd.Series, default: int = 84500) -> pd.Series:
    """Wandelt Einkommensstufe in jährliches Haushaltseinkommen um."""
    return stage.map(INCOME_MAP).fillna(default).astype(int)


def compute_duration_minutes(seconds: pd.Series) -> pd.Series:
    """Wandelt Sekunden in Minuten um."""
    return seconds.astype(float) / 60.0


def compute_access_time_oev(ivt_min: pd.Series, total_min: pd.Series) -> pd.Series:
    """Berechnet Zugangs- und Umsteigezeit (Gesamt − reine Fahrzeit)."""
    return (total_min - ivt_min).clip(lower=0)


def compute_access_time_oev_from_api(gehzeit_s: pd.Series, umsteigezeit_s: pd.Series, minute_offset: pd.Series) -> pd.Series:
    """Zugangszeit ÖV aus API-Daten (Gehzeit + Umsteigezeit + Puffer)."""
    total_seconds = gehzeit_s.astype(float) + umsteigezeit_s.astype(float) + minute_offset.astype(float) * 60
    return total_seconds / 60.0


def compute_access_time_miv(start_raumtyp: pd.Series, ziel_raumtyp: pd.Series) -> pd.Series:
    """Zugangszeit MIV (Parken/Suchen) abhängig von Raumtyp Start + Ziel."""
    raumtyp_map = {1: 4.8, 2: 3.0, 3: 2.0}  # Minuten
    start = start_raumtyp.map(raumtyp_map).fillna(3.0)
    ziel  = ziel_raumtyp.map(raumtyp_map).fillna(3.0)
    return start + ziel


def compute_access_time_av(start_raumtyp: pd.Series) -> pd.Series:
    """Zugangszeit AV (Abholzeit, autonomes Fahrzeug)."""
    av_map = {1: 4.0, 2: 7.0, 3: 12.0}  # Minuten
    return start_raumtyp.map(av_map).fillna(7.0)


def compute_costs_miv(distance_km: pd.Series, cost_per_km: float, annualization: int) -> pd.Series:
    """Jahreskosten MIV (Distanz × Kosten pro km × Tage)."""
    return distance_km * cost_per_km * annualization


def compute_costs_oev(distance_km: pd.Series, cost_per_km: float, annualization: int, ga_cap: float) -> pd.Series:
    """Jahreskosten ÖV mit GA-Deckelung."""
    raw = distance_km * cost_per_km * annualization
    return raw.clip(upper=ga_cap)


def compute_costs_av(distance_km: pd.Series, cost_per_km: float, annualization: int) -> pd.Series:
    """Jahreskosten AV (Distanz × Kosten pro km × Tage)."""
    return distance_km * cost_per_km * annualization


# ────────────────────────────────────────────────
#  Daten laden
# ────────────────────────────────────────────────

def load_sources(args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lädt alle benötigten Quelldateien."""
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    wege_df = read_csv(data_dir / args.file_wege, sep=args.sep, encoding=args.encoding)
    wege_df = enrich_with_keys(wege_df)

    miv_api_df = read_csv(output_dir / args.file_miv_api, sep=",", encoding="utf-8")

    oev_parts = []
    for fname in args.file_oev_parts:
        path = output_dir / fname
        if path.exists():
            oev_parts.append(read_csv(path, sep=",", encoding="utf-8"))

    if not oev_parts:
        raise FileNotFoundError("Keine ÖV-API-Teildateien gefunden.")

    oev_api_df = pd.concat(oev_parts, ignore_index=True)

    hh_df = read_csv(data_dir / args.file_haushalte, sep=args.sep, encoding=args.encoding)

    return wege_df, miv_api_df, oev_api_df, hh_df


# ────────────────────────────────────────────────
#  Wide-Format mit allen drei Alternativen erstellen
# ────────────────────────────────────────────────

def build_wide_with_av(wege_df: pd.DataFrame, miv_api_df: pd.DataFrame,
                      oev_api_df: pd.DataFrame, hh_df: pd.DataFrame,
                      args) -> pd.DataFrame:
    """
    Erzeugt einen Wide-Datensatz mit Attributen für ÖV, MIV und AV
    pro gewähltem Weg (entweder ÖV oder MIV gewählt).
    """
    hh_df = hh_df[['HHNR', 'f20601']].rename(columns={'f20601': 'income_stage'})

    # ── Wege mit gewähltem ÖV (MIV als Alternative) ──────────────────────────
    oev_wege = wege_df[wege_df['wmittel1a'] == 3].copy()
    oev_wide = miv_api_df[['WEGID', 'distance_m', 'duration_s']].merge(
        oev_wege[['WEGID', 'HHNR', 'dauer1', 'dauer2', 'w_rdist',
                  'S_stadt_land_2012', 'Z_stadt_land_2012']],
        on='WEGID', how='inner'
    )
    oev_wide = oev_wide.merge(hh_df, on='HHNR', how='left')
    oev_wide['income'] = map_income(oev_wide['income_stage'])

    oev_wide['duration_oev_min'] = oev_wide['dauer1'].astype(float)
    oev_wide['access_time_oev'] = compute_access_time_oev(oev_wide['duration_oev_min'], oev_wide['dauer2'])
    oev_wide['duration_miv_min'] = compute_duration_minutes(oev_wide['duration_s'])
    oev_wide['distance_km'] = oev_wide['distance_m'] / 1000.0
    oev_wide['access_time_miv'] = compute_access_time_miv(oev_wide['S_stadt_land_2012'], oev_wide['Z_stadt_land_2012'])
    oev_wide['cost_oev'] = compute_costs_oev(oev_wide['w_rdist'], args.oev_cost_per_km, args.annualization, args.ga_cost_cap)
    oev_wide['cost_miv'] = compute_costs_miv(oev_wide['distance_km'], args.miv_cost_per_km, args.annualization)

    # ── Wege mit gewähltem MIV (ÖV als Alternative) ──────────────────────────
    miv_wege = wege_df[wege_df['wmittel1a'] == 2].copy()
    miv_wide = oev_api_df[['WEGID', 'oev_dauer_s', 'oev_distanz_km',
                           'gehzeit_s', 'umsteigezeit_s', 'minute_offset']].merge(
        miv_wege[['WEGID', 'HHNR', 'dauer1', 'dauer2', 'w_rdist',
                  'S_stadt_land_2012', 'Z_stadt_land_2012']],
        on='WEGID', how='inner'
    )
    miv_wide = miv_wide.merge(hh_df, on='HHNR', how='left')
    miv_wide['income'] = map_income(miv_wide['income_stage'])

    miv_wide['duration_miv_min'] = miv_wide['dauer2'].astype(float)
    miv_wide['access_time_miv'] = compute_access_time_miv(miv_wide['S_stadt_land_2012'], miv_wide['Z_stadt_land_2012'])
    miv_wide['duration_oev_min'] = compute_duration_minutes(miv_wide['oev_dauer_s'])
    miv_wide['access_time_oev'] = compute_access_time_oev_from_api(
        miv_wide['gehzeit_s'], miv_wide['umsteigezeit_s'], miv_wide['minute_offset']
    )
    miv_wide['cost_miv'] = compute_costs_miv(miv_wide['w_rdist'], args.miv_cost_per_km, args.annualization)
    miv_wide['cost_oev'] = compute_costs_oev(miv_wide['oev_distanz_km'], args.oev_cost_per_km, args.annualization, args.ga_cost_cap)

    # ── Zusammenführen und AV-Attribute ergänzen ─────────────────────────────
    wide_df = pd.concat([oev_wide, miv_wide], ignore_index=True)
    wide_df = enrich_with_keys(wide_df)

    wide_df['access_time_av'] = compute_access_time_av(wide_df['S_stadt_land_2012'])
    wide_df['duration_av_min'] = wide_df['duration_miv_min']          # Annahme: AV-Fahrzeit = MIV-Fahrzeit
    wide_df['cost_av'] = compute_costs_av(wide_df['w_rdist'], args.av_cost_per_km, args.annualization)

    return wide_df


# ────────────────────────────────────────────────
#  Umwandlung in Long-Format (eine Zeile pro Alternative)
# ────────────────────────────────────────────────

def wide_to_long_with_av(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    Wandelt Wide-Format in Long-Format um.
    Ergebnis: pro WEGID genau drei Zeilen (oev, miv, av)
    mit den Attributen access_time, duration_min, cost und choice.
    """
    mapping = {
        "oev": {"access_time": "access_time_oev", "duration_min": "duration_oev_min", "cost": "cost_oev"},
        "miv": {"access_time": "access_time_miv", "duration_min": "duration_miv_min", "cost": "cost_miv"},
        "av":  {"access_time": "access_time_av",  "duration_min": "duration_av_min",  "cost": "cost_av"},
    }

    base_cols = ['WEGID', 'HHNR', 'income']

    parts = []
    for mode in ["oev", "miv", "av"]:
        cols = mapping[mode]
        part = wide_df[base_cols + list(cols.values())].copy()
        part = part.rename(columns={
            cols["access_time"]: "access_time",
            cols["duration_min"]: "duration_min",
            cols["cost"]: "cost",
        })
        part["mode"] = mode
        parts.append(part)

    long_df = pd.concat(parts, ignore_index=True)

    # Choice-Variable hinzufügen (1 = gewählter Modus)
    wide_df["modus"] = wide_df["modus"].astype(str).str.strip().str.lower()
    long_df = long_df.merge(
        wide_df[['WEGID', 'modus']],
        on='WEGID',
        how='left'
    )
    long_df["choice"] = (long_df["mode"] == long_df["modus"]).astype(int)
    long_df = long_df.drop(columns=["modus"])

    # Feste Reihenfolge pro WEGID: oev → miv → av
    mode_order = pd.CategoricalDtype(categories=["oev", "miv", "av"], ordered=True)
    long_df["mode"] = long_df["mode"].astype(mode_order)
    long_df = long_df.sort_values(["WEGID", "mode"]).reset_index(drop=True)

    # Validierung
    n_wege = long_df["WEGID"].nunique()
    if len(long_df) != n_wege * 3:
        raise ValueError(f"Erwartet {n_wege * 3:,} Zeilen, gefunden: {len(long_df):,}")
    if long_df.groupby("WEGID")["choice"].sum().ne(1).any():
        raise ValueError("Nicht genau eine gewählte Alternative pro WEGID")

    logger.info(f"Long-Format erstellt: {len(long_df):,} Zeilen ({n_wege:,} Wege × 3 Alternativen)")
    return long_df


# ────────────────────────────────────────────────
#  Argumente und Hauptprogramm
# ────────────────────────────────────────────────

def parse_args():
    """Parst Kommandozeilenargumente (mit sinnvollen Defaults)."""
    parser = argparse.ArgumentParser(description="Erstellung modellfähiger Daten im Long-Format (ÖV, MIV, AV)")
    for key, value in CONFIG.items():
        arg_name = f"--{key}"
        if key == "file_oev_parts":
            parser.add_argument(arg_name, nargs="*", default=value)
        elif isinstance(value, (int, float)):
            parser.add_argument(arg_name, type=type(value), default=value)
        else:
            parser.add_argument(arg_name, default=value)
    return parser.parse_args()


def main():
    """Hauptfunktion – steuert den gesamten Ablauf."""
    args = parse_args()
    logger.info("Starte Datenvorbereitung – Erstellung Long-Format mit ÖV, MIV, AV")

    wege_df, miv_api_df, oev_api_df, hh_df = load_sources(args)
    wide_df = build_wide_with_av(wege_df, miv_api_df, oev_api_df, hh_df, args)
    long_df = wide_to_long_with_av(wide_df)

    output_path = Path(args.output_dir) / args.final_model_file
    write_csv(long_df, output_path)

    logger.info("Fertig – modellfähiger Datensatz erstellt")


if __name__ == "__main__":
    main()