# -*- coding: utf-8 -*-
"""
01_api_fetch.py

Ziel
----
Vervollständigung der Wegedaten mit Reisezeiten und Distanzen der jeweils
nicht gewählten Verkehrsmittel-Alternative.

- Für Wege mit gewähltem ÖV → MIV-Daten via Google Distance Matrix API
- Für Wege mit gewähltem MIV → ÖV-Daten via Google Directions API (Transit)

Funktionen:
- Batch-Verarbeitung mit Fehlerbehandlung und Retry-Mechanismus
- Zwischenspeicherung der Ergebnisse in mehreren CSV-Dateien
- Logging und Überspringen bereits vorhandener Ergebnisdateien

Ausgaben:
- MIV-Ergebnisse:   output/distance_duration_results.csv
- ÖV-Ergebnisse:    output/miv_wege_sample_partX_oev.csv  (X = 1..5)
"""

import pandas as pd
import requests
import os
import time
import logging
from datetime import datetime
from urllib.parse import urlencode
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ────────────────────────────────────────────────
#  Logging einrichten
# ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("api_fetch")


# ────────────────────────────────────────────────
#  Konfiguration – zentrale Parameter
# ────────────────────────────────────────────────
CONFIG = {
    'BATCH_SIZE_MIV': 5,               # kleine Batches wegen URL-Längenlimit
    'BATCH_SIZE_OEV': 1000,            # nur für Datei-Logik relevant
    'SLEEP_BETWEEN_CALLS': 1.2,        # Sekunden zwischen API-Aufrufen
    'SLEEP_ON_SSL_ERROR': 10,          # längere Pause bei SSL/TLS-Fehlern
    'RETRY_COUNT': 3,                  # Anzahl Wiederholungen bei Server-Fehlern
    'DISTANCE_MATRIX_API_KEY_FILE': 'config/distance_matrix_api_key.txt',
    'DIRECTIONS_API_KEY_FILE': 'config/directions_api_key.txt',
    'PART_COUNT': 5,                   # Anzahl der MIV-Teildateien
    'OUTPUT_DIR': 'output'
}


# ────────────────────────────────────────────────
#  Hilfsfunktionen
# ────────────────────────────────────────────────

def load_api_key(key_file: str) -> str:
    """Lädt Google API-Schlüssel aus Textdatei."""
    path = key_file
    if not os.path.exists(path):
        raise FileNotFoundError(f"API-Schlüssel-Datei nicht gefunden: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def create_session() -> requests.Session:
    """Erstellt eine requests-Session mit Retry-Verhalten."""
    session = requests.Session()
    retries = Retry(
        total=CONFIG['RETRY_COUNT'],
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session


# ────────────────────────────────────────────────
#  1. MIV-Daten für ÖV-Wege abfragen (Distance Matrix API)
# ────────────────────────────────────────────────

def fetch_miv_for_oev_weges(input_path: str, output_path: str) -> None:
    """
    Fragt für alle Wege mit gewähltem ÖV die MIV-Fahrzeit und Distanz ab.
    Speichert Ergebnisse nur, wenn die Ausgabedatei noch nicht existiert.
    """
    if os.path.exists(output_path):
        logger.info(f"MIV-Daten bereits vorhanden → überspringe Abfrage: {output_path}")
        return

    api_key = load_api_key(CONFIG['DISTANCE_MATRIX_API_KEY_FILE'])
    session = create_session()

    # Eingabedaten laden & filtern
    df = pd.read_csv(input_path, sep=';', encoding='iso8859-1')
    df['WEGID'] = df['HHNR'].astype(str) + '_' + df['WEGNR'].astype(str)
    df = df[(df['wmittel1a'] == 3) &                  # ÖV gewählt
            (df['S_X'] != df['Z_X']) & (df['S_Y'] != df['Z_Y'])]  # keine identischen Koordinaten

    logger.info(f"Abfrage MIV für {len(df):,} ÖV-Wege")

    results = []
    for i in range(0, len(df), CONFIG['BATCH_SIZE_MIV']):
        batch = df.iloc[i:i + CONFIG['BATCH_SIZE_MIV']]

        origins = "|".join(f"{row['S_Y']},{row['S_X']}" for _, row in batch.iterrows())
        destinations = "|".join(f"{row['Z_Y']},{row['Z_X']}" for _, row in batch.iterrows())

        params = {
            'origins': origins,
            'destinations': destinations,
            'mode': 'driving',
            'key': api_key
        }
        url = "https://maps.googleapis.com/maps/api/distancematrix/json?" + urlencode(params)

        try:
            resp = session.get(url, timeout=15)
            data = resp.json()

            if data.get('status') != 'OK':
                logger.warning(f"API-Status nicht OK: {data.get('status')}")
                continue

            for idx, row in enumerate(data['rows']):
                element = row['elements'][idx]
                if element['status'] == 'OK':
                    r = {
                        'WEGID': batch.iloc[idx]['WEGID'],
                        'duration_s': element['duration']['value'],
                        'distance_m': element['distance']['value'],
                        'dauer1': batch.iloc[idx]['dauer1'],
                        'dauer2': batch.iloc[idx]['dauer2'],
                        'duration_oev_min': batch.iloc[idx]['dauer2'],
                        'w_rdist': element['distance']['value'] / 1000.0
                    }
                    results.append(r)

        except Exception as e:
            logger.error(f"Fehler bei MIV-Batch {i//CONFIG['BATCH_SIZE_MIV']+1}: {e}")

        time.sleep(CONFIG['SLEEP_BETWEEN_CALLS'])

    pd.DataFrame(results).to_csv(output_path, index=False)
    logger.info(f"MIV-Daten gespeichert: {output_path}  ({len(results):,} Datensätze)")


# ────────────────────────────────────────────────
#  2. ÖV-Daten für MIV-Wege abfragen (Directions API – Transit)
# ────────────────────────────────────────────────

def parse_transfer_times(steps: list) -> list:
    """Ermittelt Wartezeiten zwischen ÖV-Verbindungen."""
    transfers = []
    last_arrival = None
    for step in steps:
        if step.get("travel_mode") == "TRANSIT":
            dep = int(step["transit_details"]["departure_time"]["value"])
            arr = int(step["transit_details"]["arrival_time"]["value"])
            if last_arrival is not None:
                transfers.append(dep - last_arrival)
            last_arrival = arr
    return transfers


def fetch_oev_for_miv_parts() -> None:
    """
    Fragt für MIV-Wege (aufgeteilt in Teildateien) die ÖV-Verbindung ab.
    Pro Teil-Datei wird eine eigene Ergebnisdatei erzeugt.
    """
    api_key = load_api_key(CONFIG['DIRECTIONS_API_KEY_FILE'])
    session = create_session()

    # Zukunftszeitpunkt für konsistente Abfahrtszeiten
    reference_time = int(datetime.now().timestamp()) + 3600  # +1 Stunde

    for part in range(1, CONFIG['PART_COUNT'] + 1):
        input_file = f"{CONFIG['OUTPUT_DIR']}/miv_wege_sample_part{part}.csv"
        output_file = f"{CONFIG['OUTPUT_DIR']}/miv_wege_sample_part{part}_oev.csv"

        if os.path.exists(output_file):
            logger.info(f"ÖV-Daten bereits vorhanden → überspringe Teil {part}: {output_file}")
            continue

        if not os.path.exists(input_file):
            logger.warning(f"Eingabedatei fehlt: {input_file}")
            continue

        df = pd.read_csv(input_file)
        logger.info(f"Verarbeite Teil {part} – {len(df):,} Wege")

        results = []

        for idx, row in df.iterrows():
            origin = f"{row['S_Y']},{row['S_X']}"
            dest = f"{row['Z_Y']},{row['Z_X']}"
            minute_offset = 0

            result = {
                "WEGID": row["WEGID"],
                "oev_dauer_s": None,
                "oev_distanz_km": None,
                "gehzeit_s": 0,
                "umsteigezeit_s": 0,
                "access_time_oev": 0,
                "umstiege": 0,
                "steps": 0,
                "minute_offset": None,
                "status": "NO_ATTEMPT"
            }

            # Mehrere Versuche mit zeitlichem Versatz bei Nichtfinden
            for attempt in range(3):
                departure_time = reference_time + minute_offset * 60
                params = {
                    "origin": origin,
                    "destination": dest,
                    "mode": "transit",
                    "departure_time": departure_time,
                    "key": api_key
                }
                url = "https://maps.googleapis.com/maps/api/directions/json?" + urlencode(params)

                try:
                    resp = session.get(url, timeout=12)
                    data = resp.json()

                    if data.get("status") == "OK" and data.get("routes"):
                        leg = data["routes"][0]["legs"][0]
                        steps = leg["steps"]

                        result["oev_dauer_s"] = leg["duration"]["value"]
                        result["oev_distanz_km"] = leg["distance"]["value"] / 1000.0
                        result["gehzeit_s"] = sum(
                            s["duration"]["value"] for s in steps if s["travel_mode"] == "WALKING"
                        )
                        transfers = parse_transfer_times(steps)
                        result["umsteigezeit_s"] = sum(transfers)
                        result["umstiege"] = len(transfers)
                        result["steps"] = len(steps)
                        result["minute_offset"] = minute_offset
                        result["status"] = "OK"
                        result["access_time_oev"] = result["gehzeit_s"] + result["umsteigezeit_s"] + minute_offset * 60
                        break

                except Exception as e:
                    logger.error(f"Fehler Teil {part}, Zeile {idx}, Versuch {attempt+1}: {e}")

                minute_offset += 5
                time.sleep(CONFIG['SLEEP_BETWEEN_CALLS'])

            results.append(result)
            time.sleep(CONFIG['SLEEP_BETWEEN_CALLS'])

        pd.DataFrame(results).to_csv(output_file, index=False)
        logger.info(f"Teil {part} abgeschlossen → {output_file}  ({len(results):,} Datensätze)")


# ────────────────────────────────────────────────
#  Hauptprogramm
# ────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starte API-Abfragen für MIV- und ÖV-Alternativen")

    # 1. MIV-Daten für alle ÖV-Wege
    fetch_miv_for_oev_weges(
        input_path="data/wege.csv",
        output_path=f"{CONFIG['OUTPUT_DIR']}/distance_duration_results.csv"
    )

    # 2. ÖV-Daten für alle MIV-Wege (in Teilen)
    fetch_oev_for_miv_parts()

    logger.info("API-Abfragen abgeschlossen")