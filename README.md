# Bachelorarbeit: Einfluss autonomer Fahrzeuge auf das Schweizer Verkehrssystem

Dieses Repository enthält den **Quellcode** und die Dokumentation zur Bachelorarbeit von Raphael Meier an der Hochschule Luzern – Wirtschaft (Bachelor of Science in Economics and Data Science in Mobility).

**Thema**  
Modalsplitberechnung und Analyse der Nachfrageelastizitäten im Kontext autonomer Fahrzeuge (AV) im Schweizer Personenverkehr

Die Arbeit schätzt auf Basis des Mikrozensus Mobilität und Verkehr (MZMV) 2021 ein binäres Logit-Modell (MIV vs. ÖV), überträgt die Parameter auf ein multinomiales Logit-Modell mit AV-Alternative und berechnet prognostizierte Marktanteile, den Wert der Reisezeiteinsparung (VTTS) sowie direkte Preis- und Zeitelastizitäten.

**Schlüsselwörter:** Autonome Fahrzeuge, Modal Split, Diskrete Wahlmodelle, Value of Travel Time Savings, Nachfrageelastizitäten, Schweiz

## Wichtiger Hinweis – Nicht veröffentlichte Ordner

Folgende Verzeichnisse sind absichtlich nicht im Repository enthalten:

- `config/` → enthält Google Maps API-Schlüssel  
- `data/` → enthält die sensiblen Rohdaten des MZMV 2021 (nicht öffentlich zugänglich)  
- `output/` → enthält Zwischenergebnisse und verarbeitete Datensätze  

Das Repository enthält daher nur den reproduzierbaren Code und die Ergebnisstruktur. Um die komplette Pipeline auszuführen, benötigen Sie Zugriff auf die Originaldaten und eigene API-Schlüssel.

## Ordnerstruktur (öffentlicher Teil)
projekt/<br>
├── .venv/<br>  
├── scripts/                    # Alle Python-Skripte der Analyse-Pipeline<br>
│   ├── 01_api_fetch.py<br>
│   ├── 02_data_preparation_pipeline.py<br>
│   ├── 03_data_finalisation.py<br>
│   └── 04_model_pipeline.py<br>
├── results/                    # Beispiel-Ergebnisdateien (Modalsplit, Elastizitäten, Logs)<br>
│   ├── modalsplit_summary.csv<br>
│   ├── prognose_detailed.csv<br>
│   ├── model_parameters.csv<br>
│   ├── elastizitaeten_direkt.csv<br>
│   └── pipeline_log.txt<br>
└── requirements.txt<br>

Nicht enthalten / ignoriert:  
`config/` – `data/` – `output/`

## Voraussetzungen

- Python 3.8–3.11  
- Google Maps API-Schlüssel mit Zugriff auf  
  - Distance Matrix API  
  - Directions API (Transit)  
- Zugriff auf die MZMV 2021 Datensätze `wege.csv` und `haushalte.csv` (nicht öffentlich)

## Installation & Vorbereitung

1. Repository klonen
   ```bash
   git clone https://github.com/<Ihr-Benutzername>/<repo-name>.git
   cd <repo-name>
2. Virtuelle Umgebung erstellen & Abhängigkeiten installieren
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
3. API-Schlüssel bereitstellen
   Erstellen Sie den Ordner config/ und legen Sie dort zwei reine Textdateien an:<br>
   config/<br>
   ├── distance_matrix_api_key.txt<br>
   └── directions_api_key.txt
4. Eingabedaten bereitstellen
   MZMV-Datensätze in (nicht versionierten) Ordner data/ legen:<br>
   data/<br>
   ├── wege.csv<br>
   └── haushalte.csv

## Ausführung der Pipeline

Führen Sie die Skripte der Reihe nach aus:
1. python scripts/01_api_fetch.py
2. python scripts/02_data_preparation_pipeline.py
3. python scripts/03_data_finalisation.py
4. python scripts/04_model_pipeline.py

Ergebnisse → Ordner results/
Hinweis: Die Pfade in den Skripten sind teilweise absolut codiert – ggf. anpassen.

## Ergebnisse

- Prognostizierter Modalsplit (ca.):
  MIV 37 % | ÖV 28 % | AV 35 %
- VTTS: 52.23 CHF pro Stunde
- Direkte Elastizitäten AV:
  Preis: –0.060 | Zeit: –0.151<br>
  → Reisezeit ist ca. 2.5-mal entscheidender als Preis

## Kontakt
Raphael Meier<br>
Hochschule Luzern – Wirtschaft<br>
raphael.meier@stud.hslu.ch<br>
Betreut von: Dr. Martin Schonger<br>
Februar 2026