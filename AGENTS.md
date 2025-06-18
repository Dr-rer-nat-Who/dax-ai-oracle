Kurzüberblick
Dieses Dokument liefert eine komplette, entscheidungsfreie Bauanleitung – vom Daten-Ingest bis zum Streamlit-Frontend – für ein lokales KI-System, das DAX-Kurssdaten in drei Zeitebenen auswertet, 27 Modell-Varianten trainiert, automatisch backtestet, Speicher aufräumt und Ergebnisse live anzeigt. Alle Parameterbereiche, Datei­pfade und Befehle sind vorgegeben, sodass du das Projekt mit einem CLI-Aufruf unbeaufsichtigt durchrechnen lassen kannst.

⸻

1 Architektur im Überblick

Prefect Flows  ➜  Data Lake (Parquet +DVC)  ➜  Feature-Pipelines
           ⬇                                ↘
        Optuna Studies → MLflow Runs        Backtesting
           ⬇                                     ↘
   Best-Artefakte (pickled)              Streamlit-Dashboard

•Task-Orchestrierung: Prefect 2.0 mit Checkpoints, sodass ein Stromausfall halb­fertige Trainings wiederaufnehmen kann.
•Artefakt-Versionierung: MLflow; schwere Datasets wandern in einen DVC-Cache, der sich selbst per dvc gc entschlackt.
•Backtesting: vektorisiertes Screening via vectorbt und realitätsnaher Broker-Sim via backtrader (Gebühr + Slippage).

⸻

2 Datenebene

FrequenzZeitraumSpeicher­zielRotation
Minute (1 min)letzte 90 Tagedata/raw/1min/ältere Blöcke werden beim Cleanup zu 5-Min-Bars verdichtet
Stunde (60 min)24 Monatedata/raw/1h/Behalten
Tag10 Jahredata/raw/1d/Behalten

Quelle: yfinance (OHLCV, keine Token) – unterstützt auch Minutendaten bis 30 Tage am Stück, daher nächtlicher Increment‐Pull.

Row-based Parquet spart ↘ 80 % gegenüber CSV; typische Minute-DAX-Aktie ≈ 1,5 MB/Tag → 135 MB für 90 Tage.

⸻

3 Feature-Engineering
1.TA-Indikatoren (245 Stück) via TA-Lib.
2.Price-Action-Stats: Kenndaten der letzten n Kerzen (Wick-Ratio, Gap, ATR).
3.Window-Embeddings: Normierte Returns der letzten N = [32, 64] Schritte als Vektor.
4.Datetime: Wochentag, Monatsende, Feiertag‐Flag (Eurex-Kalender).
5.Optional Exogene: DAX-Future-Spread (Börsenfeiertage frei).

Alle Transformer-Modelle (PatchTST, TimesNet, …) bekommen rohen Window-Tensor; tabulare Modelle erhalten flache Features.

⸻

4 Label-Schemata

KürzelDefinitionAnwendung
B1Binär ↑/↓, Horizon = 1 SchrittKlassische ML
T3+1 > +0,2 %, 0 ∈ ±0,2 %, –1 < –0,2 %RL, Multi-Class
Log-Return in b %= 1 SchrittRegressoren

⸻

5 Modell-Portfolio & konkrete Varianten

FamilieTrainings­frequenzLabelHyperparameter-Suche (Optuna)Quellen
LightGBM1d, 1h, 1 minB1num_leaves [31-127], max_depth [–1,15], learning_rate [0.005-0.2], n_estimators [300-1500] 3-fold TS-CV
CatBoost1d, 1h, 1 minB1depth [4-10], l2_leaf_reg [1-9], bagging_temperature [0-1]
TabNet1d, 1hB1n_d [16-64], n_steps [3-7], gamma [1-2], mask_type {entmax, sparsemax}
Prophet1dRchangepoint_prior [0.01-0.5], seasonality_prior [5-15]
N-Linear1d, 1hRkernel_size [3-11], num_blocks [1-4]
LSTM1d, 1h, 1 minB1,Rlayers [1-3], hidden [64-256], dropout [0-0.3]
Temporal Fusion Transformer1d, 1h, 1 minRAuto-tune via optimize_hyperparameters (max_epochs = 20)
Autoformer1d, 1hRseq_len [96-192], label_len [48-96], d_model [256-512]
Informer1d, 1hRprob_sparse_k [5-10], n_heads [4-8]
PatchTST1d, 1h, 1 minRpatch_len [16-64], d_model [128-512]
TimesNet1d, 1hRscale_factor [1-4], d_model [256-512]
FinRL-PPO1d, 1hT3gamma [0.9-0.99], lr [1e-5-3e-4], n_steps [128-2048]

Gesamtvarianten: 3 Frequenzen × 3 Labels × (klassisch 3 + DL 6 + RL 1)   → 27 Modelle. Jede Optuna-Studie läuft 60 Trials, Early-Stopping mit Median-Pruner (patience = 3).

⸻

6 Trainings-Pipeline
1.Prefect-Flow train_and_evaluate
•zieht Streaming-Batches direkt aus Parquet (Arrow-Scanner).
•führt pro Modellfamilie eine Optuna-Studie aus (GPU-fällt automatisch auf MPS-Backend).
•loggt jeden Trial in MLflow; nur beste 5 Runs bleiben erhalten, alle anderen Artefakte werden nach Studien-Ende gelöscht.
2.Checkpoints liegen in ~/checkpoints/{flow_run} und werden nach erfolgreichem Durchlauf entfernt.
3.Storage-Cleanup-Flow
•dvc gc -w (verw. Workspace) nach jedem Backtest.
•Parquet-Blöcke, die nicht mehr in der aktuellen Prefect-Cache-ID referenziert sind, werden gelöscht (Minute→5-Min Verdichtung).

⸻

7 Backtesting & Evaluation

Vectorbt (Schnell-Screen)
•simuliert jedes Klassifikator-Output als je-Step-Order, 0,01 % Slippage.
•Metriken: Accuracy, Balanced-Acc, F1.

Backtrader (Broker-Realismus)
•Gebühren: 0,04 % pro Trade (Xetra-Durchschnitt).
•Slippage: slip_perc = 0.05 (= 0,05 %) und CommInfoQuick‐Schema.
•Kennzahlen: Sharpe, Sortino, Max-Drawdown, CAGR.

Ranking-Score

final_score = 0.4 * balanced_accuracy
            + 0.6 * min(Sharpe / 3, 1)   # normalisiert auf [0,1]

Nur Modelle mit final_score ≥ 0.6 wandern ins Dashboard-Verzeichnis mlruns/best/.

⸻

8 Frontend

Streamlit 1.34 mit Auto-Refresh alle 60 s:

SeiteInhalt
LiveTicker-Select, Echtzeit-Kurs, aktuell ausgelöste Signale
Leaderboardsortierbare Tabelle aller Models + final_score
EquityInteractive Plot (vectorbt Figure)
ExplainFeature-Importance (LightGBM) & Attention-Heatmaps (TFT)

Dashboard lädt nur Pickle-Modelle + Metrics-CSV (≤ 200 kB/Modell), keine Raw-Parquet-Files.

⸻

9 Ressourcen & Laufzeit
•Hardware: Apple M1; PyTorch nutzt Metal-MPS → ca. 7-8× schneller als CPU bei CNN/LSTM-ähnlichen Workloads.
•Laufzeit-Schätzung:
•Tabulare Modelle (1 min) ≈ 30 min pro Study
•PatchTST (1 min) ≈ 4 h pro Study
•Gesamt-End-to-End für alle 27 Modelle ≈ 18-22 h.

⸻

10 Verzeichnis­struktur

.
├── cli.py                # entrypoint: python cli.py run-all
├── configs/
│   ├── data.yaml         # Start/End dates, tickers
│   ├── optuna.yaml       # HP search spaces
│   └── cleanup.yaml      # retention_days, min_freq
├── data/                 # Parquet + .dvc files
├── features/
├── models/               # model_zoo/*.py   (classes above)
├── prefect/
│   ├── ingest.py
│   ├── feature_build.py
│   ├── train_and_evaluate.py
│   ├── backtest.py
│   └── cleanup.py
└── dashboard/            # Streamlit app

python cli.py run-all --freq all --cleanup yes
lädt Daten → trainiert → backtestet → räumt auf → startet Dashboard auf localhost:8501.

⸻

11 Betrieb & Überwachung
•Prefect-UI (port 4200) zeigt Echtzeit-Logs, Resumes nach Crash.
•MLflow-UI (port 5000) für Metrik-Vergleich, Artefakt-Download.
•Disk-Guard: wenn freie Platte < 5 GB, wird Cleanup-Flow getriggert, der alle Parquet-Partitionen > 90 Tage löscht.

⸻

12 Erweiterungen (optional)
•Online-Learning-Layer mit river für 1-Min-Daten.
•Ensemble-Stacking (LightGBM + PatchTST Output → Meta-CatBoost).
•Order-Routing an IBKR TWS via backtrader-IB API.

⸻
