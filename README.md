# Vision Based Landing Detection system

Vision-only apogee and leg-deployment timing for CanSat descent. Which the model only takes the cansat's
vision as the input while taking accelerometer and barometer post launch reference to determine te accuracy of the model during field deployment. Due to the restriction of cansat weight and cost of on-boarding module, LoRa is used as both up and downlink to recieve camera data as well as signal to deploy the leg based from ground. Laptop receives the camera data and IMU data from the cansat and processes it using the trained model to determine the right moment to deploy the leg. 

---

## Architecture Overview

```mermaid
flowchart TB
    subgraph INTERNET ["Internet Sources"]
        V[Downloaded Launch Videos]
    end

    subgraph PIPELINE ["Flow 1 — Training Data"]
        EX[OpenCV Feature Extraction]
    end

    subgraph TRAINING ["Flow 2 — Model Training"]
        CSV[72-col Per-frame CSV]
        HUM{Human Annotation}
        LD[Labeled Dataset]
        XGB[XGBoost Classifier]
        LR[Logistic Regression]
        SVM[SVM Baseline]
        ACC{Accuracy Comparison}
        BEST[Best Model]
    end

    subgraph FLIGHT ["Flow 3 — Live Flight"]
        subgraph CANSAT ["Cansat"]
            BARO[Barometer]
            GYRO[Accelerometer]
            CAM[Camera]
            SERVO[Servo Motor]
            BOARD[Main Board]
        end
        subgraph GROUND ["Ground Station"]
            LORA_RX[LoRa Rx]
            PROC[ML Inference]
            SIGNAL[Deploy Signal]
        end
        MODEL[flight_model.xgb]
        LORA_RF[LoRa RF Link]
    end

    V --> EX
    EX --> CSV
    CSV --> HUM
    HUM --> LD
    LD --> XGB
    LD --> LR
    LD --> SVM
    XGB --> ACC
    LR  --> ACC
    SVM --> ACC
    ACC --> BEST

    BEST --> MODEL
    BARO -. "telemetry (not processed)" .-> BOARD
    GYRO -. "telemetry (not processed)" .-> BOARD
    CAM --> BOARD
    BOARD -- "IMU + frames (LoRa downlink)" --> LORA_RF
    LORA_RF --> LORA_RX
    LORA_RX --> PROC
    MODEL --> PROC
    PROC -- "deploy = YES" --> SIGNAL
    SIGNAL --> LORA_RF
    LORA_RF -- "uplink" --> BOARD
    BOARD --> SERVO

    style INTERNET fill:#f6f6f6,stroke:#8a8a8a
    style PIPELINE fill:#eeeeee,stroke:#767676
    style TRAINING fill:#ececec,stroke:#626262
    style FLIGHT  fill:#e8e8e8,stroke:#555555
    style CANSAT  fill:#e0e0e0,stroke:#444444
    style GROUND  fill:#d8d8d8,stroke:#333333
```

---

## Flow 1 — Training Data Pipeline

```mermaid
flowchart LR
    A[Internet Videos]
    B[OpenCV — Extract Frames\nand Optical Flow Features]
    C[72-column Per-frame CSV]
    D[Human Annotation\nTimestamp Injection]
    E{Labeled Event Markers}
    F[Labeled Dataset\nLaunch / Apogee / Deploy / Landing]

    A --> B --> C --> D --> E --> F

    style A fill:#f6f6f6,stroke:#8a8a8a,color:#000000
    style B fill:#e0e0e0,stroke:#444444,color:#000000
    style C fill:#eeeeee,stroke:#767676,color:#000000
    style D fill:#ececec,stroke:#626262,color:#000000
    style E fill:#dcdcdc,stroke:#4a4a4a,color:#000000
    style F fill:#e0e0e0,stroke:#444444,color:#000000
```

---

## Flow 2 — Model Training

```mermaid
flowchart LR
    A[Labeled Dataset]
    B[XGBoost Classifier]
    C[Logistic Regression]
    D[SVM Baseline]
    E{Accuracy Comparison}
    F[F1 / ROC-AUC Evaluation]
    G[flight_model.xgb]

    A --> B --> E
    A --> C --> E
    A --> D --> E
    E --> F --> G

    style A fill:#e0e0e0,stroke:#444444,color:#000000
    style B fill:#f6f6f6,stroke:#8a8a8a,color:#000000
    style C fill:#e0e0e0,stroke:#444444,color:#000000
    style D fill:#eeeeee,stroke:#767676,color:#000000
    style E fill:#ececec,stroke:#626262,color:#000000
    style F fill:#dcdcdc,stroke:#4a4a4a,color:#000000
    style G fill:#d0d0d0,stroke:#222222,color:#000000
```

---

## Flow 3 — Live Flight Operation

```mermaid
flowchart TB
    subgraph CANSAT ["Cansat (onboard)"]
        BARO[Barometer]
        GYRO[Accelerometer / Gyro]
        CAM[Camera]
        SERVO[Servo Motor]
        BOARD[Main Board]
        BARO -. "telemetry (not processed onboard)" .-> BOARD
        GYRO -. "telemetry (not processed onboard)" .-> BOARD
        CAM --> BOARD
    end

    subgraph GROUND ["Ground Station (laptop)"]
        LORA_RX[LoRa Receiver]
        FRAMES[Camera Frames]
        ML[ML Model — flight_model.xgb]
        PRED[Deploy Decision]
        SIGNAL[Deploy Signal — YES]
    end

    LORA_RF[LoRa RF Link]

    BOARD -- "IMU telemetry + frames\n(LoRa downlink)" --> LORA_RF
    LORA_RF --> LORA_RX
    LORA_RX --> FRAMES
    FRAMES --> ML
    ML --> PRED
    PRED -- "deploy = YES" --> SIGNAL
    SIGNAL --> LORA_RF
    LORA_RF -- "uplink" --> BOARD
    BOARD -- "actuates" --> SERVO

    style CANSAT fill:#e0e0e0,stroke:#444444
    style GROUND fill:#d8d8d8,stroke:#333333
```
## Feature extraction Pipeline

```mermaid
flowchart LR
    A[Video Frame]
    B[Shi-Tomasi Corners]
    C[Lucas-Kanade Optical Flow]
    D[FOE Estimation\nleast-squares]
    E[Radial Speeds vs FOE]
    F[Dense Flow — Farneback]
    G[Divergence and Grid]
    H[Affine and Homography]
    I[Appearance Stats]
    J[Edges / Texture / Sharpness\nSky / Ground]
    K[Velocities, Acceleration\nState, Phase]
    CSV[72-col Per-frame CSV]

    A --> B --> C
    C --> D --> E
    C --> F --> G
    C --> H
    A --> I --> J
    E --> K
    G --> K
    H --> K
    J --> K
    K --> CSV

    style A  fill:#f6f6f6,stroke:#8a8a8a
    style CSV fill:#e0e0e0,stroke:#444444
```

---

## Quick start

```bash
git clone https://github.com/7-wwh/AeroNottsCanSat-Landing-Detect.git 
cd AeroNottsCanSat-Landing-Detect
pip install -r requirements.txt

# 1. Extract vision features from a flight video → annotated video + CSV + plot
python rocket_flow.py flight.mp4

# 2. Train the model on your labeled dataset (after manual annotation)
python flight_ops/trainer.py data/labeled/flight_dataset.csv

# 3. Run ground station on flight day
python flight_ops/receiver.py --model flight_model/model.xgb
```

### Useful `rocket_flow.py` flags

| Flag | Purpose |
|------|---------|
| `--scale 0.5` | Process at half resolution (faster) |
| `--max-corners 500` | Max Shi-Tomasi corners to track |
| `--thresh 0.1` | Radial speed threshold for state classification |
| `--smooth 15` | Savitzky-Golay window for velocity smoothing |
| `--no-dense` | Skip Farneback dense flow (faster) |
| `--draw-foe` | Draw the Focus of Expansion crosshair |
| `--synthetic out.mp4` | Generate a test video (expansion→contraction) |

---

## Feature extraction

Each frame yields 72 features across 8 families, all written to the CSV:

| Family | Columns | Source |
|--------|---------|--------|
| Time + Labels | 3 | `time_s`, `state`, `phase` |
| Sparse radial | 6 | Mean/median/std/p95 radial speed about the FOE |
| Displacement | 4 | Tracked-point motion distance stats |
| Cloud distribution | 4 | Point count, density, spread |
| Direction histogram | 8 | 8-bin flow direction compass |
| FOE | 2 | Smoothed focus-of-expansion coordinates |
| Dense flow | 28 | Farneback magnitude, divergence, 3×3 grid |
| Camera ego-motion | 24 | Affine + homography + residual flow |
| Appearance | 6 | Edges, texture, sharpness, sky/ground |
| Horizon | 3 | Hough-line horizon angle/position/confidence |
| Temporal | 2 | Smoothed expansion rate + acceleration |

See [`video input/csv output/CSV_COLUMN_REFERENCE.md`](video%20input/csv%20output/CSV_COLUMN_REFERENCE.md)
for a full column-by-column walkthrough.

---

## Project structure

```
AeroNottsCanSat-Landing-Detect/
├── README.md                 ← this file
├── rocket_flow.py            ← CLI: video → 72-col CSV + annotated video
├── flight_ops/
│   ├── trainer.py            ← XGBoost + baselines, model selection
│   ├── receiver.py           ← LoRa ground station + ML inference loop
│   └── servo_controller.py   ← sends deploy signal via LoRa uplink
├── scripts/                  ← vision feature extraction library
│   ├── io.py                 ← video I/O, output paths
│   ├── state.py              ← flight-phase classification
│   ├── schema.py             ← 72-column CSV schema
│   ├── draw.py               ← HUD, trails, FOE crosshair
│   ├── synth.py              ← synthetic test-video generator
│   ├── plot.py               ← metrics visualization
│   └── features/
│       ├── sparse.py         ← Shi-Tomasi, LK, FOE, radial speed
│       ├── dense.py          ← Farneback divergence, grid, magnitude
│       ├── camera.py         ← affine + homography ego-motion
│       ├── appearance.py     ← edges, texture, sharpness, sky/ground
│       └── horizon.py        ← Hough horizon (experimental)
└── video input/
    └── csv output/           ← generated CSVs, FEATURES.md, CSV_COLUMN_REFERENCE.md
```

---

## Live flight setup

**Hardware** — five peripherals on the cansat's main board:

| Component | Role |
|-----------|------|
| Barometer | Altitude telemetry (streamed, not processed onboard) |
| Accelerometer/Gyro | IMU telemetry (streamed, not processed onboard) |
| LoRa Radio | Downlink (IMU + frames → ground) + uplink (deploy signal → cansat) |
| Camera | Captures frames for vision processing |
| Servo Motor | Deploys shock-absorbing legs (actuated locally by cansat board) |

**Deploy logic** — The XGBoost model evaluates each incoming frame. When
`radial_expansion` plateaus and `ground_fraction` rises (cansat is in the final
moments before ground contact), it emits a deploy signal. The ground station
sends a single-bit "YES" via LoRa uplink; the cansat board drives the servo
locally and cuts power immediately after actuation to prevent jitter.

---

## License

MIT — see [`LICENSE`](LICENSE).

---

*Built by the AeroNotts Rocketry team.*
