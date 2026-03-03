# Dataset: ASHRAE Global Thermal Comfort Database II

## About

The **ASHRAE Global Thermal Comfort Database II** is the largest and most diverse thermal comfort dataset available. It contains right-here-right-now subjective evaluations paired with objective indoor environmental measurements from field studies conducted worldwide.

## How to Obtain

The dataset is publicly available and can be downloaded from:

- **Official source**: [ASHRAE Global Thermal Comfort Database II](https://www.kaggle.com/datasets/claytonmiller/ashrae-global-thermal-comfort-database-ii)
- **Reference**: Földváry Ličina, V., et al. (2018). "Development of the ASHRAE Global Thermal Comfort Database II." *Building and Environment*, 142, 502-512.

## Setup

1. Download the dataset CSV file
2. Place it in this directory as `df1.csv` (or specify a custom path via `--data_path`)

```
data/
└── df1.csv    ← place the downloaded file here
```

## Features Used

After preprocessing, the model uses the following features:

| Feature | Type | Description |
|---|---|---|
| Air temperature (°C) | Continuous | Indoor dry-bulb air temperature |
| Relative humidity (%) | Continuous | Indoor relative humidity |
| Air velocity (m/s) | Continuous | Indoor air speed |
| SET | Continuous | Standard Effective Temperature |
| Met | Continuous | Metabolic rate (met units) |
| Clo | Continuous | Clothing insulation (clo units) |
| Season_sin, Season_cos | Engineered | Cyclical encoding of season |
| Climate | Target-encoded | Köppen climate classification |
| + others | Various | Additional environmental parameters |

## Target Variable

**Thermal category** — 3-class classification:
- **N** (Neutral): Comfortable
- **UC** (Uncomfortably Cool): Cold discomfort
- **UW** (Uncomfortably Warm): Warm discomfort

## Note

The raw dataset is **not included** in this repository due to licensing considerations. Please download it from the official source above.
