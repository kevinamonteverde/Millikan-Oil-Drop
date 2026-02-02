# Millikan Oil Drop Experiment

This repository contains code and data for analyzing the Millikan oil drop experiment to measure the elementary charge (e).

## Overview

The Millikan oil drop experiment measures the charge of individual oil droplets suspended in an electric field. By observing the fall velocity (gravity only) and rise velocity (with electric field), we can calculate the charge on each droplet and verify that charge is quantized in units of the elementary charge e ≈ 1.602 × 10⁻¹⁹ C.

## Repository Structure

```
.
├── Vids/                          # Raw trial videos
│   ├── trial1drop1.mov
│   ├── trial2.mov
│   ├── trial3.mov
│   └── trial4.mov
├── data/
│   └── simulated/                 # Simulated data for testing
│       ├── perfect_drops.csv
│       ├── smeared_fall_velocities.csv
│       ├── smeared_rise_velocities.csv
│       ├── q_estimates.csv
│       ├── q_per_drop.csv
│       ├── q_summary.csv
│       └── q_histogram.png
├── milikan_oil_simulation/        # Simulation code
│   └── main.py                    # Generates synthetic droplet data with noise
├── milikan_oil_analysis/          # Analysis code for simulated data
│   └── main.py                    # Estimates charge from simulated velocities
├── millikan_tracker.py            # Main video tracking script
├── calculate_charge.py            # Calculates elementary charge from velocities
├── verify_charge.py               # Verification and quality assessment
├── millikan_results.json          # Detailed tracking results (JSON)
├── millikan_velocities.csv        # Individual velocity measurements
├── millikan_summary.csv           # Summary of fall/rise velocities per droplet
├── charge_calculations.csv        # Calculated charges for each droplet
├── final_charge_results.csv       # Final processed results
├── optimal_charge_results.csv     # Optimized charge calculations
├── sensitivity_results.csv        # Sensitivity analysis results
├── millikan_analysis.xlsx         # Excel analysis spreadsheet
├── Millikan_Methodology_Guide.docx # Lab methodology documentation
├── Millikan-Oil-Drop-Manual-AP-8210A-corrected.pdf  # PASCO apparatus manual
├── tracking_visualization.mp4     # Video showing tracked droplets
└── tracking_visualization_stable.mp4  # Stabilized tracking visualization
```

## Main Scripts

### 1. `millikan_tracker.py` - Droplet Tracking

Automatically tracks oil droplets in video recordings using computer vision (OpenCV).

**What it does:**
- Calibrates pixel-to-mm conversion using the reticle grid
- Detects droplets as bright spots against the background
- Tracks droplet positions across frames
- Segments motion into rise/fall periods
- Calculates velocities in mm/s

**Usage:**
```python
from millikan_tracker import MillikanDropletTracker, process_video

# Process a single video
tracker = MillikanDropletTracker("Vids/trial1drop1.mov")
tracker.track_droplets()
results = tracker.analyze_velocities()
```

**Output:** `millikan_results.json`, `millikan_velocities.csv`

### 2. `calculate_charge.py` - Charge Calculation

Calculates the elementary charge using the Millikan formula from the PASCO AP-8210A manual.

**Formula:**
```
q = (4/3) × π × ρ × g × a³ × (vf + vr) / (E × vf)
```

Where:
- `a` = droplet radius (calculated from fall velocity using Stokes' Law with correction)
- `vf` = fall velocity
- `vr` = rise velocity
- `E` = electric field (V/d)

**Key parameters (edit as needed):**
- `d = 7.6e-3 m` - plate separation
- `V = 500 V` - plate voltage
- `rho = 886 kg/m³` - oil density
- `eta = 1.82e-5 N·s/m²` - air viscosity

**Usage:**
```bash
python calculate_charge.py
```

**Output:** `charge_calculations.csv`

### 3. `verify_charge.py` - Results Verification

Provides statistical analysis and quality assessment of the calculated elementary charge.

**What it shows:**
- All droplet measurements in a table
- Mean, median, and standard deviation of e estimates
- Percentage of measurements within 1%, 2%, 5%, 10% of accepted value
- Quantization verification

**Usage:**
```bash
python verify_charge.py
```

## Simulation Code

### `milikan_oil_simulation/main.py`

Generates synthetic droplet data for testing the analysis pipeline:
- Creates 10 random "perfect" drops with known charges (1-10 electrons)
- Adds 10% Gaussian noise to simulate real measurements
- Outputs to `data/simulated/`

### `milikan_oil_analysis/main.py`

Analyzes the simulated data to verify the charge estimation method works correctly.

**Usage:**
```bash
cd milikan_oil_simulation && python main.py
cd milikan_oil_analysis && python main.py
```

## Requirements

```
numpy
pandas
opencv-python
scipy
matplotlib
```

Install with:
```bash
pip install numpy pandas opencv-python scipy matplotlib
```

## Quick Start

1. **For real experimental data:**
   ```bash
   # Track droplets in videos (already done - results in millikan_results.json)
   python millikan_tracker.py

   # Calculate charges
   python calculate_charge.py

   # Verify results
   python verify_charge.py
   ```

2. **For simulated data:**
   ```bash
   cd milikan_oil_simulation && python main.py
   cd ../milikan_oil_analysis && python main.py
   ```

## Notes

- The tracking script paths may need to be updated for your system
- Plate voltage and separation should match your experimental setup
- The large tracking video (`millikan_tracking_full.mp4`, 696 MB) is excluded from this repository due to GitHub file size limits

## References

- PASCO AP-8210A Millikan Oil Drop Apparatus Manual (included as PDF)
- Millikan, R.A. (1913). "On the Elementary Electrical Charge and the Avogadro Constant"
