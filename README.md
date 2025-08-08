# Sara–Sasha Capstone  
**Framework to prepare geospatial data and apply pseudosampling for rodent activity analysis in Boston**  

This repository contains the data preparation and sampling workflow developed for the Sara–Sasha MIT Capstone project. The code integrates multiple Boston datasets — including 311 rodent complaints, survey data, sidewalk maps, and census block information — into a unified feature set for later predictive modeling.  

## Current Capabilities  
- **Feature extraction** – Clean, merge, and encode geographic/demographic predictors from multiple raw datasets.  
- **Pseudosampling** – Generate stratified spatial-temporal samples for robust model-ready datasets.  
- **Sidewalk data processing** – Integrate pedestrian infrastructure data for environmental context.  
- **Survey analysis** – Preprocess field survey data for use as additional predictors.  

> **Note:** This repository focuses on data preparation and sampling. 

## Project Structure  
```
src/
  __init__.py
  config.py       # Paths, CRS, constants, random seed
  utils.py        # Logging, helper functions
  features.py     # extract_features() – builds the feature set
  sampling.py     # run_sampling() – applies pseudosampling
  sidewalks.py    # process_sidewalks()
  survey.py       # analyze_survey()
main.py           # Orchestrates workflow steps via CLI
requirements.txt  # Dependencies
data/             # Local datasets (not tracked)
figures/          # Output plots/maps (generated)
```

## Installation  
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage  
Run the **full workflow**:  
```bash
python main.py
```

Run a **subset of steps** (example – pseudosampling and sidewalks):  
```bash
python main.py --steps sampling sidewalks
```

Run any step **standalone**:  
```bash
python -m src.features
python -m src.sampling
python -m src.sidewalks
python -m src.survey
```

## Configuration  
Set defaults in `src/config.py` or override via environment variables:  
- `CAPSTONE_PROJECT_ROOT` — project root path  
- `CAPSTONE_DATA_DIR` — data folder path  
- `CAPSTONE_FIGURES_DIR` — figures output path  
- `CAPSTONE_OUTPUT_DIR` — additional outputs path  
- `CAPSTONE_RANDOM_SEED` — reproducibility seed  

## Next Steps  
- Add predictive modeling module (e.g., scikit-learn, XGBoost).  
- Implement evaluation metrics and reporting.  
- Integrate model results back into the pipeline for targeted intervention strategies.  
