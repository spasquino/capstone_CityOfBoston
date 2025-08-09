# Sara–Sasha Capstone  
**Framework to prepare geospatial data and apply pseudosampling for rodent activity analysis in Boston**  

This repository contains the data preparation and sampling workflow developed for the Sara–Sasha MIT Capstone project. The code integrates multiple Boston datasets — including 311 rodent complaints, survey data, sidewalk maps, and census block information — into a unified feature set for later predictive modeling.  

All original Jupyter notebooks have been converted into pure Python scripts for easier reproducibility, automation, and integration into pipelines. The repository now contains both the converted `.py` files and existing helper modules/configurations.

## Current Capabilities  
- **Feature extraction** – Clean, merge, and encode geographic/demographic predictors from multiple raw datasets.  
- **Pseudosampling** – Generate stratified spatial-temporal samples for robust model-ready datasets.  
- **Sidewalk data processing** – Integrate pedestrian infrastructure data for environmental context.  
- **Survey analysis** – Preprocess field survey data for use as additional predictors.  
- **Actionable and General Models** – Implement and run modeling workflows from converted scripts.  
- **Address recommendations** – Generate location-based recommendations.  

> **Note:** This repository focuses on data preparation and sampling, with supporting modeling scripts included but not fully integrated into `main.py`.

## Project Structure  
```
src/
  __init__.py
  config.py                 # Paths, CRS, constants, random seed
  utils.py                  # Logging, helper functions
  features.py               # extract_features() – builds the feature set
  sampling.py               # run_sampling() – applies pseudosampling
  sidewalks.py              # process_sidewalks()
  survey.py                 # analyze_survey()
converted_py/               # Converted .py files from original notebooks
  actionable_model.py
  address_recommendations.py
  features.py
  general_model.py
  model_input.py
  sampling.py
  sidewalks.py
  clip_points.py
  get_data_db.py
  config.cfg
main.py                     # Orchestrates workflow steps via CLI
requirements.txt            # Dependencies
data/                       # Local datasets (not tracked)
figures/                    # Output plots/maps (generated)
```

## Installation  
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage  
Run the **full workflow** (core pipeline only):  
```bash
python main.py
```

Run a **subset of steps**:  
```bash
python main.py --steps sampling sidewalks
```

Run any converted script **standalone**:  
```bash
python converted_py/actionable_model.py
python converted_py/general_model.py
```

Run any original pipeline step standalone:  
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
- Integrate converted modeling scripts directly into the pipeline.  
- Add predictive modeling module (e.g., scikit-learn, XGBoost).  
- Implement evaluation metrics and reporting.  
- Integrate model results back into the pipeline for targeted intervention strategies.  
