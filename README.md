# Sara–Sasha Capstone  
**Framework to prepare geospatial data and apply pseudosampling for rodent activity analysis in Boston**  

This repository contains the data preparation, sampling workflow and predictive/descriptive models developed for the City of Boston - MIT Capstone project. The original code integrates multiple Boston datasets — including 311 rodent complaints, survey data, sidewalk maps, and census block information — into a unified feature set for later predictive modeling. The data is not included in this repository for confidenciality reasons. 

## Current Capabilities  
- **Feature extraction** – Clean, merge, and encode geographic/demographic predictors from multiple raw datasets.  
- **Pseudosampling** – Generate stratified spatial-temporal samples for robust model-ready datasets.  
- **Sidewalk data processing** – Integrate pedestrian infrastructure data for environmental context.  
- **Survey analysis** – Preprocess field survey data for use as additional predictors.  
- **Actionable and General Models** – Implement and run modeling workflows from converted scripts.  
- **Address recommendations** – Generate location-based recommendations.  

> **Note:** This repository focuses on data preparation and sampling, with supporting modeling scripts included but not fully integrated into `main.py`.

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
