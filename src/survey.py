from pathlib import Path
from .config import DATA_DIR, FIGURES_DIR, OUTPUT_DIR, RANDOM_SEED, CRS_WGS84
from .utils import setup_logging, ensure_dir
import argparse


def survey():
    # Packages
    import pandas as pd
    import numpy as np
    from collections import Counter
    import re

    # Read data and change column names
    df = pd.read_csv(r"C:\Users\40009389\Documents\data\Survey123\survey.csv")
    df.columns = ["ObjectID", "GlobalID", "CreationDate", "Creator", "EditDate", "Editor", 
                  "CurrentDate", "SurveyType", "ComplaintID", "BaitType", "BaitFound", 
                  "BaitAdded", "BaitLeft", "HoleType", "WaterVolume", "NumberRats", 
                  "Comments", "BaitTypeOther", "Address", "BaitingGeneral", "NumberBurrows", 
                  "LicenseNumber", "ManHoleID", "SmokeTestResult", "NumberSewers", "x", "y"]

    df.head()

    # Distribution of complaint types
    c = Counter(df["SurveyType"])
    c.most_common()

    # Doesn't seem like that many Complaint-Based...

    # How many unique complaint IDs are there?
    complaint_df = df[df.SurveyType=="Complaint-Based"]
    print(f"Number of Unique Complaints Addressed: {len(complaint_df.ComplaintID.unique())}")

    # How many of the unique complaint IDs come from lagan?
    def is_number(string):
        if pd.isna(string):
            return False
        if re.match(r"\d+", string) is not None:
            return True
        else:
            return False

    complaint_df_valid_id = complaint_df[complaint_df.ComplaintID.apply(is_number)]
    print(f"Number of Valid Unique Complaints Addressed: {len(complaint_df_valid_id.ComplaintID.unique())}")


    # Since when has Survey123 been used?
    df.CreationDate = pd.to_datetime(complaint_df.CreationDate, format="%m/%d/%Y %I:%M:%S %p")
    print(f"First Survey123 Entry: {complaint_df.CreationDate.min()}")

    # But there have been 7215 complaints since then according to lagan rodent table...

    # df[~(df.ComplaintID.apply(is_number) | pd.isna(df.ComplaintID))]
    df[df.ComplaintID == "101004954739"]





# Backward compatibility: call into original function if present
def analyze_survey(**kwargs):
    """
    Public entry point for this step. Accepts keyword args to customize behavior.
    """
    logger = setup_logging()
    logger.info("Starting analyze_survey()")
    if "survey" in globals():
        return survey(**kwargs) if callable(globals().get("survey")) else None
    # Fallback: nothing to run
    logger.warning("Original function 'survey()' not found; nothing executed.")
    return None

def _cli():
    parser = argparse.ArgumentParser(description="analyze_survey step")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Path to data dir")
    parser.add_argument("--figures-dir", type=str, default=str(FIGURES_DIR), help="Path to figures dir")
    args = parser.parse_args()
    ensure_dir(Path(args.figures_dir))
    return analyze_survey()

if __name__ == "__main__":
    _cli()
