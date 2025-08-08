import argparse
from src.features import extract_features
from src.sampling import run_sampling
from src.sidewalks import process_sidewalks
from src.survey import analyze_survey
from src.utils import setup_logging

STEPS = {
    "features": extract_features,
    "sampling": run_sampling,
    "sidewalks": process_sidewalks,
    "survey": analyze_survey,
}

def main():
    logger = setup_logging()
    parser = argparse.ArgumentParser(description="Run capstone workflow steps")
    parser.add_argument("--steps", nargs="+", default=["features", "sampling", "sidewalks", "survey"],
                        help=f"Subset of steps to run in order. Options: {list(STEPS.keys())}")
    args = parser.parse_args()

    for step in args.steps:
        func = STEPS.get(step)
        if not func:
            logger.warning("Unknown step: %s", step)
            continue
        logger.info("=== Running step: %s ===", step)
        func()

if __name__ == "__main__":
    main()
