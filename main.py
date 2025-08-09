#!/usr/bin/env python3
import argparse
import importlib
import sys

def load_callable(mod_name, candidates):
    try:
        mod = importlib.import_module(f"src.{mod_name}")
    except Exception:
        return None
    for fn in candidates:
        attr = getattr(mod, fn, None)
        if callable(attr):
            return attr
    return None

def available_steps():
    registry = {}
    registry["features"] = load_callable("features", ["extract_features", "main", "run"])
    registry["sampling"] = load_callable("sampling", ["run_sampling", "main", "run"])
    registry["sidewalks"] = load_callable("sidewalks", ["process_sidewalks", "main", "run"])
    registry["survey"] = load_callable("survey", ["analyze_survey", "main", "run"])
    registry["model_input"] = load_callable("model_input", ["main", "build_dataset", "run"])
    registry["general_model"] = load_callable("general_model", ["main", "train", "run"])
    registry["actionable_model"] = load_callable("actionable_model", ["main", "train", "run"])
    registry["address_recommendations"] = load_callable("address_recommendations", ["main", "run"])
    registry["clip_points"] = load_callable("clip_points", ["main", "run"])
    registry["get_data_db"] = load_callable("get_data_db", ["main", "run"])
    return {k: v for k, v in registry.items() if v is not None}

def main():
    steps = available_steps()
    parser = argparse.ArgumentParser(description="Project Orchestrator")
    parser.add_argument("--list", action="store_true", help="List available steps")
    parser.add_argument("--steps", nargs="+", help="Steps to run in order")
    parser.add_argument("--all", action="store_true", help="Run all available steps")
    args, unknown = parser.parse_known_args()

    if args.list:
        if not steps:
            print("No steps available.")
            sys.exit(0)
        print("Available steps:")
        for k in steps:
            print(f"- {k}")
        sys.exit(0)

    run_queue = []
    if args.all or (not args.steps):
        # Default order preference if present
        preferred = [
            "get_data_db",
            "features",
            "sampling",
            "sidewalks",
            "survey",
            "model_input",
            "general_model",
            "actionable_model",
            "address_recommendations",
            "clip_points",
        ]
        for name in preferred:
            if name in steps:
                run_queue.append(name)
        # include any other discovered steps not in preferred
        for name in steps:
            if name not in run_queue:
                run_queue.append(name)
    else:
        for name in args.steps:
            if name not in steps:
                print(f"Step not found or not importable: {name}")
                sys.exit(1)
            run_queue.append(name)

    for name in run_queue:
        fn = steps[name]
        print(f"=== Running step: {name} ===")
        try:
            # pass through unknown args to step if it supports a CLI
            if hasattr(fn, "__module__") and unknown:
                fn(*unknown)  # best-effort passthrough
            else:
                fn()
        except TypeError:
            fn()
        except SystemExit:
            pass

if __name__ == "__main__":
    main()
