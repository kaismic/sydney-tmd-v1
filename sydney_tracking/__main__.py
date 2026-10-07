import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Import completed Sydney study outputs into MLflow without retraining")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--experiment", default="Sydney TMD V1")
    args = parser.parse_args(argv)
    from .importer import Importer
    result = Importer(args.run_dir, args.tracking_uri, args.experiment).run()
    print(json.dumps(result, indent=2), flush=True)
    print(f"{args.tracking_uri}/#/experiments/{result['experiment_id']}/runs/{result['run_id']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
