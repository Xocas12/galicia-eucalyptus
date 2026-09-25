"""Command line.

euc run --config configs/fast.yaml          # full synthetic pipeline + report
euc catalog                                 # print the real-data source catalogue
"""

from __future__ import annotations

import argparse
import logging

from .config import load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="euc", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the end-to-end pipeline and write the report")
    r.add_argument("--config", default="configs/fast.yaml")
    r.add_argument("--out", default=None, help="override output_dir from the config")
    sub.add_parser("catalog", help="list the real data sources")
    rl = sub.add_parser("real", help="real-data pipeline for Galicia (downloads, then analysis)")
    rl.add_argument("stage", choices=["fetch", "run", "all"], nargs="?", default="all")
    rl.add_argument("--out", default="outputs/real")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.cmd == "real":
        from .real.layers import all_layers
        from .real.s2 import build_period

        if args.stage in ("fetch", "all"):
            all_layers()
            for period in ("2024", "2017"):
                build_period(period)
            from .real.landsat import EPOCHS, epoch_features, fetch_index

            fetch_index()
            for epoch in EPOCHS:
                epoch_features(epoch)
        if args.stage in ("run", "all"):
            from .real.analysis import run_real
            from .real.brief import write_brief

            path = write_brief(run_real(), args.out)
            print(f"brief written to {path}")
        return 0

    if args.cmd == "catalog":
        from .data.catalog import catalog_frame

        print(catalog_frame().to_string(index=False))
        return 0

    from .pipeline import run
    from .reporting import write_report

    cfg = load_config(args.config)
    if args.out:
        cfg.output_dir = args.out
    res = run(cfg)
    path = write_report(res, cfg.output_dir)
    print(f"report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
