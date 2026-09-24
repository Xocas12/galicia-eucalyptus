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
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

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
