"""`agr` command line."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    from . import evaluate, serve

    p = argparse.ArgumentParser(prog="agr", description="Serve and evaluate Agr decision models")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("serve", help="serve the decision API and the demos")
    serve.add_args(v)
    v.set_defaults(fn=serve.run)

    e = sub.add_parser("eval", help="score a System One endpoint on a labelled file")
    c = sub.add_parser("compare", help="compare two `agr eval --out` files question by question")
    evaluate.add_args(e, c)
    e.set_defaults(fn=evaluate.evaluate)
    c.set_defaults(fn=evaluate.compare)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
