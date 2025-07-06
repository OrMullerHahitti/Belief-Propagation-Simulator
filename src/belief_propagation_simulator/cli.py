"""Command line interface for Belief Propagation Simulator."""

import argparse
from ._version import __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Belief Propagation Simulator command line interface"
    )
    parser.add_argument(
        "--version", action="store_true", help="Print package version and exit"
    )
    args = parser.parse_args()

    if args.version:
        print(__version__)
    else:
        print(
            "Belief Propagation Simulator CLI is under development. "
            "See documentation for usage examples."
        )


if __name__ == "__main__":
    main()
