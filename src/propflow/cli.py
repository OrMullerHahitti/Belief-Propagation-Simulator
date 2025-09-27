"""Command-Line Interface for the PropFlow Simulator.

This module provides a basic CLI for interacting with the PropFlow package.
Currently, its main function is to display the package version.
"""

import argparse
from ._version import __version__


def main() -> None:
    """The main entry point for the command-line interface.

    Parses command-line arguments and executes the corresponding action.
    """
    parser = argparse.ArgumentParser(
        description="Belief Propagation Simulator command line interface"
    )
    parser.add_argument(
        "--version", action="store_true", help="Print package version and exit"
    )
    args = parser.parse_args()

    if args.version:
        print(f"PropFlow Version: {__version__}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
