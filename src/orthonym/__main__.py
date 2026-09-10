"""
Entry point for `python -m orthonym`.

Supports two modes:
  python -m orthonym "CCO" -> name a SMILES (existing CLI)
  python -m orthonym validate "name" -> OPSIN round-trip validation
"""

import sys


def main():
    """Route to naming CLI or validation subcommand."""
    args = sys.argv[1:]

    # Check for validate subcommand
    if args and args[0] == "validate":
        return _run_validate(args[1:])

    # Default: pass through to existing CLI
    from .cli import main as cli_main
    return cli_main(args)


def _run_validate(args):
    """Run OPSIN round-trip validation on a name.

    Usage: python -m orthonym validate "ethanol"
           python -m orthonym validate "ethanol" --smiles CCO
           python -m orthonym validate "ethanol" --version 2.8.0
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate IUPAC name via OPSIN round-trip (FMT-06)",
        prog="python -m orthonym validate",
    )
    parser.add_argument(
        "name",
        help="IUPAC name to validate",
    )
    parser.add_argument(
        "--smiles",
        help="Original SMILES for InChI comparison (optional)",
    )
    parser.add_argument(
        "--version",
        default="2.9.0",
        help="OPSIN JAR version (default: 2.9.0)",
    )
    parser.add_argument(
        "--both-versions",
        action="store_true",
        help="Test against both OPSIN 2.8.0 and 2.9.0",
    )

    parsed = parser.parse_args(args)

    from .validation.format_validator import validate_name_format
    from .validation.opsin_roundtrip import (
        opsin_parse,
        opsin_parse_both_versions,
        opsin_roundtrip_check,
    )

    name = parsed.name
    print(f"Name: {name}")

    # Format validation
    fmt_ok, fmt_reason = validate_name_format(name)
    print(f"Format valid: {fmt_ok} ({fmt_reason})")

    if parsed.both_versions:
        # Cross-version test
        result = opsin_parse_both_versions(name)
        print(f"OPSIN 2.8.0: {result['v28'] or 'FAILED'}")
        print(f"OPSIN 2.9.0: {result['v29'] or 'FAILED'}")
        print(f"Versions agree: {result['agree']}")
    elif parsed.smiles:
        # Full round-trip check
        result = opsin_roundtrip_check(
            parsed.smiles, name, jar_version=parsed.version
        )
        print(f"OPSIN SMILES: {result['opsin_smiles'] or 'FAILED'}")
        print(f"InChI match: {result['inchi_match']}")
        print(f"Passed: {result['passed']}")
        if result["error"]:
            print(f"Error: {result['error']}")
    else:
        # Simple parse check
        opsin_smiles = opsin_parse(name, jar_version=parsed.version)
        print(f"OPSIN parse: {opsin_smiles or 'FAILED'}")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
