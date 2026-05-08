"""
Command-line interface for Orthonym.

Usage:
    orthonym "CCO"
    orthonym "CC(=O)O" --style general
    python -m orthonym "SMILES"
"""

import argparse
import sys
from typing import List

from . import name_compound, __version__


def main(args: List[str] = None) -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Orthonym: Generate IUPAC names from SMILES",
        prog="orthonym"
    )
    
    parser.add_argument(
        "smiles",
        nargs="?",
        help="SMILES string to convert to IUPAC name"
    )
    
    parser.add_argument(
        "--style",
        choices=["pin", "general", "cas"],
        default="pin",
        help="Naming style: pin (Preferred IUPAC Name), general, or cas (default: pin)"
    )
    
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output including SMILES and style"
    )
    
    parser.add_argument(
        "--batch", "-b",
        type=str,
        help="Process multiple SMILES from a file (one per line)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for batch processing (default: stdout)"
    )

    parser.add_argument(
        "--confidence",
        action="store_true",
        help="Show confidence metadata alongside the name"
    )

    # Phase 156 Plan-03 Task 4 (CONTEXT.md D-17 telemetry):
    # Print the OPSIN-grammar pre-validation seven-bucket counter
    # histogram to stderr after naming. Always goes through the
    # `Orthonym(...).get_validation_stats()` accessor — never reads
    # a module-global counter (AP-19).
    parser.add_argument(
        "--validation-stats",
        action="store_true",
        help="Print OPSIN grammar validation counters (D-17 telemetry) to stderr"
    )

    parsed = parser.parse_args(args)
    
    # Batch processing mode
    if parsed.batch:
        return _process_batch(parsed.batch, parsed.output, parsed.style,
                              parsed.verbose, parsed.confidence)

    # Single SMILES mode
    if not parsed.smiles:
        parser.print_help()
        return 1

    try:
        # Phase 156 Plan-03 Task 4: --validation-stats branch goes
        # through Orthonym(...).get_validation_stats() per AP-19
        # (never read a module-global counter). Branch is taken BEFORE
        # the existing default `print(name)` so the histogram is the
        # only stderr output path when the flag is on.
        if parsed.validation_stats:
            from orthonym.namer import Orthonym
            namer = Orthonym(style=parsed.style)
            name = namer.name(parsed.smiles)
            print(name)
            stats = namer.get_validation_stats()
            print(f"Validation stats: {stats}", file=sys.stderr)
            return 0

        if parsed.confidence:
            result = name_compound(parsed.smiles, style=parsed.style,
                                   include_confidence=True)
            print(f"Name:       {result['name']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Handler:    {result['handler']}")
            if result.get('factors'):
                print("Factors:")
                for k, v in result['factors'].items():
                    print(f"  {k}: {v:.4f}")
        elif parsed.verbose:
            name = name_compound(parsed.smiles, style=parsed.style)
            print(f"SMILES: {parsed.smiles}")
            print(f"Style:  {parsed.style}")
            print(f"Name:   {name}")
        else:
            name = name_compound(parsed.smiles, style=parsed.style)
            print(name)

        return 0

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def _process_batch(input_file: str, output_file: str, style: str,
                    verbose: bool, confidence: bool = False) -> int:
    """Process multiple SMILES from a file."""
    try:
        with open(input_file, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        return 1
    except IOError as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        return 1

    results = []
    errors = 0

    for smiles in smiles_list:
        try:
            if confidence:
                result = name_compound(smiles, style=style,
                                       include_confidence=True)
                results.append(
                    f"{smiles}\t{result['name']}\t"
                    f"{result['confidence']:.4f}\t{result['handler']}"
                )
            else:
                nm = name_compound(smiles, style=style)
                results.append(f"{smiles}\t{nm}")
        except Exception as e:
            results.append(f"{smiles}\tERROR: {e}")
            errors += 1

    # Write output
    output_text = "\n".join(results)

    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(output_text + "\n")
            if verbose:
                print(f"Processed {len(smiles_list)} SMILES, {errors} errors")
                print(f"Output written to: {output_file}")
        except IOError as e:
            print(f"Error writing output file: {e}", file=sys.stderr)
            return 1
    else:
        print(output_text)
        if verbose:
            print(f"\nProcessed {len(smiles_list)} SMILES, {errors} errors",
                  file=sys.stderr)

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
