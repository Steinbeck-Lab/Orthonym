"""The command line's option parser, for the option reference page.

``orthonym.cli.main`` builds its parser inline. To show exactly that parser
(and no hand-kept copy of it), run ``main`` with ``parse_args`` replaced by a
function that hands the parser back and stops. Nothing is named.
"""
import argparse


class _Stop(Exception):
    pass


def parser() -> argparse.ArgumentParser:
    captured = {}
    original = argparse.ArgumentParser.parse_args

    def grab(self, args=None, namespace=None):
        captured["parser"] = self
        raise _Stop

    argparse.ArgumentParser.parse_args = grab
    try:
        from orthonym import cli
        try:
            cli.main([])
        except _Stop:
            pass
    finally:
        argparse.ArgumentParser.parse_args = original
    return captured["parser"]
