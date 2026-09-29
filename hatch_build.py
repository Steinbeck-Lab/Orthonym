"""Install-time prefetch of the pinned OPSIN and centres jars.

Runs when the wheel is built (``pip install.`` and ``pip install -e.``). It
downloads both jars from their official releases into the user cache and checks
their. The jars are NOT added to the wheel, so no build output ever
redistributes them.

This is best effort: a failed download does not fail the install (pip hides
build output, and a wheel may be installed elsewhere). The guarantee is at run
time: a jar that is still missing is downloaded the first time Orthonym needs it,
and Orthonym refuses to name when it can neither find nor download a jar, telling
the user to run ``orthonym --fetch-jars``.
"""
import importlib.util
import os
import sys

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version, build_data):
        if os.environ.get("ORTHONYM_SKIP_JAR_PREFETCH", "").strip().lower() in ("1", "true", "yes", "on"):
            return
        path = os.path.join(self.root, "src", "orthonym", "jars.py")
        try:
            spec = importlib.util.spec_from_file_location("_orthonym_jars_prefetch", path)
            jars = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(jars)
            jars.fetch_all(verbose=True)
        except Exception as exc:  # never fail the install; the runtime check is the guarantee
            print(f"[orthonym] could not prefetch the OPSIN/centres jars ({exc}); "
                  f"run `orthonym --fetch-jars` after installing.", file=sys.stderr)
