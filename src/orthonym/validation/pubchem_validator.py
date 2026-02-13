"""
PubChem PUG-REST name-to-structure lookup with local JSON file cache.

Provides a cache-first approach: check local JSON cache before making
any network request. Negative results (name not found) are also cached
to avoid repeated 404 lookups. Transient errors (timeouts, network issues)
are NOT cached so they can be retried.

Rate limiting: 0.25s sleep before each API call (4 req/sec, below
PubChem's 5 req/sec limit).
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional

import requests
import requests.utils


# Default cache location: project_root/data/pubchem_cache.json
DEFAULT_CACHE_PATH = (
    Path(__file__).parent.parent.parent.parent / "data" / "pubchem_cache.json"
)

PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
    "{name}/property/IsomericSMILES,InChI/JSON"
)

# Rate limiting: 4 requests per second (below PubChem's 5/sec limit)
MIN_DELAY = 0.25


def load_cache(cache_path: Optional[Path] = None) -> dict:
    """
    Load PubChem lookup cache from a JSON file.

    Args:
        cache_path: Path to cache file. Defaults to data/pubchem_cache.json.

    Returns:
        Dictionary mapping IUPAC names to lookup results (or None for
        cached negative results). Returns empty dict if file is missing
        or corrupt.
    """
    if cache_path is None:
        cache_path = DEFAULT_CACHE_PATH

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_cache(cache: dict, cache_path: Optional[Path] = None) -> None:
    """
    Save PubChem lookup cache to a JSON file.

    Args:
        cache: Dictionary mapping names to results.
        cache_path: Path to cache file. Defaults to data/pubchem_cache.json.
    """
    if cache_path is None:
        cache_path = DEFAULT_CACHE_PATH

    # Ensure parent directory exists
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def lookup_name_pubchem(
    name: str,
    cache: dict,
    skip_api: bool = False,
) -> Optional[Dict[str, str]]:
    """
    Look up an IUPAC name via PubChem PUG-REST API with cache.

    Cache-first: if the name is in cache, the cached value is returned
    immediately (even if it is None, indicating a previous negative result).
    Only makes an API call on cache miss.

    Args:
        name: IUPAC name to look up.
        cache: Mutable cache dictionary (updated in-place on API call).
        skip_api: If True, return None on cache miss without calling API.
                  Useful for offline/test mode.

    Returns:
        Dict with 'smiles' and 'inchi' keys on success, or None if the
        name cannot be resolved. Cached negative results also return None.
    """
    # Cache hit (includes cached None for negative results)
    if name in cache:
        return cache[name]

    # Offline mode: no API calls
    if skip_api:
        return None

    # Rate limiting before API call
    time.sleep(MIN_DELAY)

    try:
        encoded_name = requests.utils.quote(name, safe="")
        url = PUBCHEM_URL.format(name=encoded_name)
        resp = requests.get(url, timeout=10)

        if resp.status_code == 200:
            data = resp.json()
            props = data["PropertyTable"]["Properties"][0]
            result = {
                "smiles": props.get("IsomericSMILES", ""),
                "inchi": props.get("InChI", ""),
            }
            cache[name] = result
            return result

        elif resp.status_code == 404:
            # Name not found in PubChem -- cache the negative result
            cache[name] = None
            return None

        else:
            # Unexpected HTTP status -- do NOT cache (may be transient)
            return None

    except (requests.RequestException, KeyError, IndexError, ValueError):
        # Network error, JSON parse error, etc. -- do NOT cache (transient)
        return None
