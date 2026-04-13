"""
VSAC Bulk Value Set Downloader
================================
Downloads all publicly available value sets from the NLM VSAC FHIR API.

Requirements:
    pip install requests tqdm

Usage:
    export UMLS_API_KEY="your-api-key-here"
    python create_dataset/download_vsac.py

    # Or pass directly:
    python create_dataset/download_vsac.py --api-key YOUR_KEY

    # Resume an interrupted download:
    python create_dataset/download_vsac.py  # automatically resumes from checkpoint

Output structure:
    vsac_data/
        metadata/          # ValueSet resources (name, description, OID, steward, etc.)
            <oid>.json
        expansions/        # Full code expansions
            <oid>.json
        index.jsonl        # One record per value set: oid, title, steward, code_system, code_count
        failed_oids.txt    # OIDs that failed after retries (for manual inspection)
        checkpoint.txt     # Last successfully processed OID index (for resume)

Notes:
    - VSAC rate limit: 20 requests/sec. This script targets ~10 req/sec to be safe.
    - With ~18,000 value sets and 2 API calls each, expect ~1 hour to complete.
    - Run once and cache; don't re-run unless you want fresh data.
    - Requires a free UMLS license: https://uts.nlm.nih.gov/license.html
"""

import logging
import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
FHIR_BASE = "https://cts.nlm.nih.gov/fhir"
OUTPUT_DIR = REPO_ROOT / "vsac_data"
RATE_LIMIT_DELAY = 0.12        # seconds between requests (~8 req/sec, safe under 20/sec limit)
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 5              # seconds between retries
PAGE_SIZE = 100                # value sets per search page

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("vsac_download.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ── API helpers ───────────────────────────────────────────────────────────────

def make_session(api_key: str) -> requests.Session:
    """Create a session with UMLS basic auth (password = API key)."""
    session = requests.Session()
    session.auth = ("apikey", api_key)
    session.headers.update({"Accept": "application/json"})
    return session


def get_with_retry(session: requests.Session, url: str, params: dict = None) -> Optional[dict]:
    """GET with exponential retry on failure."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 404:
                log.warning(f"404 Not Found: {url}")
                return None
            elif resp.status_code == 429:
                wait = RETRY_BACKOFF * (attempt + 1)
                log.warning(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                log.warning(f"HTTP {resp.status_code} for {url} (attempt {attempt+1})")
                time.sleep(RETRY_BACKOFF)
        except requests.RequestException as e:
            log.warning(f"Request error on attempt {attempt+1}: {e}")
            time.sleep(RETRY_BACKOFF * (attempt + 1))
    return None


# ── Discovery: enumerate all value set OIDs ───────────────────────────────────

def discover_all_oids(session: requests.Session) -> list[dict]:
    """
    Page through the VSAC FHIR ValueSet search endpoint to collect all
    value set stubs (oid, title, publisher/steward, status, date).
    Returns list of dicts.
    """
    log.info("Discovering all value set OIDs via FHIR search...")
    stubs = []
    url = f"{FHIR_BASE}/ValueSet"
    params = {"_count": PAGE_SIZE, "_elements": "id,title,name,publisher,status,date,description"}
    page = 0

    while url:
        data = get_with_retry(session, url, params=params)
        if not data:
            log.error("Failed to fetch value set list page. Stopping discovery.")
            break

        total = data.get("total", "?")
        entries = data.get("entry", [])
        page += 1

        for entry in entries:
            resource = entry.get("resource", {})
            oid = resource.get("id", "")
            if oid:
                stubs.append({
                    "oid": oid,
                    "title": resource.get("title", resource.get("name", "")),
                    "publisher": resource.get("publisher", ""),
                    "status": resource.get("status", ""),
                    "date": resource.get("date", ""),
                    "description": resource.get("description", ""),
                })

        log.info(f"  Page {page}: {len(entries)} value sets fetched ({len(stubs)}/{total} total)")

        # Follow FHIR pagination via Bundle.link[relation=next]
        url = None
        params = None  # subsequent pages use the full next URL
        for link in data.get("link", []):
            if link.get("relation") == "next":
                url = link["url"]
                break

        time.sleep(RATE_LIMIT_DELAY)

    log.info(f"Discovery complete: {len(stubs)} value sets found.")
    return stubs


# ── Expansion fetcher ─────────────────────────────────────────────────────────

def fetch_expansion(session: requests.Session, oid: str) -> Optional[dict]:
    """
    Fetch full code expansion for a single value set OID.
    Returns the FHIR ValueSet resource with expansion, or None on failure.
    """
    url = f"{FHIR_BASE}/ValueSet/{oid}/$expand"
    data = get_with_retry(session, url)
    time.sleep(RATE_LIMIT_DELAY)
    return data


def parse_expansion(expansion_resource: dict) -> dict:
    """Extract key fields from a FHIR ValueSet expansion resource."""
    expansion = expansion_resource.get("expansion", {})
    contains = expansion.get("contains", [])

    codes = []
    code_systems = set()
    for concept in contains:
        codes.append({
            "system": concept.get("system", ""),
            "code": concept.get("code", ""),
            "display": concept.get("display", ""),
            "inactive": concept.get("inactive", False),
        })
        if concept.get("system"):
            code_systems.add(concept["system"])

    return {
        "total": expansion.get("total", len(codes)),
        "code_systems": sorted(code_systems),
        "codes": codes,
    }


# ── Main download loop ────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: Path) -> int:
    """Return the index of the last successfully processed value set."""
    if checkpoint_path.exists():
        try:
            return int(checkpoint_path.read_text().strip())
        except ValueError:
            pass
    return 0


def save_checkpoint(checkpoint_path: Path, idx: int):
    checkpoint_path.write_text(str(idx))


def download_all(api_key: str):
    # Setup directories
    meta_dir = OUTPUT_DIR / "metadata"
    exp_dir = OUTPUT_DIR / "expansions"
    meta_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = OUTPUT_DIR / "checkpoint.txt"
    failed_path = OUTPUT_DIR / "failed_oids.txt"
    index_path = OUTPUT_DIR / "index.jsonl"

    session = make_session(api_key)

    # Step 1: discover OIDs (or load cached discovery)
    discovery_cache = OUTPUT_DIR / "discovery.json"
    if discovery_cache.exists():
        log.info("Loading cached discovery results...")
        with open(discovery_cache) as f:
            stubs = json.load(f)
    else:
        stubs = discover_all_oids(session)
        with open(discovery_cache, "w") as f:
            json.dump(stubs, f, indent=2)
        log.info(f"Discovery cached to {discovery_cache}")

    # Step 2: resume from checkpoint
    start_idx = load_checkpoint(checkpoint_path)
    if start_idx > 0:
        log.info(f"Resuming from index {start_idx} / {len(stubs)}")

    failed_oids = []
    indexed_oids = set()
    if index_path.exists():
        with open(index_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    indexed_oids.add(json.loads(line)["oid"])
                except (json.JSONDecodeError, KeyError):
                    log.warning("Skipping malformed index row while building resume state.")

    # Step 3: download expansions
    with open(index_path, "a") as index_f:
        for idx, stub in enumerate(tqdm(stubs, desc="Downloading value sets")):
            if idx < start_idx:
                continue

            oid = stub["oid"]
            exp_file = exp_dir / f"{oid}.json"
            meta_file = meta_dir / f"{oid}.json"

            # Skip if already downloaded
            if exp_file.exists() and meta_file.exists():
                save_checkpoint(checkpoint_path, idx + 1)
                continue

            # Save stub metadata
            with open(meta_file, "w") as f:
                json.dump(stub, f, indent=2)

            # Fetch expansion
            expansion_resource = fetch_expansion(session, oid)
            if expansion_resource is None:
                log.warning(f"Failed to expand {oid} ({stub.get('title', '')})")
                failed_oids.append(oid)
                with open(failed_path, "a") as f:
                    f.write(oid + "\n")
                save_checkpoint(checkpoint_path, idx + 1)
                continue

            # Parse and save expansion
            parsed = parse_expansion(expansion_resource)
            with open(exp_file, "w") as f:
                json.dump(expansion_resource, f)  # save full resource

            # Write to index
            index_record = {
                "oid": oid,
                "title": stub.get("title", ""),
                "publisher": stub.get("publisher", ""),
                "status": stub.get("status", ""),
                "date": stub.get("date", ""),
                "description": stub.get("description", ""),
                "code_count": parsed["total"],
                "code_systems": parsed["code_systems"],
            }
            if oid not in indexed_oids:
                index_f.write(json.dumps(index_record) + "\n")
                index_f.flush()
                indexed_oids.add(oid)

            save_checkpoint(checkpoint_path, idx + 1)

    log.info(f"Download complete. {len(stubs) - len(failed_oids)} succeeded, {len(failed_oids)} failed.")
    if failed_oids:
        log.info(f"Failed OIDs written to {failed_path}. Re-run to retry them.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bulk download VSAC value sets via FHIR API")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("UMLS_API_KEY"),
        help="UMLS API key (or set UMLS_API_KEY env variable)"
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "vsac_data"),
        help="Directory to write output (default: vsac_data/)"
    )
    args = parser.parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = Path(args.output_dir)

    if not args.api_key:
        parser.error(
            "No API key provided. Set UMLS_API_KEY environment variable or use --api-key.\n"
            "Get a free key at: https://uts.nlm.nih.gov/license.html"
        )

    download_all(args.api_key)


if __name__ == "__main__":
    main()
