"""Scrape Transfermarkt market-value history for specific player IDs.

Usage:
    python transfermrkt/scrape_comparison_players.py 451338 621033
    python transfermrkt/scrape_comparison_players.py 451338 621033 -o transfermrkt/comparison_players.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://www.transfermarkt.com/ceapi/marketValueDevelopment/graph/{player_id}"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def name_from_details_url(details_url: str, player_id: str) -> str:
    """Derive a lowercase display name from Transfermarkt's details_url slug."""
    if details_url:
        slug = details_url.strip("/").split("/")[0]
        name = slug.replace("-", " ").strip()
        if name:
            return name
    return f"player {player_id}"


def fetch_market_value(player_id: str, timeout: int = 30) -> dict:
    url = API_URL.format(player_id=player_id)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} for player_id={player_id}")
        return json.loads(response.read().decode())


def scrape_players(player_ids: list[str], pause_s: float = 0.5) -> dict:
    results = {}
    for i, player_id in enumerate(player_ids):
        player_id = str(player_id).strip()
        if not player_id:
            continue
        if not re.fullmatch(r"\d+", player_id):
            raise ValueError(f"Invalid player_id (expected digits): {player_id!r}")

        print(f"Fetching player_id={player_id}...", flush=True)
        data = fetch_market_value(player_id)
        history = data.get("list") or []
        if not history:
            raise RuntimeError(f"No market-value history returned for player_id={player_id}")

        name = name_from_details_url(data.get("details_url", ""), player_id)
        team = history[-1].get("verein") or "Unknown Club"

        results[name] = {
            "player_id": player_id,
            "team": team,
            "market_value_data": {
                "marketValueDevelopment": data,
            },
        }
        print(
            f"  -> {name.title()} | team={team} | points={len(history)} "
            f"| current={data.get('current', '?')}",
            flush=True,
        )

        if i < len(player_ids) - 1 and pause_s > 0:
            time.sleep(pause_s)

    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Transfermarkt market values for given player IDs."
    )
    parser.add_argument(
        "player_ids",
        nargs="+",
        help="One or more Transfermarkt player IDs (e.g. 451338 621033)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="transfermrkt/comparison_players.json",
        help="Output JSON path (default: transfermrkt/comparison_players.json)",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.5,
        help="Seconds to wait between requests (default: 0.5)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        results = scrape_players(args.player_ids, pause_s=args.pause)
    except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: scrape failed: {exc}", file=sys.stderr)
        return 1

    if not results:
        print("ERROR: no players scraped", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {len(results)} player(s) -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
