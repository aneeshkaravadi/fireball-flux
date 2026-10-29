"""Download the whole CNEOS fireball table and save a dated CSV snapshot in data/.

    python examples/fetch_data.py

The analysis reads the committed snapshot, so results don't change when NASA adds events.
Run this to refresh it, then point SNAPSHOT in make_figures.py at the new file.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from fireballs import data

ROOT = Path(__file__).resolve().parents[1]


def main():
    payload = data.fetch()
    today = dt.date.today().isoformat()
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"fireball_api_{today}.json").write_text(json.dumps(payload))
    raw = data.parse_api(payload)
    out = ROOT / "data" / f"cneos_fireballs_{today}.csv"
    data.write_snapshot(raw, out)
    print(f"{len(raw)} fireballs -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
