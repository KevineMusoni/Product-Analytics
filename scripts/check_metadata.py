"""Check that data/datapackage.json still describes the files in data/raw/.

Verifies, for every resource: the file exists, its size and SHA-256 checksum match, and, for tabular
resources, that the header matches the documented field names and every documented constraint holds.
Exit status is non-zero on any mismatch, so CI fails when data and documentation drift apart.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


def main() -> int:
    package = json.loads((DATA / "datapackage.json").read_text(encoding="utf-8"))
    problems: list[str] = []
    for res in package["resources"]:
        path = DATA / res["path"]
        if not path.exists():
            problems.append(f"{res['name']}: missing file {path}")
            continue
        raw = path.read_bytes()
        if len(raw) != res["bytes"]:
            problems.append(f"{res['name']}: size {len(raw)} != documented {res['bytes']}")
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        if digest != res["hash"]:
            problems.append(f"{res['name']}: checksum mismatch")
        schema = res.get("schema")
        if not schema:
            continue
        frame = pd.read_csv(path)
        documented = [f["name"] for f in schema["fields"]]
        if list(frame.columns) != documented:
            problems.append(f"{res['name']}: columns {list(frame.columns)} != documented {documented}")
            continue
        key = schema.get("primaryKey")
        if key and not frame[key].is_unique:
            problems.append(f"{res['name']}: primary key {key} is not unique")
        for field in schema["fields"]:
            minimum = field.get("constraints", {}).get("minimum")
            if minimum is not None and frame[field["name"]].min() < minimum:
                problems.append(f"{res['name']}.{field['name']}: below minimum {minimum}")
            allowed = field.get("constraints", {}).get("enum")
            if allowed is not None and not set(frame[field["name"]].unique()) <= set(allowed):
                problems.append(f"{res['name']}.{field['name']}: values outside {allowed}")
    for problem in problems:
        print("FAIL", problem)
    print("metadata OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
