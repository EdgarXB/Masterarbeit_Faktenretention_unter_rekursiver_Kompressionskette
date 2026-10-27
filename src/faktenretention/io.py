"""CSV, hashes and output protection without experiment imports."""
from pathlib import Path
import csv
import gzip
import hashlib
import os
import json
from faktenretention.paths import PROJECT_ROOT as PACKAGE

def read_csv(path, *, missing_ok=False):
    path = Path(path)
    if missing_ok and not path.exists():
        return []
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path,'rt',encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))

def write_csv(path, rows, fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:
        fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def append_csv(path, row, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + '\n')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def relative_path(path, root):
    """Provenance paths also work when inputs and outputs have separate parents."""
    return os.path.relpath(path, root)

def safe_output(path):
    path=Path(path).resolve()
    if path == PACKAGE or any(path == PACKAGE/name or (PACKAGE/name) in path.parents for name in ['data','prompts','src','notebooks','reference','archive','tests','docs','.git']):
        raise ValueError('Ausgaben dürfen eingefrorene Daten, Code oder Prompts nicht überschreiben.')
    return path


def read_table(path):
    """The round-trip CSV parser used by the detailed historical comparisons."""
    import pandas as pd
    return pd.read_csv(path, keep_default_na=False, low_memory=False, float_precision="round_trip")
