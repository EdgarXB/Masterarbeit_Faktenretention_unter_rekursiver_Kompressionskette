"""CSV, hashes and output protection without experiment imports."""
from pathlib import Path
import csv
import gzip
import hashlib
from faktenretention.paths import PROJECT_ROOT as PACKAGE

def read_csv(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path,'rt',encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))

def write_csv(path, rows, fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:
        fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def safe_output(path):
    path=Path(path).resolve()
    if path == PACKAGE or any(path == PACKAGE/name or (PACKAGE/name) in path.parents for name in ['data','prompts','src','notebooks','reference','archive','tests','docs','.git']):
        raise ValueError('Ausgaben dürfen eingefrorene Daten, Code oder Prompts nicht überschreiben.')
    return path
