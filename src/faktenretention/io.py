"""CSV, hashes and output protection without experiment imports."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = PROJECT_ROOT

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


def safe_output(path):
    path=Path(path).resolve()
    if path == PACKAGE or any(path == PACKAGE/name or (PACKAGE/name) in path.parents for name in ['data','prompts','src','notebooks','tests','.git']):
        raise ValueError('Ausgaben dürfen eingefrorene Daten, Code oder Prompts nicht überschreiben.')
    return path


def validate_article_keys(rows, *, require_article_ids=True):
    seen_indices = set()
    seen_ids = set()
    for position, row in enumerate(rows, 1):
        try:
            index = int(str(row['dataset_index']))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'Zeile {position}: dataset_index muss eine nichtnegative Ganzzahl sein.') from error
        if index < 0:
            raise ValueError(f'Zeile {position}: dataset_index muss nichtnegativ sein.')
        key = (str(row.get('split', '')), index)
        if key in seen_indices:
            raise ValueError(f'Doppelter dataset_index {index} im Split {key[0]!r}. Keine Generierung gestartet.')
        seen_indices.add(key)
        if require_article_ids:
            article_id = str(row.get('article_id', '')).strip()
            if not article_id or article_id in seen_ids:
                raise ValueError(f'Leere oder doppelte article_id in Zeile {position}. Keine Generierung gestartet.')
            seen_ids.add(article_id)


def validate_candidate_keys(rows):
    seen = set()
    seen_sources = set()
    source_articles = {}
    for row in rows:
        candidate_id = str(row.get('candidate_id', ''))
        if not candidate_id or candidate_id in seen:
            raise ValueError('Leere oder doppelte candidate_id. Keine Modellauswertung gestartet.')
        seen.add(candidate_id)
        if all(key in row for key in ['split', 'dataset_index', 'target_words']):
            source = (str(row['split']), int(row['dataset_index']))
            stage_key = (*source, int(row['target_words']))
            if stage_key in seen_sources:
                raise ValueError('Doppelter Artikel-/Stufenschlüssel in Demonstrationskandidaten.')
            seen_sources.add(stage_key)
            article_hash = row.get('article_sha256')
            if article_hash:
                if source in source_articles and source_articles[source] != article_hash:
                    raise ValueError('Ein dataset_index verweist auf unterschiedliche Kandidatenartikel.')
                source_articles[source] = article_hash
