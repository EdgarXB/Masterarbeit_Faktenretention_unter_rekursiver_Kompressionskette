# Zweck: Vollständigkeit und Prüfsummen des eingefrorenen Datenbestands prüfen.
# Eingaben: manifest.json, data/*.csv(.gz), data/eingefrorene_quellen.zip.
# Ausgaben: results/datenpruefung.json; optional entpackte Arbeitskopie.
# LLM-Aufrufe: keine. Beispiel: python 01_prepare.py
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def read_csv(name):
    path = ROOT / 'data' / name
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def source_bytes(entry, archive, tables, notebooks):
    if 'package_file' in entry:
        return (ROOT / entry['package_file']).read_bytes()
    if 'reference_member' in entry:
        return tables.read(entry['reference_member'])
    if 'notebook_member' in entry:
        return notebooks.read(entry['notebook_member'])
    return archive.read(entry['archive_path'])


def verify_data():
    manifest_path = ROOT / 'manifest.json'
    if manifest_path.is_file():
        package = json.loads(manifest_path.read_text())
        for entry in package['files']:
            if entry.get('immutable'):
                path = ROOT / entry['file']
                if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                    raise ValueError('Veränderte Paketdatei: '+entry['file'])
    inventory = json.loads((ROOT / 'data/quelleninventar.json').read_text())
    with zipfile.ZipFile(ROOT / 'data/eingefrorene_quellen.zip') as archive, \
         zipfile.ZipFile(ROOT / 'results/originaltabellen.zip') as tables, \
         zipfile.ZipFile(ROOT / 'data/referenznotebooks.zip') as notebooks:
        for entry in inventory['files']:
            content = source_bytes(entry, archive, tables, notebooks)
            if hashlib.sha256(content).hexdigest() != entry['sha256']:
                raise ValueError('Veränderte Quelle: ' + entry['archive_path'])
    articles = read_csv('artikel.csv.gz')
    facts = read_csv('fakten.csv.gz')
    summaries = read_csv('zusammenfassungen.csv.gz')
    answers = read_csv('qa_antworten.csv.gz')
    metrics = read_csv('modellmetriken.csv')
    assert len(articles) == 100 and len({x['article_id'] for x in articles}) == 100
    assert len(facts) == 2000 and len({x['canonical_fact_id'] for x in facts}) == 2000
    assert len(summaries) == 600 and len(metrics) == 2400 and len(answers) == 48000
    key = ['article_id', 'summary_prompt_variant', 'stage_target_words', 'canonical_fact_id']
    assert len({tuple(x[k] for k in key) for x in answers}) == 48000
    assert sum(bool(x['error']) for x in answers) == 196
    empty = sum(not x[f'summary_{t}'].strip() for x in summaries for t in [500, 300, 150, 75])
    assert empty == 2
    for row in articles:
        assert hashlib.sha256(row['article'].encode()).hexdigest() == row['article_sha256']
    report = dict(articles=100, facts=2000, summaries=2400, qa_positions=48000,
                  technical_missing=196, empty_outputs=empty, checked_source_files=len(inventory['files']),
                  llm_generation_calls=0)
    (ROOT / 'results/datenpruefung.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def extract_working_copy(destination):
    destination.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((ROOT / 'data/quelleninventar.json').read_text())
    with zipfile.ZipFile(ROOT / 'data/eingefrorene_quellen.zip') as archive, \
         zipfile.ZipFile(ROOT / 'results/originaltabellen.zip') as tables, \
         zipfile.ZipFile(ROOT / 'data/referenznotebooks.zip') as notebooks:
        for entry in inventory['files']:
            path = destination / entry['archive_path']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source_bytes(entry, archive, tables, notebooks))


