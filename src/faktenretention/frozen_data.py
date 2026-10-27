# Zweck: Vollständigkeit und Prüfsummen des eingefrorenen Datenbestands prüfen.
# Eingaben: manifest.json, data/*.csv(.gz), data/eingefrorene_quellen.zip.
# Ausgaben: results/datenpruefung.json; optional entpackte Arbeitskopie.
# LLM-Aufrufe: keine. Beispiel: python 01_prepare.py
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
from faktenretention.io import read_csv as read_csv_path, safe_output, sha

from faktenretention.paths import PROJECT_ROOT as ROOT


def read_csv(name):
    return read_csv_path(ROOT / 'data' / name)


def baseline_bytes(name):
    """Read preserved evidence without making historical code importable."""
    with zipfile.ZipFile(ROOT / 'archive/baseline_ad7eb1a.zip') as archive:
        return archive.read(name)


def source_bytes(entry, archive, tables, notebooks):
    if 'package_file' in entry:
        path = ROOT / entry['package_file']
        if not path.is_file():
            return baseline_bytes(entry['package_file'])
        return path.read_bytes()
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


def verify_frozen_checksums():
    """Keep original checksum bytes and resolve deliberately archived code paths."""
    checked = []
    for line in (ROOT/'data/frozen_assets.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = ROOT/name
        content = path.read_bytes() if path.is_file() else baseline_bytes(name)
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError('Verändertes eingefrorenes Artefakt: '+name)
        checked.append(name)
    return checked


# The only translation from the historical project tree to active notebook paths.
_ANALYSIS_FOLDERS = {
    'human_validation_grafiken': '01_human_faktenrelevanz',
    'human_validation_qa_grafiken': '02_human_qa_retention',
    'hauptexperiment_qa_ergebnisse/aktuell_20260930': '03_qa_retention',
    'hauptexperiment_laengen_ergebnisse/aktuell_20260930': '04_laenge_und_adjustierte_retention',
    'hauptexperiment_verlust_ergebnisse/aktuell_20260930': '05_verlustmuster',
    'hauptexperiment_metriken_ergebnisse/aktuell_20260930': '06_weitere_metriken',
    'hauptexperiment_laengenmatched_ergebnisse/aktuell_20260930': '07_laengenmatched_matrizen',
    'hauptexperiment_texttabellen_ergebnisse/aktuell_20260930': '08_tabellenexport',
}
_SOURCE_EVIDENCE = {
    'README_START.md', 'six_method_pilot_common.py', 'run_six_method_final_100.py',
    'compute_length_adjusted_retention.py', 'iterative_summarize.py',
    'hauptexperiment_01_qa_faktenretention.quellnotebook.json',
    'hauptexperiment_02_laengeneinhaltung.quellnotebook.json',
    'hauptexperiment_03_verlustmuster.quellnotebook.json',
    'build_hauptexperiment_06_laengenmatched_matrizen.py',
}


def prepared_path(name):
    """Return a simple relative input path, or None for unused historical files."""
    prefixes = {
        'project/02_Auswertung/hauptexperiment_qa_daten/': 'qa/',
        'project/02_Auswertung/hauptexperiment_metriken_daten/': 'metrics/',
        'project/02_Auswertung/human_validation_qa_daten/': 'human_qa/',
        'supplement/daten/llm_vergleich/': 'llm/',
    }
    for prefix, target in prefixes.items():
        if name.startswith(prefix):
            return target + name.removeprefix(prefix)
    if name == 'project/02_Auswertung/human_validation_faktenrelevanz_ergebnisse.csv':
        return 'human_faktenrelevanz.csv'
    if Path(name).name in _SOURCE_EVIDENCE:
        return 'sources/' + Path(name).name
    for folder, analysis in _ANALYSIS_FOLDERS.items():
        prefix = 'project/02_Auswertung/' + folder + '/'
        if name.startswith(prefix):
            return 'tables/' + analysis + '/' + name.removeprefix(prefix)
    return None


def prepare_notebook_data(destination=None, *, include_tables=True):
    """Prepare verified inputs once; no kernels, experiments or model resources.

    Tables are historical predecessor inputs for direct Run All. Automated runs
    omit them and use only their newly computed, verified predecessors instead.
    """
    destination = Path(destination or ROOT / 'data/prepared').expanduser().resolve()
    # This is the one allowed generated data directory inside the frozen tree.
    if destination != ROOT / 'data/prepared':
        safe_output(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Datenordner muss neu oder leer sein: ' + str(destination))
    from faktenretention.notebook_runner import load_analyses
    specs = load_analyses()
    reused = {item['reference']: item['path'] for spec in specs for item in spec['reused']}
    cache_names = {Path(name).name: name for name in reused.values()}
    inventory = json.loads((ROOT / 'data/quelleninventar.json').read_text())['files']
    destination.mkdir(parents=True, exist_ok=True)
    manifest_paths = []
    with zipfile.ZipFile(ROOT / 'data/eingefrorene_quellen.zip') as sources, \
         zipfile.ZipFile(ROOT / 'results/originaltabellen.zip') as tables, \
         zipfile.ZipFile(ROOT / 'data/referenznotebooks.zip') as notebooks:
        for entry in inventory:
            name = prepared_path(entry['archive_path'])
            if entry.get('reference_member') in reused:
                name = reused[entry['reference_member']]
            elif entry['role'] == 'reference_result' and 'reference_member' not in entry:
                name = cache_names.get(Path(entry['archive_path']).name, name)
            if name is None or (not include_tables and name.startswith('tables/')):
                continue
            content = source_bytes(entry, sources, tables, notebooks)
            if hashlib.sha256(content).hexdigest() != entry['sha256']:
                raise ValueError('Veränderte Quelle: ' + entry['archive_path'])
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            if name.startswith('tables/') and path.name == 'execution_manifest.json':
                manifest_paths.append(path)
    # Only derived path metadata is translated. Original manifests remain in ZIPs.
    # Update in analysis order so hashes of predecessor manifests remain valid.
    for path in sorted(manifest_paths):
        obj = json.loads(path.read_text())
        for entry in obj.get('source_hashes', []):
            name = prepared_path('project/02_Auswertung/' + entry['file'])
            if name is None:
                raise ValueError('Unbekannter Herkunftspfad: ' + entry['file'])
            entry['file'] = name
            entry['sha256'] = sha(destination / name)
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prüfsummen und historischen Bestand offline prüfen.')
    parser.add_argument('--checksums', action='store_true')
    args = parser.parse_args()
    if args.checksums:
        print(str(len(verify_frozen_checksums()))+' unveränderte Prüfsummen bestätigt.')
    else:
        verify_data()
