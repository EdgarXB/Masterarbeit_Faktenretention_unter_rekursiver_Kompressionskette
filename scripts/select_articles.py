#!/usr/bin/env python3
"""Zufällige, nachvollziehbare CNN/DailyMail-Stichproben außerhalb des Repos erzeugen."""
import argparse
import csv
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from faktenretention.text_rules import count_words

DATASET = 'abisee/cnn_dailymail'
REVISION = '96df5e686bee6baa90b8bee7c28b81fa3fa6223d'
SHARDS = {'train': 3, 'validation': 1, 'test': 1}


def select(rows, count, seed, excluded):
    """Reservoir sampling over valid unique articles; stable for a fixed input order."""
    if count < 1:
        raise ValueError('Die Anzahl muss positiv sein.')
    rng = random.Random(seed)
    selected, seen, eligible = [], set(excluded), 0
    for index, row in enumerate(rows):
        article = (row.get('article') or '').strip()
        highlights = (row.get('highlights') or '').strip()
        digest = hashlib.sha256(article.encode('utf-8')).hexdigest()
        if not 795 <= count_words(article) <= 805 or not highlights or digest in seen:
            continue
        seen.add(digest)
        eligible += 1
        item = {'dataset_index': index, 'article': article, 'highlights': highlights,
                'article_id': row.get('id', ''), 'dataset': DATASET, 'config': '3.0.0'}
        if len(selected) < count:
            selected.append(item)
        else:
            slot = rng.randrange(eligible)
            if slot < count:
                selected[slot] = item
    if len(selected) != count:
        raise ValueError(f'Nur {eligible} geeignete Artikel verfügbar; {count} angefordert.')
    return sorted(selected, key=lambda x: x['dataset_index']), eligible


def frozen_hashes():
    manifest = json.loads((ROOT / 'prompts/bank_manifest.json').read_text())
    hashes = set(manifest['source_article_hashes'])
    with gzip.open(ROOT / 'data/artikel.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            if row.get('article_sha256'):
                hashes.add(row['article_sha256'])
    return hashes


def parquet_rows(split, cache):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    total = SHARDS[split]
    for index in range(total):
        filename = f'3.0.0/{split}-{index:05d}-of-{total:05d}.parquet'
        path = hf_hub_download(repo_id=DATASET, repo_type='dataset', revision=REVISION,
                               filename=filename, cache_dir=str(cache))
        for batch in pq.ParquetFile(path).iter_batches(columns=['article', 'highlights', 'id'], batch_size=512):
            yield from batch.to_pylist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', required=True, choices=SHARDS)
    parser.add_argument('--count', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cache', type=Path, default=Path('results/model-resources/hf-cache'))
    parser.add_argument('--exclude', action='append', type=Path, default=[], help='Zusätzliche frühere Auswahl-CSV; wiederholbar')
    args = parser.parse_args()
    output = args.output.resolve()
    # Avoid placing copyrighted input into tracked or unignored repository paths.
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT / 'results/runs'):
        parser.error('Ausgabe außerhalb des Repositories oder unter results/runs/ ablegen.')
    receipt = output.with_suffix(output.suffix + '.manifest.json')
    if output.exists() or receipt.exists():
        parser.error('Ausgabe oder Manifest existiert bereits. Neuen Namen wählen.')
    excluded = frozen_hashes()
    for path in args.exclude:
        with path.open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                article = (row.get('article') or '').strip()
                digest = row.get('article_sha256') or (hashlib.sha256(article.encode()).hexdigest() if article else '')
                if not digest:
                    parser.error(f'Keine Artikel oder Hashes in {path}')
                excluded.add(digest)
    selected, eligible = select(parquet_rows(args.split, args.cache), args.count, args.seed, excluded)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0])); writer.writeheader(); writer.writerows(selected)
    metadata = {'dataset': DATASET, 'config': '3.0.0', 'revision': REVISION, 'split': args.split,
                'seed': args.seed, 'eligible_count': eligible, 'count': len(selected),
                'selected_indices': [r['dataset_index'] for r in selected],
                'article_sha256': [hashlib.sha256(r['article'].encode()).hexdigest() for r in selected],
                'csv_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'excluded_hash_count': len(excluded), 'rule': '795..805 words, nonempty highlights, unique SHA-256'}
    receipt.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(f'{len(selected)} Artikel aus {args.split} ausgewählt: {output}\nManifest: {receipt}')


if __name__ == '__main__':
    main()
