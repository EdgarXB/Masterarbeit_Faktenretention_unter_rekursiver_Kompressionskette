#!/usr/bin/env python3
"""Historisch eingefrorene Gewichte explizit herunterladen und lokal einrichten."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from faktenretention.metric_resources import REVISIONS, HEAD_SHA256

REPOS = {'embedding': 'sentence-transformers/all-MiniLM-L6-v2', 'bertscore': 'distilroberta-base',
         'summac_nli': 'tals/albert-xlarge-vitaminc-mnli'}
WEIGHTS = {
 '2d9f3c32a61f2668e012708be7ec04e753e35e366f87b9e4bce89f4d073cfc16': 'https://registry.ollama.ai/v2/milkey/Seed-OSS-36B-Instruct/blobs/sha256:2d9f3c32a61f2668e012708be7ec04e753e35e366f87b9e4bce89f4d073cfc16',
 '280af6832eca23cb322c4dcc65edfea98a21b8f8ab07dc7553bd6f7e6e7a3313': 'https://registry.ollama.ai/v2/library/gemma4/blobs/sha256:280af6832eca23cb322c4dcc65edfea98a21b8f8ab07dc7553bd6f7e6e7a3313',
 '31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34': 'https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/0669b98607d47046c7c2b3f801011d54a08cfccf/Qwen3.8-27B-Q4_K_M.gguf',
 '2e968a6af97ce35d8971890b257b9b7edabf20ad91450501fa53162a19ee33eb': 'https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/0669b98607d47046c7c2b3f801011d54a08cfccf/mmproj-Qwen3.8-27B-Q8_0.gguf',
}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, target, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != expected:
            raise ValueError(f'Vorhandene Datei hat einen falschen Hash: {target}')
        print('Bereits geprüft:', target, flush=True)
        return
    temporary = target.with_name(target.name + '.part')
    print('Download:', url, flush=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
            shutil.copyfileobj(response, output, 1024 * 1024)
        if sha(temporary) != expected:
            raise ValueError(f'Falscher SHA-256 für {url}; temporäre Datei entfernt.')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def metrics(root):
    import nltk
    # Einige historische Revisionen haben keinen gültigen Xet-Token-Endpunkt;
    # der reguläre HTTPS-Download funktioniert weiterhin.
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    from huggingface_hub import snapshot_download
    root.mkdir(parents=True, exist_ok=True)
    resources = {}
    for role, repo in REPOS.items():
        relative = Path(role) / REVISIONS[role]
        print('Modell:', repo, REVISIONS[role], flush=True)
        snapshot_download(repo_id=repo, revision=REVISIONS[role], local_dir=root / relative,
                          allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '1_Pooling/*'])
        resources[role] = str(relative)
    head = 'summac_conv_vitc_sent_perc_e.bin'
    fetch('https://raw.githubusercontent.com/tingofurro/summac/master/' + head, root / head, HEAD_SHA256)
    if not nltk.download('punkt_tab', download_dir=str(root / 'nltk_data'), raise_on_error=True):
        raise RuntimeError('NLTK punkt_tab konnte nicht geladen werden.')
    resources.update(summac_head=head, nltk_data='nltk_data')
    target = root / 'resources.json'
    content = json.dumps(resources, indent=2) + '\n'
    if target.exists() and target.read_text() != content:
        raise ValueError(f'Abweichende Ressourcenkonfiguration existiert: {target}')
    target.write_text(content)
    print('Ressourcendatei:', target, flush=True)


def ollama(root):
    metadata = ROOT / 'data/model_metadata'
    store = root / 'ollama'
    for digest, url in WEIGHTS.items():
        fetch(url, store / 'blobs' / ('sha256-' + digest), digest)
    models = json.loads((ROOT / 'data/models.json').read_text())
    copies = {}
    for model in models.values():
        name, tag = model['name'].split(':')
        source = metadata / 'manifests' / name / tag
        if sha(source) != model['digest']:
            raise ValueError(f'Manifest-Hash stimmt nicht: {source}')
        manifest = json.loads(source.read_text())
        for item in [manifest['config'], *manifest['layers']]:
            digest = item['digest'].split(':', 1)[1]
            filename = 'sha256-' + digest
            target = store / 'blobs' / filename
            original = metadata / 'blobs' / filename
            source_blob = original if original.is_file() else target
            if not source_blob.is_file() or source_blob.stat().st_size != item['size'] or sha(source_blob) != digest:
                raise ValueError(f'Blob fehlt oder stimmt nicht: {source_blob}')
            if source_blob != target:
                copies[target] = source_blob
        copies[store / 'manifests/registry.ollama.ai/library' / name / tag] = source
    for target, source in copies.items():
        if target.exists() and sha(target) != sha(source):
            raise ValueError(f'Abweichende Datei wird nicht überschrieben: {target}')
    for target, source in copies.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
    print('Drei historische Ollama-Manifeste und Blobs eingerichtet:', store, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/model-resources'))
    parser.add_argument('--metrics', action='store_true')
    parser.add_argument('--ollama', action='store_true')
    args = parser.parse_args()
    if not (args.metrics or args.ollama):
        parser.error('Mindestens --metrics oder --ollama angeben; Downloads erfolgen nur ausdrücklich.')
    root = args.root.resolve()
    if root.is_relative_to(ROOT) and not root.is_relative_to(ROOT / 'results/model-resources'):
        parser.error('Modellverzeichnis außerhalb des Repositories oder unter results/model-resources/ wählen.')
    if args.metrics:
        metrics(root)
    if args.ollama:
        ollama(root)


if __name__ == '__main__':
    main()
