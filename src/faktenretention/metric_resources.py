"""Explizit übergebene lokale Metrikgewichte, niemals implizite Downloads."""
from pathlib import Path
import hashlib
import json
import os

RESOURCES = {}
REVISIONS = dict(embedding='c9745ed1d9f207416be6d2e6f8de32d1f16199bf',
                 bertscore='fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b',
                 summac_nli='3082ba54344bd9ddada2be1c5e9b4131721d2a5d')
HEAD_SHA256 = 'cd880581f59bd47b8968c7f48d754cd617dfe9b256d8d76fdcbc0fa4a3b3c6fe'


def configure(resources):
    global RESOURCES
    RESOURCES = dict(resources)
    # Diese beiden dokumentierten Bibliotheksschalter unterbinden HF-Downloads.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'


def local_model(role):
    value = RESOURCES.get(role)
    if not value:
        raise ValueError('Lokaler Ressourcenpfad fehlt: '+role)
    path = Path(value).resolve()
    if not path.is_dir() or not (path/'config.json').is_file():
        raise ValueError('Lokales Modellverzeichnis fehlt oder ist unvollständig: '+role)
    revision = path.name
    if (path/'revision.txt').is_file():
        revision = (path/'revision.txt').read_text().strip()
    if revision != REVISIONS[role]:
        raise ValueError('Nicht die dokumentierte Revision für '+role)
    return path


def summac_resources():
    nli = local_model('summac_nli')
    head = Path(RESOURCES.get('summac_head','')).resolve()
    if not head.is_file() or hashlib.sha256(head.read_bytes()).hexdigest() != HEAD_SHA256:
        raise ValueError('Lokaler trainierter SummaC-Checkpoint fehlt oder hat einen anderen Hash.')
    tokenizer_root=RESOURCES.get('nltk_data')
    if not tokenizer_root or not (Path(tokenizer_root)/'tokenizers/punkt_tab/english').is_dir():
        raise ValueError('Lokale englische NLTK-punkt_tab-Dateien fehlen: nltk_data.')
    return nli, head


def configure_summac(nli_path, head):
    from summac import model_summac
    # Nur der Speicherort ändert sich; Klassenindizes und Architektur bleiben gleich.
    model_summac.model_map['vitc']['model_card'] = str(nli_path)
    nltk_data = RESOURCES.get('nltk_data')
    if not nltk_data or not Path(nltk_data).is_dir():
        raise ValueError('Lokale NLTK-Tokenizerdaten explizit in resources.json angeben.')
    import nltk
    nltk.data.path = [str(Path(nltk_data).resolve())]
