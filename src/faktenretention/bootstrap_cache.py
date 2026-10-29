"""Prüfe den eingefrorenen Bootstrap ohne bytegleichen erneuten GLM-Fit."""
import hashlib
import json
from pathlib import Path

import numpy as np


FIT_ATOL = 1e-8


def load_bootstrap_cache(path, *, X, Y, beta, bootstrap_counts, worker_source, software):
    """Binde exakte Eingaben an den Originalcache; toleriere nur Fit-Rundung.

    Der Referenzfit stammt aus der Umgebung, in der der ursprüngliche
    Fingerprint bytegenau reproduziert wurde. Auch mit Referenzparametern
    müssen Design, Zähler, Ziehungen, Worker und Versionen exakt passen.
    """
    path = Path(path)
    advice = ('Dokumentierte Python-/Paketversionen und unveränderte Quellen verwenden '
              'oder RECOMPUTE_BOOTSTRAP ausdrücklich aktivieren.')

    def reject(reason):
        raise RuntimeError(f'Bootstrap-Cache passt nicht: {reason}. {advice}')

    manifest_path = path.with_suffix('.json')
    if not path.is_file() or not manifest_path.is_file():
        reject('Cache oder Referenzmanifest fehlt')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['schema_version'] != 1:
        reject('unbekannte Manifestversion')
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['cache_sha256']:
        reject('Cache-Datei verändert')
    if software != manifest['software']:
        differences = sorted(key for key in software.keys() | manifest['software'].keys()
                             if software.get(key) != manifest['software'].get(key))
        reject('abweichende Python-/Paketversionen: ' + ', '.join(differences))

    reference = np.asarray(manifest['reference_beta'], dtype=np.float64)
    beta = np.asarray(beta)
    if (beta.shape != reference.shape or not np.isfinite(beta).all()
            or not np.allclose(beta, reference, rtol=0, atol=FIT_ATOL)
            or not np.allclose(X @ beta, X @ reference, rtol=0, atol=FIT_ATOL)):
        reject('erneuter Modellfit weicht numerisch vom Referenzfit ab')

    # Derselbe historische Fingerprint, mit den gespeicherten Referenzparametern.
    fingerprint = hashlib.sha256(
        X.tobytes() + Y.tobytes() + reference.tobytes() + bootstrap_counts.tobytes()
        + worker_source.encode() + json.dumps(software, sort_keys=True).encode()
    ).hexdigest()
    if fingerprint != manifest['fingerprint']:
        reject('Designmatrix, Zähler, Ziehungen oder numerischer Worker verändert')
    with np.load(path, allow_pickle=False) as archive:
        if str(archive['fingerprint']) != fingerprint:
            reject('Cache-Fingerprint stimmt nicht mit dem Referenzmanifest überein')
        if not np.array_equal(archive['bootstrap_counts'], bootstrap_counts):
            reject('gespeicherte Ziehungsgewichte verändert')
        coefficients = archive['coefficients'].copy()
        records = json.loads(str(archive['records']))
    if coefficients.shape != (len(bootstrap_counts), X.shape[1]) or len(records) != len(bootstrap_counts):
        reject('unvollständige Bootstrap-Anpassungen')
    return coefficients, records, fingerprint
