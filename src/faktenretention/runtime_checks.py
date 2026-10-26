"""Frühe Plattform- und Versionsprüfung ohne NumPy-, Modell- oder Netzwerkimport."""
from pathlib import Path
import importlib.metadata
import json
import platform

from faktenretention.paths import PROJECT_ROOT as ROOT


def require_supported_platform(system=None):
    system = platform.system() if system is None else system
    if system != 'Linux':
        raise SystemExit('Unterstützt wird Linux (unter Windows: Linux in WSL2). '
                         'Native Windows- und macOS-Ausführung sind nicht freigegeben.')


def check_analysis_environment(*, strict_cache=True, versions=None):
    require_supported_platform()
    expected = json.loads((ROOT/'data/analysis_environment.json').read_text())
    if versions is None:
        versions = {'python': platform.python_version()}
        for name in expected['analysis_requirements']:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = 'nicht installiert'
    missing = [name for name in expected['analysis_requirements'] if versions.get(name) in [None, 'nicht installiert']]
    mismatches = [f'{name}: erwartet {version}, vorhanden {versions.get(name, "nicht installiert")}'
                  for name, version in expected['cache_fingerprint_versions'].items()
                  if versions.get(name) != version]
    if missing or (strict_cache and mismatches):
        details = '\n  '.join((mismatches if strict_cache else []) + ['Fehlendes Paket: '+name for name in missing])
        raise SystemExit('Analyseumgebung ist nicht mit dem getesteten Reproduktionsweg kompatibel.\n  '
                         + details + '\nFür die eingefrorenen Regressions-Bootstrap-Caches werden Python '
                         + expected['cache_fingerprint_versions']['python']
                         + ' und die dokumentierten Paketversionen benötigt.\n'
                         'Umgebung gemäß README mit requirements-analysis.lock.txt einrichten; '
                         'keine automatische Neuberechnung der Caches.')
    return {'platform': platform.system(), 'cache_compatible': not mismatches, 'versions': versions}
