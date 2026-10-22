# Zweck: Historische Offline-Auswertung oder gemeinsame Analyse neuer Daten.
# Eingaben: eingefrorene Einzelwerte/Caches oder --run mit neuen Tabellen.
# Ausgaben: Statistik, Grafiken und Referenzvergleich unter results/ und figures/.
# LLM-Aufrufe: keine. Beispiel: python 05_analyze.py
"""Offline-Auswertung aus statischen Originalfunktionen; keine Notebook-Umschreibung."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'data/code'))
import argparse,hashlib,gzip,json,os,subprocess,tempfile,zipfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
NOTEBOOKS=['hauptexperiment_01_qa_faktenretention', 'hauptexperiment_02_laengeneinhaltung', 'hauptexperiment_03_verlustmuster', 'hauptexperiment_04_weitere_evaluationsmetriken', 'human_validation_faktenrelevanz', 'human_validation_qa_faktenretention', 'hauptexperiment_05_alle_texttabellen', 'hauptexperiment_06_laengenmatched_vergleichsmatrizen', '01_Lead_k_Qualitaetskontrolle', '02_QA_Faktenretention_Sensitivitaet', '03_QA_Faktenretention_Laengenkontrolle', '04_LLM_Scorervergleich']
def compare_table(new_path, reference_bytes, label):
    """Alle numerischen Tabellenfelder vergleichen; Pfad-/Laufzeitbelege getrennt halten."""
    import io
    try:
        old_bytes = gzip.decompress(reference_bytes) if new_path.name.endswith('.gz') else reference_bytes
        old = pd.read_csv(io.BytesIO(old_bytes), keep_default_na=False, low_memory=False)
        new = pd.read_csv(new_path, keep_default_na=False, low_memory=False)
    except pd.errors.EmptyDataError:
        return dict(file=label, status='empty_table', passed=len(reference_bytes.strip()) == len(new_path.read_bytes().strip()))
    if new_path.name == 'export_manifest.csv':
        missing = sorted(set(old['file']) - set(new['file']))
        added = sorted(set(new['file']) - set(old['file']))
        expected = sorted(['bootstrap_kendall_10000_seed32.npz','bootstrap_summac_referenzwechsel.npz',
                           'bootstrap_paare_10000_seed32.npz','bootstrap_gruppen_10000_seed32.npz'])
        return dict(file=label, kind='artifact_inventory', expected_difference=missing == expected and not added,
                    removed_intermediate_arrays=missing, added_files=added, passed=None)
    if list(old.columns) != list(new.columns) or len(old) != len(new):
        return dict(file=label, passed=False, reason='shape_or_columns', old_shape=list(old.shape), new_shape=list(new.shape))
    ignored = [c for c in old if any(t in c.lower() for t in ['sha256','fingerprint','seconds','elapsed','warning']) or c.lower() in ['datei','source','file','path','cache_used','source_state_file','source_file','source_path']]
    keys = [c for c in old if c not in ignored and pd.api.types.is_string_dtype(old[c]) and pd.api.types.is_string_dtype(new[c])]
    if keys:
        old = old.sort_values(keys, kind='stable').reset_index(drop=True)
        new = new.sort_values(keys, kind='stable').reset_index(drop=True)
    checked, problems, max_error = [], [], 0.0
    for column in old:
        if column in ignored:
            continue
        left = pd.to_numeric(old[column], errors='coerce').astype(float)
        right = pd.to_numeric(new[column], errors='coerce').astype(float)
        if left.notna().any() or right.notna().any():
            ok = np.allclose(left, right, rtol=0, atol=1e-12, equal_nan=True)
            neither_numeric = left.isna() & right.isna()
            ok = ok and old.loc[neither_numeric,column].astype(str).equals(new.loc[neither_numeric,column].astype(str))
            error = (left-right).abs().max()
            if pd.notna(error): max_error = max(max_error, float(error))
        else:
            ok = old[column].astype(str).equals(new[column].astype(str))
        checked.append(column)
        if not ok: problems.append(column)
    return dict(file=label, passed=not problems, checked_columns=checked,
                excluded_provenance_or_runtime_columns=ignored, differences=problems, max_absolute_error=max_error)


def main():
    parser = argparse.ArgumentParser(description='Vollständige Offline-Statistik; keine LLM-Aufrufe')
    parser.add_argument('--only', choices=NOTEBOOKS, help='Eine einzelne Auswertung zur gezielten Nachprüfung')
    parser.add_argument('--run',type=Path,help='Neue Experimentdaten auswerten')
    parser.add_argument('--work',type=Path,help='Expliziter Arbeits-/Ausgabeort')
    parser.add_argument('--recompute-regression-bootstrap',action='store_true')
    parser.add_argument('--workers',type=int,default=1)
    args = parser.parse_args()
    if args.run:
        from offline_guard import install
        install()
        from analyze_run import analyze_run
        analyze_run(args.run);return
    inventory = json.loads((ROOT/'data/quelleninventar.json').read_text())
    reference_names = inventory['reference_results']
    reports, logs, patches = [], {}, {}
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='32')
    work_parent=args.work.resolve() if args.work else ROOT/'results'
    work_parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='statistik_',dir=work_parent) as temporary:
        work = Path(temporary)
        subprocess.run([sys.executable, str(ROOT/'01_prepare.py'), '--working-copy', str(work)], check=True,
                       stdout=subprocess.DEVNULL)
        with zipfile.ZipFile(ROOT/'results/originaltabellen.zip') as archive:
            references = {name: archive.read(flat) for name,flat in reference_names.items() if not name.endswith('.npz')}
        baseline_mtimes = {name: (work/name).stat().st_mtime_ns for name in references}
        all_mtimes = {str(p.relative_to(work)):p.stat().st_mtime_ns for p in work.rglob('*') if p.is_file()}
        environment['MPLCONFIGDIR'] = str(work/'matplotlib')
        (work/'supplement').mkdir(exist_ok=True)
        for name in ([args.only] if args.only else NOTEBOOKS):
            script=ROOT/'data/code'/('analysis__'+name+'.py')
            print('Offline-Auswertung: '+name,flush=True)
            log=work/(name+'.log')
            command=[sys.executable,str(script),str(work),str(args.workers)]
            if args.recompute_regression_bootstrap:command.append('--recompute-regression-bootstrap')
            with log.open('w') as handle:
                result=subprocess.run(command,env=environment,stdout=handle,stderr=subprocess.STDOUT)
            logs[name+'.log']=log.read_bytes()
            if result.returncode:
                print(log.read_text()[-6000:])
                (ROOT/'results/letzter_fehler.log').write_bytes(log.read_bytes())
                raise RuntimeError('Auswertung abgebrochen: '+name)
        with zipfile.ZipFile(ROOT/'results/neuberechnet.zip','w',zipfile.ZIP_DEFLATED) as tables, \
             zipfile.ZipFile(ROOT/'figures/neuberechnet.zip','w',zipfile.ZIP_DEFLATED) as images:
            for reference, flat in reference_names.items():
                path = work/reference
                if not path.is_file() or not path.name.endswith(('.csv','.csv.gz')) or reference not in references: continue
                if path.stat().st_mtime_ns == baseline_mtimes[reference]: continue
                # Unverändert gebliebene Referenzen werden nicht als Neuberechnung ausgegeben.
                if path.read_bytes() == references[reference]:
                    # Bitidentität kann echte Reproduktion sein; die Outputauswahl folgt den ausgeführten Notebooks.
                    pass
                item = compare_table(path, references[reference], reference)
                reports.append(item)
                tables.write(path,flat)
            for folder in [work/'project/02_Auswertung',work/'supplement/ergebnisse']:
                for path in folder.rglob('*'):
                    if not path.is_file(): continue
                    relative = str(path.relative_to(work))
                    if path.stat().st_mtime_ns == all_mtimes.get(relative): continue
                    flat = relative.replace('/','__')
                    if path.suffix in ['.png','.pdf','.svg']:
                        images.write(path,flat)
                    elif path.suffix == '.npz' and args.recompute_regression_bootstrap and path.name in ['bootstrap_10000_linear.npz','bootstrap_10000_spline.npz']:
                        tables.write(path,flat)
                    elif path.suffix in ['.csv','.gz','.md','.json','.txt'] and relative not in reference_names:
                        tables.write(path,flat)
        with zipfile.ZipFile(ROOT/'results/ausfuehrungsnachweis.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for name, content in logs.items(): archive.writestr(name,content)
            for name, content in patches.items(): archive.writestr(name,content)
        reused_tables = [name for name in references if name.endswith(('.csv','.csv.gz')) and (work/name).stat().st_mtime_ns == baseline_mtimes[name]]
        summary = dict(reused_reference_tables=reused_tables, llm_generation_calls=0, notebooks_executed=[args.only] if args.only else NOTEBOOKS,
                       numeric_tolerance=1e-12, tables_checked=len(reports),
                       tables_passed=sum(bool(x.get('passed')) for x in reports), comparisons=reports,
                       statistical_tables_checked=sum(x.get('kind') != 'artifact_inventory' for x in reports),
                       expected_inventory_differences=sum(bool(x.get('expected_difference')) for x in reports),
                       regression_bootstrap='original refits requested' if args.recompute_regression_bootstrap else 'existing fingerprint-checked caches; optional --recompute-regression-bootstrap')
        (ROOT/'results/validierung.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        print(f"Statistische Tabellen: {summary['tables_passed']}/{summary['statistical_tables_checked']} bestanden; {summary['expected_inventory_differences']} erwartete Inventaränderung.")
        if not all(x.get('passed') or x.get('expected_difference') for x in reports):
            raise SystemExit('Abweichungen in results/validierung.json prüfen.')


if __name__ == '__main__':
    main()
