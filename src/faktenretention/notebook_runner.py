"""Fresh-kernel notebook execution with declared outputs and isolated reports."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import sys
import zipfile
import re
from faktenretention.paths import PROJECT_ROOT
from faktenretention.io import safe_output
from faktenretention.runtime_checks import check_analysis_environment
from faktenretention.validation import compare_png, compare_table, require_comparisons

def load_analyses(project=PROJECT_ROOT):
    specs=json.loads((project/'notebooks/analyses.json').read_text())['analyses']
    for spec in specs:
        key = spec['id']
        spec['notebook'] = f'notebooks/{key}.ipynb'
        for kind, folder, prefix in [('expected', 'tables', 'table_reference_prefix'),
                                     ('figures', 'figures', 'figure_reference_prefix'),
                                     ('reused', 'caches', 'input_reference_prefix')]:
            spec[kind] = [dict(path=f'{folder}/{key}/{name}', reference=spec[prefix]+name)
                          for name in spec[kind]]
        spec['additional_exports'] = [str(Path(item['path']).with_suffix('.'+ext))
            for item in spec['figures'] for ext in spec['figure_formats']]
    ids=[s['id'] for s in specs]
    if not ids or len(ids)!=len(set(ids)):raise ValueError('Leeres oder mehrdeutiges Analyseinventar.')
    for spec in specs:
        if not spec['expected']:raise ValueError('Erwartete Ausgaben fehlen: '+spec['id'])
        for dep in spec['dependencies']:
            if dep not in ids[:ids.index(spec['id'])]:raise ValueError('Ungeordnete Abhängigkeit: '+dep)
    return specs

def selected_analyses(specs,only=None):
    if only is None:return specs
    by_id={s['id']:s for s in specs};aliases={s.get('legacy',s['id']):s['id'] for s in specs}
    only=aliases.get(only,only)
    if only not in by_id:raise ValueError('Unbekannte Analyse: '+only)
    selected=set()
    def visit(key):
        if key in selected:return
        for dep in by_id[key]['dependencies']:visit(dep)
        selected.add(key)
    visit(only)
    return [s for s in specs if s['id'] in selected]

def prepare_run(output):
    from faktenretention.frozen_data import prepare_notebook_data
    output=safe_output(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Ausgabeverzeichnis muss neu oder leer sein: '+str(output))
    output.mkdir(parents=True,exist_ok=True)
    prepare_notebook_data(output/'inputs', include_tables=False)
    return output

def verify_outputs(spec,work):
    comparisons=[];missing=[]
    with zipfile.ZipFile(PROJECT_ROOT/'results/originaltabellen.zip') as tables, \
         zipfile.ZipFile(PROJECT_ROOT/'figures/originalabbildungen.zip') as figures:
        for item in spec['expected']:
            path=work/item['path']
            if not path.is_file():missing.append(item['path']);continue
            if path.name.endswith(('.csv','.csv.gz')):
                reference=tables.read(item['reference'])
                comparisons.append(compare_table(path,reference,item['path']))
        for item in spec['figures']:
            path=work/item['path']
            if not path.is_file():missing.append(item['path']);continue
            comparisons.append(compare_png(path,figures.read(item['reference']),item['path']))
    for name in spec.get('additional_exports',[]):
        if not (work/name).is_file():missing.append(name)
    if missing:raise FileNotFoundError('Erwartete Ausgaben fehlen: '+', '.join(missing))
    return comparisons

def execute_notebook(spec,work,output,*,timeout=1200):
    import nbformat
    from nbclient import NotebookClient
    notebook=nbformat.read(PROJECT_ROOT/spec['notebook'],as_version=4)
    for cell in notebook.cells:
        if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
    destination=output/'notebooks'/Path(spec['notebook']).name;destination.parent.mkdir(exist_ok=True)
    for cell in notebook.cells:
        if cell.cell_type=='code' and 'parameters' in cell.metadata.get('tags',[]):
            for name, value in [('DATA_DIR',work/'inputs'),('OUTPUT_DIR',output),('TABLES',work/'tables')]:
                cell.source=re.sub(r'^'+name+r' = .+$', name+' = Path('+repr(str(value))+')', cell.source, flags=re.M)
    env=os.environ.copy()
    env.pop('FAKTENRETENTION_WORK',None)
    env.update(
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='32',MPLCONFIGDIR=str(output/'matplotlib'))
    client=NotebookClient(notebook,timeout=timeout,kernel_name='python3',allow_errors=False,record_timing=True)
    client.create_kernel_manager()
    client.km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    try:client.execute(cwd=str(work),env=env)
    finally:nbformat.write(notebook,destination)
    return destination

def run_analyses(specs,selected,output,*,execute=execute_notebook,timeout=1200):
    output=Path(output).resolve();work=prepare_run(output)
    chosen={s['id'] for s in selected};records=[];by_id={}
    for spec in specs:
        record={'id':spec['id'],'status':'skipped','passed':False,'reason':'not_selected'}
        records.append(record);by_id[spec['id']]=record
        if spec['id'] not in chosen:continue
        if any(not by_id[d]['passed'] for d in spec['dependencies']):record['reason']='dependency_failed';continue
        print('Notebook: '+spec['id'],flush=True)
        try:
            for item in spec['expected']+spec['figures']:
                if (work/item['path']).exists():raise ValueError('Neue Ausgabe existiert vor Ausführung: '+item['path'])
            execute(spec,work,output,timeout=timeout)
            record['comparisons']=verify_outputs(spec,work)
            record['passed']=require_comparisons(record['comparisons'])
            record.update(status='fully_executed',reason=None,reused_inputs=spec['reused'])
        except Exception as error:
            has_outputs=any((work/e['path']).exists() for e in spec['expected'])
            record.update(status='partially_executed' if has_outputs else 'failed',reason=str(error),error_type=type(error).__name__)
            print('Fehler: '+spec['id']+': '+str(error)[-1000:],flush=True)
        (output/'validation.json').write_text(json.dumps({'analyses':records,'numeric_tolerance':1e-12},ensure_ascii=False,indent=2))
    comparisons=[c for r in records for c in r.get('comparisons',[])]
    passed=bool(chosen) and all(by_id[k]['passed'] for k in chosen) and require_comparisons(comparisons)
    report={'passed':passed,'scope':'all' if len(chosen)==len(specs) else 'selection_with_dependencies',
        'requested_analyses':sorted(chosen),'analyses':records,'numeric_tolerance':1e-12,
        'tables_checked':sum(c.get('kind')!='figure' for c in comparisons),
        'figures_checked':sum(c.get('kind')=='figure' for c in comparisons),
        'llm_generation_calls':0,'regression_bootstrap':'existing fingerprint-checked caches only'}
    (output/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report

def execute_new_run(run,output,timeout):
    run=Path(run).resolve();output=safe_output(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Ausgabeverzeichnis muss neu oder leer sein.')
    output.mkdir(parents=True,exist_ok=True)
    os.environ['FAKTENRETENTION_NEW_RUN']=str(run);os.environ['FAKTENRETENTION_NEW_OUTPUT']=str(output/'analysis')
    execute_notebook({'notebook':'notebooks/ergaenzungen/neue_laufdaten.ipynb'},run,output,timeout=timeout)
    required=['laenge_artikelebene.csv','mittelwerte_95ci.csv','analysis_manifest.json','D.png']
    missing=[p for p in required if not (output/'analysis'/p).is_file()]
    report={'source':'new_run','passed':not missing,'missing':missing,'historical_panel_models':False,'llm_generation_calls':0}
    (output/'validation.json').write_text(json.dumps(report,indent=2))
    if missing:raise FileNotFoundError(str(missing))
    print(output/'validation.json')

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',help='Analyse-ID oder historischer Name; Abhängigkeiten inklusive')
    parser.add_argument('--output','--work',dest='output',type=Path,help='Neues Laufverzeichnis')
    parser.add_argument('--run',type=Path,help='Neue Experimentdaten im Ergänzungsnotebook auswerten')
    parser.add_argument('--prepare',action='store_true',help='Einmalige Datenvorbereitung für direktes Run All')
    parser.add_argument('--data-dir',type=Path,help='Eigener Ordner für --prepare (Standard: data/prepared)')
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--check-environment',action='store_true');parser.add_argument('--timeout',type=int,default=1200)
    args=parser.parse_args(argv);specs=load_analyses()
    if args.list:
        for spec in specs:print(spec['id']+'  '+spec['notebook'])
        return
    selected=selected_analyses(specs,args.only)
    status=check_analysis_environment(strict_cache=not args.run and any(s['id']=='11_statistische_laengenkontrolle' for s in selected))
    if args.check_environment:print(json.dumps(status,ensure_ascii=False,indent=2));return
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output=args.output or PROJECT_ROOT/'results/runs'/run_id
    if args.run:return execute_new_run(args.run,output,args.timeout)
    if args.prepare:
        from faktenretention.frozen_data import prepare_notebook_data
        print(prepare_notebook_data(args.data_dir or args.output));return
    report=run_analyses(specs,selected,output,timeout=args.timeout)
    print(str(output.resolve()/'validation.json'))
    if not report['passed']:raise SystemExit('Unvollständiger Lauf oder Abweichungen; validation.json prüfen.')

if __name__=='__main__':main()
