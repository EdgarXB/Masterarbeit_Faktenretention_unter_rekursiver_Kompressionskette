"""Fresh-kernel notebook execution with declared outputs and isolated reports."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import uuid
import zipfile
from faktenretention.paths import PROJECT_ROOT
from faktenretention.io import safe_output
from faktenretention.runtime_checks import check_analysis_environment
from faktenretention.validation import compare_png, compare_table, require_comparisons

def load_analyses(project=PROJECT_ROOT):
    specs=json.loads((project/'notebooks/analyses.json').read_text())['analyses']
    ids=[s['id'] for s in specs]
    if not ids or len(ids)!=len(set(ids)):raise ValueError('Leeres oder mehrdeutiges Analyseinventar.')
    for spec in specs:
        if not spec['expected']:raise ValueError('Erwartete Ausgaben fehlen: '+spec['id'])
        for dep in spec['dependencies']:
            if dep not in ids[:ids.index(spec['id'])]:raise ValueError('Ungeordnete Abhängigkeit: '+dep)
    return specs

def selected_analyses(specs,only=None):
    if only is None:return specs
    by_id={s['id']:s for s in specs};aliases={s['legacy']:s['id'] for s in specs}
    only=aliases.get(only,only)
    if only not in by_id:raise ValueError('Unbekannte Analyse: '+only)
    selected=set()
    def visit(key):
        if key in selected:return
        for dep in by_id[key]['dependencies']:visit(dep)
        selected.add(key)
    visit(only)
    return [s for s in specs if s['id'] in selected]

def prepare_workspace(output,specs,project=PROJECT_ROOT):
    from faktenretention.frozen_data import source_bytes
    output=safe_output(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Ausgabeverzeichnis muss neu oder leer sein: '+str(output))
    output.mkdir(parents=True,exist_ok=True);work=output/'workspace';work.mkdir()
    inventory=json.loads((project/'data/quelleninventar.json').read_text())
    reused={e['path'] for s in specs for e in s['reused']}
    with zipfile.ZipFile(project/'data/eingefrorene_quellen.zip') as sources, \
         zipfile.ZipFile(project/'results/originaltabellen.zip') as tables, \
         zipfile.ZipFile(project/'data/referenznotebooks.zip') as notebooks, \
         zipfile.ZipFile(project/'figures/originalabbildungen.zip') as figures:
        for entry in inventory['files']:
            name=entry['archive_path']
            if 'reference_member' in entry and name not in reused:continue
            target=work/name;target.parent.mkdir(parents=True,exist_ok=True)
            content=source_bytes(entry,sources,tables,notebooks)
            if hashlib.sha256(content).hexdigest()!=entry['sha256']:raise ValueError('Veränderte Quelle: '+name)
            target.write_bytes(content)
        for archive,kind in [(tables,'tables'),(figures,'figures')]:
            folder=output/'reference'/kind;folder.mkdir(parents=True)
            for entry in archive.infolist():
                if '/' in entry.filename or entry.is_dir():raise ValueError('Unerwarteter Referenzpfad: '+entry.filename)
                (folder/entry.filename).write_bytes(archive.read(entry))
    (work/'supplement').mkdir(exist_ok=True)
    (work/'workspace.json').write_text(json.dumps({'project':str(project),'reused_inputs':sorted(reused)},indent=2))
    return work

def verify_outputs(spec,work,output):
    comparisons=[];missing=[]
    for item in spec['expected']:
        path=work/item['path']
        if not path.is_file():missing.append(item['path']);continue
        if path.name.endswith(('.csv','.csv.gz')):
            comparisons.append(compare_table(path,(output/'reference/tables'/item['reference']).read_bytes(),item['path']))
    for item in spec['figures']:
        path=work/item['path']
        if not path.is_file():missing.append(item['path']);continue
        comparisons.append(compare_png(path,(output/'reference/figures'/item['reference']).read_bytes(),item['path']))
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
    env=os.environ.copy()
    env.update(FAKTENRETENTION_PROJECT=str(PROJECT_ROOT),FAKTENRETENTION_WORK=str(work),
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
    output=Path(output).resolve();work=prepare_workspace(output,specs)
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
            record['comparisons']=verify_outputs(spec,work,output)
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
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--list',action='store_true')
    parser.add_argument('--check-environment',action='store_true');parser.add_argument('--timeout',type=int,default=1200)
    parser.add_argument('--workers',type=int,choices=[1],default=1,help='Kompatibilität: Offline-Notebooks laufen seriell mit einem Worker')
    args=parser.parse_args(argv);specs=load_analyses()
    if args.list:
        for spec in specs:print(spec['id']+'  '+spec['notebook'])
        return
    selected=selected_analyses(specs,args.only)
    status=check_analysis_environment(strict_cache=not args.run and any(s['id']=='11_statistische_laengenkontrolle' for s in selected))
    if args.check_environment:print(json.dumps(status,ensure_ascii=False,indent=2));return
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    output=args.output or PROJECT_ROOT/'results/runs'/run_id
    if args.run:return execute_new_run(args.run,output,args.timeout)
    if args.prepare:print(prepare_workspace(output.resolve(),specs));return
    report=run_analyses(specs,selected,output,timeout=args.timeout)
    print(str(output.resolve()/'validation.json'))
    if not report['passed']:raise SystemExit('Unvollständiger Lauf oder Abweichungen; validation.json prüfen.')

if __name__=='__main__':main()
