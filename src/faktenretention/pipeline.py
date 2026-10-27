"""Prozedurale Verbindung der ursprünglichen Experimentphasen.

Die fachlichen Regeln kommen aus den mitgelieferten Originalmodulen. Dieses
Modul verbindet deren Dateien und explizite Parameter; beim Import läuft nichts.
"""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import sys
import time
from faktenretention.request_budget import activate, save_json
from faktenretention.input_checks import validate_article_keys
from faktenretention.text_rules import count_words, bounds_for_target, lead_k, direct_prompt, seed_raw, extract_text, extract_thinking, complete_sentence

from faktenretention.io import read_csv, write_csv, sha, safe_output
from faktenretention.paths import PROJECT_ROOT as PACKAGE
from faktenretention.protocol import STAGES, METHODS
MODELS = {
 'summary':dict(name='thesis-seed:v1',digest='7a66a2f466bf48fdafa7004a7975a7f5fac6e667a6de7d01751aacb98b3f387c'),
 'facts':dict(name='thesis-gemma:v1',digest='6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7'),
 'qa':dict(name='thesis-qwen38-qa:v1',digest='d80421b9191aff0d4d8ea0911068452f11f3963e3e282c6b9d778932c4b268b3')}










def prepare_articles(source, output, *, limit=None, split='test', resume=False):
    output=safe_output(output)
    rows=read_csv(source)
    if not rows or not {'article','highlights'}.issubset(rows[0]):
        raise ValueError('Artikel-CSV benötigt article und highlights.')
    prepared=[];seen=set();excluded=[]
    indexed=[(int(row.get('dataset_index') or i),i,row) for i,row in enumerate(rows)]
    # Auch Kollisionen zwischen expliziten Indizes und automatisch vergebenen Zeilenindizes ablehnen.
    validate_article_keys([dict(dataset_index=index, split=split) for index,_,_ in indexed],
                          require_article_ids=False)
    for dataset_index, position, row in sorted(indexed):
        article=str(row['article']).strip();highlights=str(row['highlights']).strip()
        digest=hashlib.sha256(article.encode()).hexdigest()
        if not 795 <= count_words(article) <= 805 or not highlights or digest in seen:
            excluded.append(position);continue
        seen.add(digest)
        prepared.append(dict(article_id=str(len(prepared)+1),dataset=row.get('dataset') or 'user_csv',config=row.get('config') or '',
            split=split,dataset_index=str(dataset_index),word_count=str(count_words(article)),article_sha256=digest,
            article=article,highlights=highlights,source_article_id=str(row.get('article_id',''))))
        if limit is not None and len(prepared)>=limit:break
    if not prepared:raise ValueError('Keine Artikel erfüllen die ursprüngliche 795–805-Wort-Regel.')
    target=output/'data/articles.csv'
    receipt_path=output/'preparation.json'
    receipt=dict(protocol_version=1,source_sha256=sha(source),accepted=len(prepared),excluded_positions=excluded,
        options=dict(limit=limit,split=split),
        rule='795..805 regex words, nonempty highlights, ascending dataset_index, unique article SHA-256')
    if target.exists() or receipt_path.exists():
        if not resume:
            raise ValueError('Artikeldatei bereits vorhanden; neues Ausgabeverzeichnis wählen.')
        if not target.is_file() or not receipt_path.is_file():
            raise ValueError('Unvollständige Vorbereitung: Artikeldatei und preparation.json werden gemeinsam benötigt.')
        stored=json.loads(receipt_path.read_text())
        expected={**receipt,'prepared_sha256':sha(target)}
        if stored!=expected or read_csv(target)!=prepared:
            raise ValueError('Vorbereitete Daten oder Konfiguration verändert bzw. nicht nachweisbar identisch. '
                             'Keine Generierung gestartet; neues Ausgabeverzeichnis verwenden.')
        validate_article_keys(prepared)
        return target
    write_csv(target,prepared)
    save_json(receipt_path,{**receipt,'prepared_sha256':sha(target)})
    return target


def load_resources(path):
    if path is None:return {}
    path=Path(path).resolve();resources=json.loads(path.read_text())
    for key,value in list(resources.items()):
        if value is not None:resources[key]=str((path.parent/value).resolve())
    return resources


def run_original(main, arguments):
    old=sys.argv
    try:
        sys.argv=['historical_phase',*map(str,arguments)]
        result=main()
        if result not in [None,0]:raise RuntimeError('Originalphase meldet unvollständige Ergebnisse.')
    finally:sys.argv=old


def start_run(run, *, mode, method='plain', article_limit=1, stages=None, max_requests=30, host='127.0.0.1:11435', summary_only=False, resources=None, semantic_metrics=False, bank=None, verify_models=True):
    run=safe_output(run);stages=stages or ([500] if mode=='mini' else STAGES)
    if mode=='mini' and not 1<=article_limit<=5:raise ValueError('Mini-Test: 1–5 Artikel.')
    articles=read_csv(run/'data/articles.csv')
    validate_article_keys(articles)
    if mode=='mini':articles=articles[:article_limit]
    methods=[method] if mode=='mini' else METHODS
    if not articles:raise ValueError('Keine vorbereiteten Artikel.')
    if stages != STAGES[:len(stages)]:
        raise ValueError('Neue rekursive Ketten müssen mit 500 beginnen und die Stufenreihenfolge erhalten.')
    if len(stages)>1 and mode=='mini' and not summary_only:
        raise ValueError('Vier Mini-Teststufen mit vollständiger QA überschreiten 30 Anfragen. --summary-only ausdrücklich verwenden.')
    minimum=len(articles)*(sum(m!='lead_k' for m in methods)*len(stages)+(0 if summary_only else 1+20*len(methods)*len(stages)))
    if minimum>max_requests:raise ValueError('Bereits die Mindestanzahl der geplanten Anfragen überschreitet das Gesamtbudget.')
    if bank is None:bank=PACKAGE/'prompts'
    bank=Path(bank).resolve()
    bank_info=json.loads((bank/'bank_manifest.json').read_text())
    if set(a['article_sha256'] for a in articles)&set(bank_info.get('source_article_hashes',[])):
        raise ValueError('Artikel überschneiden sich mit der Demonstrationsentwicklung.')
    if semantic_metrics:
        import faktenretention.metric_resources as metric_resources
        metric_resources.configure(resources or {})
        metric_resources.local_model('bertscore');metric_resources.summac_resources()
    required=['summary'] if any(m!='lead_k' for m in methods) else []
    if not summary_only:required+=['facts','qa']
    config=dict(mode=mode,article_ids=[r['article_id'] for r in articles],articles_sha256=sha(run/'data/articles.csv'),
        methods=methods,stages=stages,max_requests=max_requests,host=host,models=MODELS,required_roles=required,
        summary_only=summary_only,semantic_metrics=semantic_metrics,resources=resources or {},
        bank_manifest_sha256=sha(bank/'bank_manifest.json'))
    print(json.dumps(dict(mode=mode,articles=len(articles),methods=methods,stages=stages,
         maximum_requests=max_requests,minimum_requests=minimum,models={r:MODELS[r] for r in required}),ensure_ascii=False,indent=2))
    activate(run,config,verify_models=verify_models)
    write_csv(run/'data/selected_articles.csv',articles)
    return config


def generate_facts(run, config):
    import faktenretention.facts as original
    run=Path(run)
    validate_article_keys(read_csv(run/'data/selected_articles.csv'))
    run_original(original.main,['--articles',run/'data/selected_articles.csv','--output-dir',run/'facts',
        '--experiment-run-id','reproduction','--model','facts='+MODELS['facts']['name'],
        '--preferred-canonical-model',MODELS['facts']['name'],'--host',config['host'],
        '--facts-per-article','20','--candidate-pool-size','24','--temperature','0.0','--seed','32',
        '--num-ctx','8192','--max-attempts','5','--resume'])
    canonical=read_csv(run/'facts/canonical_facts.csv')
    from collections import Counter
    counts=Counter(r['article_id'] for r in canonical)
    if any(counts.get(a)!=20 for a in config['article_ids']):
        raise RuntimeError('Originaler Fakten-Gate nicht erfüllt; keine fachlich andere Ersatzselektion.')
    write_csv(run/'data/canonical_facts.csv',canonical)


def render_summary_prompt(method, text, target, bank):
    if method=='plain':return f'summarize in {target}\n\n'+text.strip()
    if method=='zero_shot_cot':return direct_prompt(text,target,True)
    manifest=json.loads((Path(bank)/'bank_manifest.json').read_text())
    entry=manifest['methods'][method][str(target)]
    path=Path(bank)/entry['path']
    if sha(path)!=entry['sha256']:raise ValueError('Geänderter Demonstrationsprompt: '+entry['path'])
    template=path.read_text()
    if template.count('{{NEW_INPUT}}')!=1:raise ValueError('Ungültiger Promptplatzhalter.')
    return template.replace('{{NEW_INPUT}}',text.strip(),1)


def generate_summaries(run, config, bank=None):
    from faktenretention.model_transport import ollama_generate
    run=Path(run);bank=Path(bank) if bank else PACKAGE/'prompts'
    articles=read_csv(run/'data/selected_articles.csv');records=[]
    validate_article_keys(articles)
    for position,article in enumerate(articles):
        generative=METHODS[1:];shift=position%len(generative)
        order=['lead_k',*generative[shift:],*generative[:shift]]
        for method in order:
            if method not in config['methods']:continue
            state_path=run/'states'/method/(article['article_id']+'.json')
            state=json.loads(state_path.read_text()) if state_path.exists() else dict(complete=False,method=method,source_sha256=article['article_sha256'],run_signature=sha(run/'execution.json'),stages=[])
            if state['source_sha256']!=article['article_sha256'] or state['run_signature']!=sha(run/'execution.json'):
                raise ValueError('Unvereinbare Quelle oder Konfiguration im Generierungszustand.')
            current=article['article'];stages=state['stages']
            for index,target in enumerate(config['stages']):
                if index<len(stages):
                    if stages[index]['input_text']!=current:raise ValueError('Unvereinbarer rekursiver Wiederaufnahmestand.')
                    current=stages[index]['selected']['response'];continue
                prompt='';raw={};started=time.monotonic()
                if method=='lead_k':response=lead_k(current,target);thinking=''
                else:
                    prompt=render_summary_prompt(method,current,target,bank)
                    raw=ollama_generate(model=MODELS['summary']['name'],prompt=seed_raw(prompt,method=='zero_shot_cot',reflection_interval=512,thinking_budget=1024),
                        host=config['host'],temperature=0.0,top_p=.95,repeat_penalty=1.0,seed=32,num_ctx=524288,
                        keep_alive='30m',num_predict=4096,raw=True,timeout=1800)
                    response=extract_text(raw).strip();thinking=extract_thinking(raw).strip()
                    if method=='few_shot_cot' and 'FINAL_SUMMARY:' in response:
                        prefix,response=response.rsplit('FINAL_SUMMARY:',1);thinking=(thinking+'\n'+prefix).strip();response=response.strip()
                words=count_words(response);low,high=bounds_for_target(target,5.0)
                selected=dict(response=response,thinking=thinking,response_word_count=words,thinking_word_count=count_words(thinking),
                    word_window_success=low<=words<=high,ends_with_complete_sentence=complete_sentence(response),
                    word_window_violation_words=max(low-words,words-high,0),word_window_violation_rate=max(low-words,words-high,0)/target,
                    empty_output=not bool(response),attempt_index=1,sampling_seed=32,temperature=0.0,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),raw_wrapper_cot=method=='zero_shot_cot',
                    full_raw_response=extract_text(raw) if raw else response,eval_count=raw.get('eval_count'),
                    done_reason=raw.get('done_reason','deterministic'),latency_seconds=time.monotonic()-started)
                stages.append(dict(target_words=target,input_text=current,input_word_count=count_words(current),selected=selected,attempts=[selected]))
                save_json(state_path,state);current=response
            state['complete']=True;save_json(state_path,state)
            row={**article,'method':method,'prompt_variant':method,'batch_run_id':'reproduction',
                'model_run_id':'summary','model_name':MODELS['summary']['name'],'collection_phase':'independent_reproduction'}
            for stage in stages:
                t=stage['target_words'];s=stage['selected'];row[f'summary_{t}']=s['response']
                row[f'summary_{t}_word_count']=s['response_word_count'];row[f'summary_{t}_attempts']=1
                row[f'summary_{t}_initial_window_success']=str(s['word_window_success'])
                row[f'summary_{t}_complete_sentence_success']=str(s['ends_with_complete_sentence'])
            records.append(row)
    write_csv(run/'data/summaries.csv',records)
    # Der Inhalt folgt dem vorhandenen Generation-State-Format, auch für neue Artikel.
    with (run/'data/generation_states.jsonl').open('w') as handle:
        for row in records:
            path=run/'states'/row['method']/(row['article_id']+'.json')
            handle.write(json.dumps(dict(article_id=int(row['article_id']),method=row['method'],state=json.loads(path.read_text())),ensure_ascii=False)+'\n')


def generate_qa(run, config):
    import faktenretention.qa as original
    run=Path(run)
    validate_article_keys(read_csv(run/'data/selected_articles.csv'))
    arguments=['--facts',run/'data/canonical_facts.csv','--summary-file',run/'data/summaries.csv',
        '--output-dir',run/'qa','--experiment-run-id','reproduction','--model','qa='+MODELS['qa']['name'],
        '--host',config['host'],'--temperature','0.0','--seed','32','--num-ctx','8192',
        '--api-retries','3','--accept-terminal-grounding-missing','--resume']
    for stage in config['stages']:arguments+=['--stage',stage]
    run_original(original.main,arguments)
    answers=read_csv(run/'qa/summary_question_answers.csv')
    write_csv(run/'data/answers.csv',answers)
    import faktenretention.metric_scoring as scorer
    (run/'qa_report').mkdir(exist_ok=True)
    scorer.score_qa(run/'data/answers.csv',run/'qa_report')


def compute_metrics(run, config, *, semantic=None):
    run=Path(run)
    semantic=config['semantic_metrics'] if semantic is None else semantic
    if semantic:
        import faktenretention.metric_resources as metric_resources
        metric_resources.configure(config['resources'])
        import faktenretention.metrics as original
        run_original(original.main,['--results',run/'data/summaries.csv','--output-dir',run/'metrics',
            '--bertscore-model',metric_resources.local_model('bertscore'),'--summac-device','cpu','--stages',*config['stages']])
        write_csv(run/'data/metrics_detailed.csv',read_csv(run/'metrics/metrics_detailed.csv'))
        return
    # Dieselben lexikalischen Werte; semantische Werte bleiben ausdrücklich fehlend.
    from rouge_score import rouge_scorer
    from faktenretention.metrics import numeric_tokens
    rouge=rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=True);rows=[]
    for row in read_csv(run/'data/summaries.csv'):
        current=row['article']
        for target in config['stages']:
            text=row[f'summary_{target}'];scores=rouge.score(row['highlights'],text)
            words=count_words(text);low,high=bounds_for_target(target,5.0)
            new_article=sorted(numeric_tokens(text)-numeric_tokens(row['article']));new_input=sorted(numeric_tokens(text)-numeric_tokens(current))
            rows.append(dict(article_id=row['article_id'],dataset_index=row['dataset_index'],method=row['method'],stage_target_words=target,
                stage_actual_words=words,accepted_min_words=low,accepted_max_words=high,word_window_success=int(low<=words<=high),
                absolute_word_error=abs(words-target),absolute_relative_word_error=abs(words-target)/target,
                word_window_violation_words=max(low-words,words-high,0),word_window_violation_rate=max(low-words,words-high,0)/target,
                complete_sentence_success=int(complete_sentence(text)),empty_output=int(not bool(text)),
                **{f'highlight_{metric}_f1':scores[metric].fmeasure for metric in ['rouge1','rouge2','rougeL']},
                highlight_bertscore_f1='',source_summac='',stage_input_summac='',
                novel_numeric_tokens_vs_article='|'.join(new_article),novel_numeric_rate_vs_article=int(bool(new_article)),
                novel_numeric_tokens_vs_stage_input='|'.join(new_input),novel_numeric_rate_vs_stage_input=int(bool(new_input))))
            current=text
    write_csv(run/'data/metrics_detailed.csv',rows)
    save_json(run/'metrics_status.json',dict(rouge='recomputed',bertscore='not_requested',summac='not_requested',fabricated_scores=False))


def construct_demonstrations(train, validation, output, *, resources, max_requests, host='127.0.0.1:11435', device='cpu', verify_models=True):
    """Originale gemeinsame Kandidatenpipeline; ausschließlich expliziter Vollmodus."""
    import faktenretention.metric_resources as metric_resources
    output=safe_output(output)
    metric_resources.configure(resources)
    embedding=metric_resources.local_model('embedding');metric_resources.summac_resources()
    train_file=prepare_articles(train,output/'candidate_input',limit=240,split='train',resume=True)
    validation_file=prepare_articles(validation,output/'validation_input',limit=30,split='validation',resume=True)
    training=read_csv(train_file);validating=read_csv(validation_file)
    if len(training)<130 or len(validating)!=30:
        raise ValueError('Originales Kandidatenprotokoll benötigt mindestens 130 Trainings- und genau 30 Validierungsartikel; maximal 240 Training.')
    train_hashes={r['article_sha256'] for r in training};validation_hashes={r['article_sha256'] for r in validating}
    if train_hashes & validation_hashes:raise ValueError('Trainings-/Validierungsartikel überschneiden sich.')
    config=dict(mode='full',article_ids=['train:'+r['dataset_index'] for r in training]+['validation:'+r['dataset_index'] for r in validating],
        methods=['auto_cot','few_shot_cot','automate_cot'],max_requests=max_requests,host=host,models=MODELS,
        required_roles=['summary'],phase='demonstration_construction',resources=resources,device=device,
        implementation_sha256=sha(__file__),
        train_sha256=sha(train_file),validation_sha256=sha(validation_file))
    print(json.dumps(dict(phase=config['phase'],training_articles=len(training),validation_articles=len(validating),
                          maximum_requests=max_requests,temperature=1.1,thinking_budget_instruction=512),indent=2))
    activate(output,config,verify_models=verify_models)
    import faktenretention.demonstrations.generate_cot_candidate_rollouts as rollouts
    import faktenretention.demonstrations.select_autocot_demonstrations as auto
    import faktenretention.demonstrations.score_and_prune_automatecot_candidates as prune
    import faktenretention.demonstrations.select_automatecot_demonstrations as optimize
    import faktenretention.demonstrations.build_shared_pool_fewshot_prompts as fewshot
    common=['--host',host,'--num-ctx','524288','--timeout','1800','--max-attempts','12']
    run_original(rollouts.main,['--input',train_file,'--output-dir',output/'candidate_rollouts',*common,
        '--minimum-articles-before-eligibility-stop','130','--minimum-eligible-per-stage','110','--minimum-auto-eligible-per-stage','25'])
    run_original(rollouts.main,['--input',validation_file,'--output-dir',output/'validation_rollouts',*common,'--allow-nonsubstantive-rationale'])
    run_original(auto.main,['--candidates',output/'candidate_rollouts/stage_candidates.jsonl','--output-dir',output/'auto_cot_selection',
        '--model-path',embedding,'--device',device,'--clusters','5','--chunk-words','160','--batch-size','32','--random-state','32','--n-init','20'])
    run_original(prune.main,['--candidates',output/'candidate_rollouts/stage_candidates.jsonl','--output-dir',output/'automate_cot_pruning',
        '--pool-size','100','--summac-device',device])
    run_original(optimize.main,['--pools',output/'automate_cot_pruning/automate_cot_pruned_pools.json',
        '--validation-candidates',output/'validation_rollouts/stage_candidates.jsonl','--output-dir',output/'automate_cot_selection',
        '--host',host,'--num-ctx','524288','--sample-size','5','--pge-avg-samples','10','--epochs','5','--batch-size','10',
        '--lr','0.001','--timeout','1800','--api-retries','3'])
    run_original(fewshot.main,['--project',output,'--output-dir',output/'prompts'])
    # Originaler Renderer, originale Reihenfolge; keine neue Auswahlregel.
    from faktenretention.demonstrations.select_automatecot_demonstrations import build_fewshot_prompt
    methods={}
    for method in ['auto_cot','automate_cot']:
        selected=json.loads((output/(method+'_selection')/(method+'_demonstrations.json')).read_text())
        methods[method]={}
        for target in STAGES:
            prompt=build_fewshot_prompt('{{NEW_INPUT}}',target,selected[str(target)])+'\n'
            path=output/'prompts'/f'{method}_{target}.txt';path.write_text(prompt)
            methods[method][str(target)]=dict(path=path.name,sha256=sha(path))
    manifest=json.loads((output/'prompts/few_shot_cot_manifest.json').read_text())
    methods['few_shot_cot']={t:dict(path=e['path'],sha256=e['sha256']) for t,e in manifest['prompts'].items()}
    save_json(output/'prompts/bank_manifest.json',dict(methods=methods,source_article_hashes=sorted(train_hashes|validation_hashes),
        construction='original common candidates, Auto-CoT KMeans, SummaC top100, original VR-PGE/Adam, seed32 Few-Shot'))
    return output/'prompts'
