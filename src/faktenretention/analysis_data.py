"""Artikelbasierte Originalkennzahlen für neue und eingefrorene Tabellen.

estimate/contrasts_for und der QA-Scorer sind unveränderte Originalfunktionen.
Historische Humanratings und Modellvergleichscaches werden neuen Daten nicht
zugeschrieben. Grafiken entstehen in notebooks/ergaenzungen/neue_laufdaten.ipynb.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from faktenretention.io import read_csv, write_csv, safe_output
from faktenretention.protocol import STAGES, METHODS
from faktenretention.paths import PROJECT_ROOT as PACKAGE
from faktenretention.text_rules import count_words,bounds_for_target
import faktenretention.scoring as scorer
from faktenretention.statistics import estimate,contrasts_for


def label_answer(row):
    if str(row.get('error','')):return 'technical_missing'
    if scorer.retained(str(row['reference_answer']),str(row['answer'])):return 'retained'
    if str(row['answer']).upper()=='NOT_FOUND':return 'lost'
    return 'reference_discordant'


def analyze_run(run=None, output=None):
    frozen=run is None
    root=PACKAGE if frozen else Path(run).resolve()
    output=safe_output(output or (PACKAGE/'results/gemeinsame_auswertung' if frozen else root/'analysis'))
    output.mkdir(parents=True,exist_ok=True)
    summaries=read_csv(root/('data/zusammenfassungen.csv.gz' if frozen else 'data/summaries.csv'))
    answer_path=root/('data/qa_antworten.csv.gz' if frozen else 'data/answers.csv')
    qa=read_csv(answer_path) if answer_path.exists() else []
    metrics_path=root/('data/modellmetriken.csv' if frozen else 'data/metrics_detailed.csv')
    lengths=[];summary_lookup={}
    for row in summaries:
        previous=row['article'];chain_failure=False
        for stage in STAGES:
            if f'summary_{stage}' not in row:continue
            text=row[f'summary_{stage}'];words=count_words(text);low,high=bounds_for_target(stage,5.0)
            chain_failure=chain_failure or not text.strip()
            item=dict(article_id=int(row['article_id']),method=row.get('method') or row['prompt_variant'],stage_target_words=stage,
                W=words,U=low,O=high,violation_words=max(low-words,words-high,0),D=max(low-words,words-high,0)/stage,
                absolute_relative_word_error=abs(words-stage)/stage,word_window_success=int(low<=words<=high),
                empty_output=not bool(text.strip()),generation_chain_failure=chain_failure,
                input_word_count=count_words(previous))
            lengths.append(item);summary_lookup[(item['article_id'],item['method'],stage)]=text;previous=text
    length=pd.DataFrame(lengths);cell=['article_id','method','stage_target_words']
    assert not length.duplicated(cell).any()
    length.to_csv(output/'laenge_artikelebene.csv',index=False)
    rows=[]
    for row in qa:
        method=row.get('method') or row['summary_prompt_variant'];key=(int(row['article_id']),method,int(row['stage_target_words']))
        label=label_answer(row)
        grounded=scorer.support_is_grounded(summary_lookup[key],row['answer'],row.get('support',''))
        if label!='technical_missing' and not grounded:raise ValueError('Nicht belegte QA-Antwort ohne technischen Fehlerstatus.')
        rows.append({**row,'article_id':key[0],'method':method,'stage_target_words':key[2],
            'label':label,'analysis_eligible':label!='technical_missing','technical_missing':label=='technical_missing',
            'retained':label=='retained','lost':label=='lost','reference_discordant':label=='reference_discordant','grounded':grounded})
    pd.DataFrame(rows).to_csv(output/'qa_labels.csv.gz',index=False,compression='gzip')
    outputs=[]
    if rows:
        frame=pd.DataFrame(rows)
        key=cell+['canonical_fact_id']
        assert not frame.duplicated(key).any()
        article=frame.groupby(cell).agg(n_total=('canonical_fact_id','size'),n_valid=('analysis_eligible','sum'),
            n_missing=('technical_missing','sum'),n_retained=('retained','sum'),n_lost=('lost','sum'),n_reference_discordant=('reference_discordant','sum')).reset_index()
        if not article.n_total.eq(20).all():raise ValueError('Jede QA-Zelle benötigt die ursprünglichen 20 Faktenpositionen.')
        article['retention_observed']=article.n_retained.div(article.n_valid.replace(0,np.nan))
        article['retention_lower']=article.n_retained/20
        article['retention_upper']=(article.n_retained+article.n_missing)/20
        article=article.merge(length,on=cell,validate='one_to_one')
        article['fully_observed']=article.n_missing.eq(0)&~article.generation_chain_failure
        article['retention_adjusted']=article.retention_lower/(1+article.D)
        article.to_csv(output/'retention_artikelebene.csv',index=False)
        for metric in ['retention_observed','retention_lower','retention_upper','retention_adjusted']:
            for (method,stage),sub in article.groupby(['method','stage_target_words']):
                outputs.append(dict(metric=metric,method=method,stage_target_words=stage,**estimate(sub.set_index('article_id')[metric])))
            # Die unveränderte Kontrastfunktion erwartet das vollständige Panel.
            if set(article.method)==set(METHODS) and set(article.stage_target_words)==set(STAGES):
                contrasts_for(article,metric,metric).to_csv(output/(metric+'_kontraste.csv'),index=False)
        # Dieselbe Artikelgewichtung innerhalb der Entitätstypen.
        if 'entity_type' in frame:
            entity=frame.groupby(cell+['entity_type']).agg(n_valid=('analysis_eligible','sum'),n_retained=('retained','sum'),n_missing=('technical_missing','sum')).reset_index()
            entity['retention_observed']=entity.n_retained.div(entity.n_valid.replace(0,np.nan))
            entity.to_csv(output/'entitaeten_artikelebene.csv',index=False)
        # Gepaarte Übergänge: technische Missingness bleibt getrennt.
        transition=[]
        for first,second in zip(STAGES,STAGES[1:]):
            a=frame[frame.stage_target_words.eq(first)];b=frame[frame.stage_target_words.eq(second)]
            pair=a.merge(b,on=['article_id','method','canonical_fact_id'],suffixes=('_from','_to'),validate='one_to_one')
            if len(pair):
                out=pair.groupby(['method','label_from','label_to']).size().rename('n').reset_index()
                out['from_stage']=first;out['to_stage']=second;transition.extend(out.to_dict('records'))
        write_csv(output/'verlustuebergaenge.csv',transition)
    for metric in ['D','W','word_window_success','absolute_relative_word_error']:
        for (method,stage),sub in length.groupby(['method','stage_target_words']):
            outputs.append(dict(metric=metric,method=method,stage_target_words=stage,**estimate(sub.set_index('article_id')[metric])))
    if metrics_path.exists():
        metrics=pd.DataFrame(read_csv(metrics_path));metrics['article_id']=metrics.article_id.astype(int)
        metrics['stage_target_words']=metrics.stage_target_words.astype(int)
        if 'method' not in metrics:metrics['method']=metrics['prompt_variant']
        for metric in ['highlight_rouge1_f1','highlight_rouge2_f1','highlight_rougeL_f1','highlight_bertscore_f1','source_summac','stage_input_summac']:
            if metric not in metrics:continue
            metrics[metric]=pd.to_numeric(metrics[metric],errors='coerce')
            for (method,stage),sub in metrics.groupby(['method','stage_target_words']):
                outputs.append(dict(metric=metric,method=method,stage_target_words=stage,**estimate(sub.set_index('article_id')[metric])))
    write_csv(output/'mittelwerte_95ci.csv',outputs)
    report=dict(source='frozen' if frozen else 'new_run',n_articles=int(length.article_id.nunique()),summary_positions=len(length),qa_positions=len(rows),
        bootstrap_draws=10000,seed=32,unit='article',new_model_calls=0,
        human_validation='historical ratings only in frozen detailed analysis',
        regression_bootstrap='historical detailed analysis in 05_analyze; not estimated from a mini test')
    (output/'analysis_manifest.json').write_text(json.dumps(report,indent=2))
    print('Gemeinsame Artikelstatistik: '+str(output))
    return output
