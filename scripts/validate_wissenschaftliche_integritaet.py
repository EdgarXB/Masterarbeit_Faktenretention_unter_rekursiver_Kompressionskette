"""Prüft eingefrorene Experimentdaten ohne Modellaufrufe; python scripts/validate_wissenschaftliche_integritaet.py."""
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import zipfile

import numpy as np
import pandas as pd

from faktenretention.data import prepare, load_inputs, panel, CELL, KEY
from faktenretention.protocol import METHODS, STAGES, CONTRASTS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
sha = lambda b: hashlib.sha256(b).hexdigest()
result = {}

with tempfile.TemporaryDirectory(prefix='integritaetspruefung_') as directory:
    prepare(directory)
    inputs = load_inputs(directory)
    qa, cells = panel(inputs)
articles, facts, summaries, answers = (inputs[k] for k in ('articles', 'facts', 'summaries', 'answers'))
article_ids = set(articles.article_id)
assert len(article_ids) == 100 and not articles.article_id.duplicated().any()
assert not articles.duplicated(['dataset', 'config', 'split', 'dataset_index']).any()
assert not facts.duplicated(['article_id', 'canonical_fact_id']).any()
assert facts.groupby('article_id').size().eq(20).all()
assert set(facts.article_id) == article_ids
expected = {(a, m) for a in article_ids for m in METHODS}
assert set(summaries[['article_id', 'method']].itertuples(index=False, name=None)) == expected
assert not summaries.duplicated(['article_id', 'method']).any()
assert set(cells[CELL].itertuples(index=False, name=None)) == {(a,m,t) for a,m in expected for t in STAGES}
assert len(qa) == 48000 and not qa.duplicated(KEY).any()
keys = summaries.merge(articles[['article_id', 'dataset_index', 'article_sha256']], on='article_id', suffixes=('', '_source'), validate='many_to_one')
assert keys.dataset_index.eq(keys.dataset_index_source).all() and keys.article_sha256.eq(keys.article_sha256_source).all()
assert facts.merge(articles[['article_id','dataset_index']],on='article_id',suffixes=('','_source')).eval('dataset_index == dataset_index_source').all()
result['coverage'] = {'articles': len(article_ids), 'facts':len(facts),'summaries':len(cells),'qa_positions':len(qa),'duplicates':0}

# Check each serialized recursive stage independently of the analysis panel.
lookup = summaries.set_index(['article_id','method'])
stages_checked = Counter()
for rec in inputs['states']:
    row = lookup.loc[(rec['article_id'], rec['method'])]
    state = rec['state']
    assert len(state['stages']) == 4 and state['source_sha256'] == row.article_sha256
    previous = None
    for stage, target in zip(state['stages'], STAGES):
        assert stage['target_words'] == target
        assert stage['selected']['response'] == row[f'summary_{target}']
        if previous is None:
            assert stage['input_text'] == '[ORIGINALTEXT ENTFERNT]'
        else:
            assert stage['input_text'] == previous
        stages_checked[str(target)] += 1
        previous = stage['selected']['response']
result['recursive_stages'] = dict(stages_checked)
result['original_input_limit'] = 'Alle 600 ersten Eingaben entfernt; Wortzahl und SHA-256 vorhanden, direkter Textvergleich nicht möglich.'

rerun = {'plain','few_shot_cot'}
protocol = json.loads((DATA/'methoden/protocol.json').read_text())
result_manifest = json.loads((DATA/'methoden/result_merge_manifest.json').read_text())
qa_manifest = json.loads((DATA/'methoden/qa_merge_manifest.json').read_text())
assert (result_manifest['retained_historical_rows'],result_manifest['replaced_plain_rows'],result_manifest['replaced_few_shot_rows'],result_manifest['historical_plain_or_few_shot_rows_in_new_table']) == (400,100,100,0)
assert (qa_manifest['retained_historical_units'],qa_manifest['new_plain_units'],qa_manifest['new_few_shot_units']) == (32000,8000,8000)
assert summaries.loc[summaries.method.isin(rerun),'collection_phase'].eq('plain_fewshot_rerun_20260930').all()
assert summaries.loc[~summaries.method.isin(rerun),'collection_phase'].eq('final_100_frozen_temperature_0').all()
for rec in inputs['states']:
    is_new = rec['method'] in rerun
    assert (rec['state']['run_signature'] == protocol['run_signature']) == is_new
    assert ('plain_fewshot_rerun_20260930' in rec['source_file']) == is_new
for method, part in answers.groupby('summary_prompt_variant'):
    new = method in rerun
    assert part.experiment_run_id.eq('plain_fewshot_rerun_20260930' if new else 'six_method_final_100').all()
    assert part.summary_batch_run_id.eq('rerun_only' if new else 'recursive_results').all()
    assert part.answer_batch_run_id.eq(('plain_fewshot_rerun_20260930' if new else 'six_method_final_100')+'__thesis_qwen38_qa_v1__summary_qa').all()
    assert set(part.article_id) == article_ids and len(part)==8000
assert answers.merge(articles[['article_id','dataset_index']],on='article_id',suffixes=('','_article'),validate='many_to_one').eval('dataset_index == dataset_index_article').all()
result['merge'] = {'historical_summary_rows':400,'rerun_summary_rows':200,'historical_qa':32000,'rerun_qa':16000,'method_article_keys_match':True}

# Check prompt checksums and split separation using the stored original-text hashes.
bank=json.loads((ROOT/'prompts/bank_manifest.json').read_text())
for method, stages in bank['methods'].items():
    assert set(stages) == set(map(str, STAGES))
    for t,e in stages.items():
        assert sha((ROOT/'prompts'/e['path']).read_bytes()) == e['sha256']
assert sha((ROOT/'prompts/final_few_shot_cot_manifest.json').read_bytes()) == protocol['fewshot_manifest_sha256']
for t,h in protocol['fewshot_prompt_sha256_by_stage'].items():
    assert sha((ROOT/f'prompts/final_few_shot_cot_{t}.txt').read_bytes()) == h
few=json.loads((ROOT/'prompts/final_few_shot_cot_manifest.json').read_text())
with zipfile.ZipFile(DATA/'demonstrationen.zip') as z:
    def read_csv(name):
        return pd.read_csv(z.open(name),keep_default_na=False)
    train=read_csv('data/candidate_train.csv');valid=read_csv('data/selection_validation.csv');test=read_csv('data/final_test_100.csv')
    assert (len(train),len(valid),len(test))==(240,30,100)
    for frame,split in [(train,'train'),(valid,'validation'),(test,'test')]:
        assert frame.split.eq(split).all() and frame.article_sha256.is_unique
    splits=[set(x.article_sha256) for x in (train,valid,test)]
    assert all(not splits[i]&splits[j] for i in range(3) for j in range(i+1,3))
    train_hash_by_index=dict(zip(train.dataset_index.astype(str),train.article_sha256))
    valid_hash_by_index=dict(zip(valid.dataset_index.astype(str),valid.article_sha256))
    match=test.merge(articles,on=['article_id','dataset_index','article_sha256','split'],validate='one_to_one')
    assert len(match)==100
    records={}
    def check_prompt_examples(prompt, selected, target):
        inputs=re.findall(r'INPUT:\n(.*?)\n\nSTEP-BY-STEP REASONING:',prompt,re.S)
        assert len(inputs)>=5
        for text,row in zip(inputs[:5],selected):
            if target==500:
                assert sha(text.encode())==row['article_sha256']
            else:
                assert text==row['input_text']
            assert row['thinking'].strip() in prompt and row['response'].strip() in prompt
    for method,name in [('auto_cot','auto_cot_selection/auto_cot_demonstrations.json'),('automate_cot','automate_cot_selection/automate_cot_demonstrations.json')]:
        data=json.loads(z.read(name))
        for t in STAGES:
            selected=data[str(t)];assert len(selected)==5
            prompt=(ROOT/'prompts'/f'{method}_{t}.txt').read_text()
            check_prompt_examples(prompt,selected,t)
            positions=[prompt.find(x['response'].strip()) for x in selected]
            assert min(positions)>=0 and positions==sorted(positions) and len(set(positions))==5
            for x in selected:
                assert x['split']=='train' and x['article_sha256'] in splits[0] and x['article_sha256'] not in splits[1]|splits[2]
                assert int(x['target_words'])==t
                assert x['candidate_id']==f"train:{x['dataset_index']}:{t}" and train_hash_by_index[str(x['dataset_index'])]==x['article_sha256']
            records[f'{method}_{t}']=[x['candidate_id'] for x in selected]
    pools=json.loads(z.read('automate_cot_pruning/automate_cot_pruned_pools.json'))
    for t in STAGES:
        ids=few['stages'][str(t)]['ordered_selected_candidate_ids']
        pool={x['candidate_id']:x for x in pools[str(t)]}
        assert len(ids)==5 and len(set(ids))==5
        for ident in ids:
            x=pool[ident];assert x['split']=='train' and x['article_sha256'] in splits[0] and x['article_sha256'] not in splits[1]|splits[2]
            assert ident==f"train:{x['dataset_index']}:{t}" and train_hash_by_index[str(x['dataset_index'])]==x['article_sha256']
        prompt=(ROOT/'prompts'/f'final_few_shot_cot_{t}.txt').read_text()
        check_prompt_examples(prompt,[pool[ident] for ident in ids],t)
        positions=[prompt.find(pool[ident]['response'].strip()) for ident in ids]
        assert min(positions)>=0 and positions==sorted(positions) and len(set(positions))==5
        records[f'few_shot_cot_{t}']=ids
    val=[json.loads(line) for line in z.read('validation_rollouts/stage_candidates.jsonl').decode().splitlines()]
    assert all(x['split']=='validation' and valid_hash_by_index[str(x['dataset_index'])]==x['article_sha256'] for x in val)
result['split']={'train':len(train),'validation':len(valid),'test':len(test),'cross_split_hash_overlaps':0,'validation_candidate_rows':len(val),'demonstration_ids':records,'prompt_hashes_verified':12}

# Recount categories and denominators directly from the position-level labels.
errors=answers.error.ne('')
assert qa.technical_missing.eq(errors).all()
assert qa.loc[qa.technical_missing,'label'].eq('technical_missing').all()
assert not qa.loc[qa.technical_missing,['retained','lost','reference_discordant']].any().any()
assert qa.loc[~qa.technical_missing,['retained','lost','reference_discordant']].sum(axis=1).eq(1).all()
assert qa.loc[qa.empty_output,'answer'].str.upper().eq('NOT_FOUND').all()
counts=qa.groupby(CELL).agg(total=('label','size'),valid=('analysis_eligible','sum'),retained=('retained','sum'),missing=('technical_missing','sum')).reset_index()
check=counts.merge(cells,on=CELL,validate='one_to_one')
assert check.total.eq(20).all() and check.missing.eq(check.n_missing).all()
assert check.valid.eq(check.n_valid).all() and check.retained.eq(check.n_retained).all()
assert np.allclose(check.R_zero,check.retained/20)
assert np.allclose(check.R_obs,check.retained/check.valid, equal_nan=True)
result['labels']={'counts':{str(k):int(v) for k,v in qa.label.value_counts().items()},'empty_output_qa':int(qa.empty_output.sum()),'technical_missing_by_method':{str(k):int(v) for k,v in qa.groupby('method').technical_missing.sum().items()}}

# Independently recompute the reported 24 means and 16 paired article-bootstrap contrasts.
mean_ref=pd.read_csv(ROOT/'results/02_faktenretention/retention_mittelwerte_95ci.csv')
contrast_ref=pd.read_csv(ROOT/'results/02_faktenretention/vier_gepaarte_kontraste_pp_95ci.csv')
def bootstrap(v):
    v=v.sort_index().dropna();rng=np.random.default_rng(32)
    draw=rng.integers(0,len(v),size=(10000,len(v)))
    means=v.to_numpy()[draw].mean(axis=1); return np.array([v.mean(),*np.quantile(means,[.025,.975])])
for m in METHODS:
    for t in STAGES:
        v=check.loc[check.method.eq(m)&check.stage_target_words.eq(t)].set_index('article_id').eval('retained / valid')
        ref=mean_ref.loc[mean_ref.method.eq(m)&mean_ref.stage_target_words.eq(t)].iloc[0]
        assert np.allclose(bootstrap(v),ref[['mean','ci_low','ci_high']].to_numpy(dtype=float),rtol=0,atol=1e-12)
for a,b in CONTRASTS:
    for t in STAGES:
        sub=check[check.stage_target_words.eq(t)];wide=sub.pivot(index='article_id',columns='method',values='R_obs')
        assert set(wide.index)==article_ids
        diff=100*(wide[a]-wide[b]);ref=contrast_ref.loc[contrast_ref['first'].eq(a)&contrast_ref['second'].eq(b)&contrast_ref.stage_target_words.eq(t)].iloc[0]
        assert np.allclose(bootstrap(diff),ref[['mean','ci_low','ci_high']].to_numpy(dtype=float),rtol=0,atol=1e-12)
result['independent_bootstrap']={'means':24,'paired_contrasts':16,'draws':10000,'seed':32,'sampling_unit':'article','all_reported_values_match':True}

# Denominator sensitivity and the 100/98-article length-adjusted comparisons
# must use the same final panel and exclude complete articles consistently.
denominator_ref=pd.read_csv(ROOT/'results/02_faktenretention/nennervergleich_k_v_k_20.csv')
for (method,target), group in cells.groupby(['method','stage_target_words']):
    ref=denominator_ref.loc[denominator_ref.method.eq(method)&denominator_ref.stage_target_words.eq(target)].iloc[0]
    actual=np.array([group.retention_observed.mean(),group.retention_lower.mean(),group.n_missing.mean()])
    assert np.allclose(actual,ref[['retention_observed','retention_lower','n_missing']].to_numpy(dtype=float),rtol=0,atol=1e-12)
excluded=set(cells.loc[cells.empty_output,'article_id'])
assert len(excluded)==2
sensitivity_ref=pd.read_csv(ROOT/'results/03_textlaenge/berichtete_adjustierte_abstaende_100_98.csv')
for row in sensitivity_ref.itertuples():
    sub=cells[cells.stage_target_words.eq(row.stage_target_words)]
    if row.n_articles==98:sub=sub[~sub.article_id.isin(excluded)]
    wide=sub.pivot(index='article_id',columns='method',values='R_adj')
    assert len(wide)==row.n_articles and set(wide.index)==(article_ids-excluded if row.n_articles==98 else article_ids)
    actual=bootstrap(100*(wide[row.first]-wide[row.second]))
    assert np.allclose(actual,[row.difference_pp,row.ci_low_pp,row.ci_high_pp],rtol=0,atol=1e-12)
result['sensitivity']={'denominator_cells':24,'adjusted_comparisons':len(sensitivity_ref),'excluded_article_ids':sorted(map(int,excluded)),'remaining_paired_articles':98,'all_match':True}
length_ref=pd.read_csv(ROOT/'results/03_textlaenge/retention_und_drei_rangfolgen_24_gruppen.csv')
for (method,target),group in cells.groupby(['method','stage_target_words']):
    ref=length_ref.loc[length_ref.method.eq(method)&length_ref.stage_target_words.eq(target)].iloc[0]
    assert int(ref.n_articles)==100 and int(ref.n_missing_qa)==int(group.n_missing.sum())
    for metric in ['R_obs','R_zero','R_adj']:
        observed=bootstrap(group.set_index('article_id')[metric])
        expected=ref[[f'{metric}_mean',f'{metric}_ci_low',f'{metric}_ci_high']].to_numpy(dtype=float)
        assert np.allclose(observed,expected,rtol=0,atol=1e-12)
result['length_retention']={'method_stage_groups':24,'metrics_per_group':3,'article_bootstrap_matches':True}

scored=DATA/'prepared/qa/aktuell_20260930/scored_archived.csv.gz'
if scored.exists():
    archive=pd.read_csv(scored,keep_default_na=False,low_memory=False).rename(columns={'summary_prompt_variant':'method'})
    merged=qa.merge(archive,on=KEY,suffixes=('_recomputed','_archive'),validate='one_to_one')
    assert len(merged)==48000
    assert merged.technical_missing_recomputed.eq(merged.technical_missing_archive).all()
    eligible=merged[~merged.technical_missing_recomputed]
    for recomputed,stored in [('retained','retention'),('lost','loss'),('reference_discordant','reference_discordant_answer')]:
        assert eligible[recomputed].astype(str).eq(eligible[stored].astype(str)).all()
    result['local_archived_qa_labels']={'distributed':False,'positions_compared':len(merged),'disagreements':0}

# An optional local historical snapshot supports a direct comparison but is not distributed.
historical=DATA/'prepared/qa/original_frozen'
if (historical/'summaries.csv.gz').exists() and (historical/'answers.csv.gz').exists():
    old_s=pd.read_csv(historical/'summaries.csv.gz',keep_default_na=False,low_memory=False)
    old_a=pd.read_csv(historical/'answers.csv.gz',keep_default_na=False,low_memory=False)
    fields=[f'summary_{t}' for t in STAGES]
    joined=summaries.merge(old_s,on=['article_id','method'],suffixes=('_new','_old'),validate='one_to_one')
    changed=joined[[f'{f}_new' for f in fields]].to_numpy()!=joined[[f'{f}_old' for f in fields]].to_numpy()
    assert not changed[~joined.method.isin(rerun)].any()
    comp=answers.merge(old_a,on=['article_id','summary_prompt_variant','stage_target_words','canonical_fact_id'],suffixes=('_new','_old'),validate='one_to_one')
    cols=['answer','support','answer_status','error']
    same=np.ones(len(comp),dtype=bool)
    for col in cols:same &= comp[col+'_new'].astype(str).eq(comp[col+'_old'].astype(str)).to_numpy()
    assert same[~comp.summary_prompt_variant.isin(rerun)].all()
    result['local_historical_comparison']={'distributed':False,'historical_other_method_summary_changes':0,'historical_other_method_qa_changes':0,'rerun_changed_summary_cells':int(changed[joined.method.isin(rerun)].sum()),'rerun_changed_qa_rows':int((~same[comp.summary_prompt_variant.isin(rerun)]).sum())}
else:
    result['local_historical_comparison']={'distributed':False,'available':False}
print(json.dumps(result,indent=2,ensure_ascii=False))
