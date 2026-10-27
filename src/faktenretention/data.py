"""Benötigte Rohdaten vorbereiten; gemeinsame QA-, Längen- und Artikelregeln."""
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np
import pandas as pd
from faktenretention.io import PROJECT_ROOT
from faktenretention.protocol import METHODS, STAGES, WINDOWS
from faktenretention.text_rules import count_words
from faktenretention import scoring

CELL = ['article_id', 'method', 'stage_target_words']
KEY = CELL + ['canonical_fact_id']
FILES = {'artikel.csv.gz': 'articles.csv', 'fakten.csv.gz': 'facts.csv',
         'zusammenfassungen.csv.gz': 'summaries.csv', 'qa_antworten.csv.gz': 'answers.csv',
         'modellmetriken.csv': 'metrics.csv', 'generierungszustaende.jsonl.gz': 'states.jsonl'}


def verify_sources():
    inventory = json.loads((PROJECT_ROOT / 'data/quellen.json').read_text())
    for entry in inventory['files']:
        path = PROJECT_ROOT / entry['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Quelle verändert: ' + entry['file'])
    return len(inventory['files'])


def prepare(destination=None):
    """Genau die sechs gemeinsamen Eingabedateien entpacken, keine Ergebnistabellen."""
    verify_sources()
    destination = Path(destination or PROJECT_ROOT / 'data/prepared')
    destination.mkdir(parents=True, exist_ok=True)
    for source, target in FILES.items():
        path = PROJECT_ROOT / 'data' / source
        with (gzip.open(path, 'rb') if source.endswith('.gz') else path.open('rb')) as stream:
            content = stream.read()
        output = destination / target
        if output.exists() and output.read_bytes() != content:
            raise ValueError('Vorbereitete Eingabe verändert: ' + str(output))
        output.write_bytes(content)
    return destination


def load_inputs(directory=None):
    directory = Path(directory or PROJECT_ROOT / 'data/prepared')
    if not all((directory / name).is_file() for name in FILES.values()):
        raise FileNotFoundError('Eingaben fehlen. Einmal im Repository: python 05_analyze.py --prepare')
    frames = {name: pd.read_csv(directory / (name + '.csv'), keep_default_na=False,
                               low_memory=False) for name in ['articles','facts','summaries','answers','metrics']}
    frames['states'] = [json.loads(line) for line in (directory/'states.jsonl').read_text().splitlines()]
    assert len(frames['articles']) == 100 and len(frames['facts']) == 2000
    assert len(frames['summaries']) == 600 and len(frames['answers']) == 48000
    assert len(frames['metrics']) == 2400 and len(frames['states']) == 600
    for row in frames['articles'].itertuples():
        assert hashlib.sha256(row.article.encode()).hexdigest() == row.article_sha256
    expected = {m:'plain_fewshot_rerun_20260930' if m in ['plain','few_shot_cot'] else
                'final_100_frozen_temperature_0' for m in METHODS}
    assert frames['summaries'].collection_phase.eq(frames['summaries'].method.map(expected)).all()
    return frames


def length_values(inputs):
    """Rekursive Eingaben und gespeicherte Zustände prüfen; W, Wortfenster und D."""
    lookup = inputs['summaries'].set_index(['article_id','method'])
    records=[]
    for rec in inputs['states']:
        state=rec['state'];row=lookup.loc[(rec['article_id'],rec['method'])]
        assert state['complete'] and state['source_sha256'] == row.article_sha256
        assert [s['target_words'] for s in state['stages']] == STAGES
        previous=row.article;chain_failure=False
        for stage in state['stages']:
            target=stage['target_words'];selected=stage['selected'];text=selected['response']
            assert text == row[f'summary_{target}'] and stage['input_text'] == previous
            W=count_words(text);U,O=WINDOWS[target]
            assert W == int(row[f'summary_{target}_word_count']) == selected['response_word_count']
            assert count_words(previous) == stage['input_word_count']
            under,over=max(U-W,0),max(W-O,0);D=(under+over)/target
            hit=U<=W<=O;empty=not text.strip();chain_failure=chain_failure or empty
            assert selected['word_window_success'] == hit
            if 'word_window_violation_rate' in selected:
                assert selected['word_window_violation_words'] == under+over
                assert np.isclose(selected['word_window_violation_rate'],D,rtol=0,atol=1e-12)
            records.append(dict(article_id=rec['article_id'],method=rec['method'],stage_target_words=target,
                collection_phase=row.collection_phase,summary=text,summary_sha256=hashlib.sha256(text.encode()).hexdigest(),
                W=W,U=U,O=O,under_words=under,over_words=over,violation_words=under+over,D=D,
                under_D=under/target,over_D=over/target,window_hit=hit,
                stored_D_available='word_window_violation_rate' in selected,
                window_status='under' if under else 'over' if over else 'within',
                empty_output=empty,empty_input=not previous.strip(),chain_failure=chain_failure,
                complete=state['complete'],complete_sentence=selected['ends_with_complete_sentence'],
                attempts=len(stage['attempts']),done_reason=selected.get('done_reason','deterministic'),
                source_state_file=rec['source_file'],input_words=count_words(previous),
                input_sha256=hashlib.sha256(previous.encode()).hexdigest(),
                output_sha256=hashlib.sha256(text.encode()).hexdigest(),stage_input=previous))
            previous=text
    frame=pd.DataFrame(records)
    assert len(frame)==2400 and not frame.duplicated(CELL).any()
    assert frame.attempts.eq(1).all() and frame.empty_output.sum()==2 and frame.empty_input.sum()==1
    return frame


def reconstruct_qa(inputs, lengths):
    """Unveränderter QA-Scorer: technische Fehler getrennt von NOT_FOUND halten."""
    qa=inputs['answers'].rename(columns={'summary_prompt_variant':'method'}).copy()
    facts=inputs['facts']
    assert facts.canonical_fact_id.is_unique and facts.groupby('article_id').size().eq(20).all()
    assert not qa.duplicated(KEY).any()
    expected={(a,m,t,f) for a,fs in facts.groupby('article_id') for m in METHODS for t in STAGES
              for f in fs.canonical_fact_id}
    assert set(qa[KEY].itertuples(index=False,name=None))==expected
    canonical=qa.merge(facts[['article_id','canonical_fact_id','question','reference_answer']],
                       on=['article_id','canonical_fact_id'],validate='many_to_one',suffixes=('','_fact'))
    for field in ['question','reference_answer']:
        assert canonical[field].eq(canonical[field+'_fact']).all()
    qa=qa.merge(lengths[CELL+['summary','empty_output','empty_input','chain_failure']],on=CELL,validate='many_to_one')
    qa['technical_missing']=qa.error.astype(str).ne('')
    qa['analysis_eligible']=~qa.technical_missing
    qa['grounded']=[scoring.support_is_grounded(s,a,e) for s,a,e in zip(qa.summary,qa.answer,qa.support)]
    assert qa.loc[qa.analysis_eligible,'grounded'].all()
    qa['retained']=[scoring.retained(ref,answer) if ok else False
                    for ref,answer,ok in zip(qa.reference_answer,qa.answer,qa.analysis_eligible)]
    qa['lost']=qa.analysis_eligible & qa.answer.str.upper().eq('NOT_FOUND')
    qa['reference_discordant']=qa.analysis_eligible & ~qa.retained & ~qa.lost
    qa['label']=np.select([qa.technical_missing,qa.retained,qa.lost],
                         ['technical_missing','retained','lost'],default='reference_discordant')
    qa['retention_value']=np.where(qa.analysis_eligible,qa.retained.astype(float),np.nan)
    qa['answer_origin']=np.where(qa.empty_output,'empty_summary_rule','qa_model')
    assert qa.technical_missing.sum()==196 and qa.analysis_eligible.sum()==47804
    assert qa.empty_output.sum()==40 and qa.loc[qa.empty_output,'answer'].eq('NOT_FOUND').all()
    assert not qa.loc[qa.empty_output,'technical_missing'].any()
    return qa


def aggregate_articles(qa):
    article=qa.groupby(CELL).agg(n_total=('canonical_fact_id','size'),n_valid=('analysis_eligible','sum'),
        n_missing=('technical_missing','sum'),n_retained=('retained','sum'),n_lost=('lost','sum'),
        generation_chain_failure=('chain_failure','any'),n_synthetic_not_found=('empty_output','sum'),
        n_reference_discordant=('reference_discordant','sum')).reset_index()
    assert len(article)==2400 and article.n_total.eq(20).all()
    article['retention_observed']=article.n_retained.div(article.n_valid.replace(0,np.nan))
    article['retention_lower']=article.n_retained/20
    article['retention_upper']=(article.n_retained+article.n_missing)/20
    article['fully_observed']=article.n_missing.eq(0)&~article.generation_chain_failure
    return article


def panel(inputs):
    """Die wiederverwendete Artikel-Verfahren-Stufen-Tabelle aus Rohdaten berechnen."""
    lengths=length_values(inputs);qa=reconstruct_qa(inputs,lengths)
    article=lengths.merge(aggregate_articles(qa),on=CELL,validate='one_to_one')
    article['V']=article.n_valid;article['K']=article.n_retained
    article['R_obs']=article.retention_observed;article['R_zero']=article.retention_lower
    article['R_adj']=article.R_zero/(1+article.D)
    article['qa_denominator_gap']=article.R_obs-article.R_zero
    article['length_penalty']=article.R_zero-article.R_adj
    return qa,article
