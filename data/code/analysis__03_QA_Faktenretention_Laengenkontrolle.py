# Statische Originalauswertung; Eingaben/Ausgaben relativ zur übergebenen Arbeitskopie.
# Keine LLM-Aufrufe. Standard: vorhandene Regressionscaches verwenden.
from pathlib import Path
import os,sys

def main(work, recompute_bootstrap=False, workers=1):
    WORK=__import__('pathlib').Path(work).resolve()
    __import__('os').chdir(WORK/('supplement' if '03_QA_Faktenretention_Laengenkontrolle'[0].isdigit() else 'project/02_Auswertung'))
    from pathlib import Path
    from fractions import Fraction
    from collections import Counter, defaultdict
    from itertools import combinations
    import ast, gzip, hashlib, html, importlib.metadata, json, math, os, platform, re, sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, MaxNLocator
    from IPython.display import display, Markdown, HTML

    sys.dont_write_bytecode = True
    BUNDLE = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p/'daten/paket_manifest.json').is_file()), None)
    assert BUNDLE is not None, 'Notebook aus dem entpackten Analysepaket starten.'
    PACKAGE = WORK/'project'
    ROOT = BUNDLE/'ergebnisse/03_laengenkontrolle'
    assert ROOT != PACKAGE and PACKAGE not in ROOT.parents, 'Ausgaben außerhalb des Quellpakets speichern.'
    DATA = PACKAGE/'02_Auswertung/hauptexperiment_qa_daten'
    ORIGINAL = PACKAGE/'02_Auswertung/hauptexperiment_qa_ergebnisse/aktuell_20260930'
    LENGTH = PACKAGE/'02_Auswertung/hauptexperiment_laengen_ergebnisse/aktuell_20260930'
    OUT = ROOT
    FIG = OUT/'abbildungen'
    FIG.mkdir(parents=True,exist_ok=True)
    VERSION = 'aktuell_20260930'
    METHODS = ['lead_k','plain','zero_shot_cot','few_shot_cot','auto_cot','automate_cot']
    NAMES = dict(zip(METHODS,['Lead-k','Plain','Zero-Shot-CoT','Few-Shot-CoT','Auto-CoT','Automate-CoT']))
    STAGES = [500,300,150,75]
    CONTRASTS = [('zero_shot_cot','plain'),('few_shot_cot','plain'),
                 ('auto_cot','few_shot_cot'),('automate_cot','auto_cot')]
    KEY = ['article_id','method','stage_target_words','canonical_fact_id']
    CELL = ['article_id','method','stage_target_words']
    SEED, B = 32, 10_000
    GENERATIVE = METHODS[1:]
    MIN_SUPPORT = 20
    WORKERS = workers
    USE_BOOTSTRAP_CACHE = True
    pd.set_option('display.max_rows',28)
    pd.set_option('display.max_columns',16)
    plt.rcParams.update({'figure.dpi':110,'savefig.dpi':200,'font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False})
    STAGE_COLORS = {500:'#0072B2',300:'#D55E00',150:'#009E73',75:'#CC79A7'}

    def export(frame,name):
        frame.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')

    def savefig(fig,name):
        for extension in ['png','svg','pdf']:
            fig.savefig(FIG/f'{name}.{extension}',bbox_inches='tight')
        plt.show()
        plt.close(fig)

    def pct(n,d): return 100*n/d if d else np.nan
    def de(x,digits=2): return f'{x:.{digits}f}'.replace('.',',') if pd.notna(x) else 'nicht bestimmbar'
    def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def read(path,strings=False):
        frame = pd.read_csv(path,keep_default_na=False,**({'dtype':str} if strings else {}))
        for column in ['article_id','stage_target_words']:
            if column in frame: frame[column] = frame[column].astype(int)
        return frame

    software = {'python':platform.python_version(),**{m:importlib.metadata.version(m)
                for m in ['numpy','pandas','scipy','statsmodels','patsy','matplotlib','nbformat','nbclient','nbconvert','ipykernel']}}
    display(pd.Series(software,name='Version').to_frame())
    import time, warnings, inspect, importlib
    import scipy.linalg as la
    from scipy.special import expit
    from scipy.stats import t as student_t
    import patsy
    import statsmodels.api as sm
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    METHOD_COLORS = dict(zip(METHODS,['#777777','#0072B2','#E69F00','#009E73','#CC79A7','#D55E00']))

    manifest = json.loads((DATA/'source_manifest.json').read_text())
    assert manifest['default_version']==VERSION
    manifest_entries = {x['file']:x for x in manifest['files']}
    source_paths = [DATA/'source_manifest.json']
    relative_inputs = ['shared/canonical_facts.csv.gz', f'{VERSION}/summaries.csv.gz',
                       f'{VERSION}/answers.csv.gz', f'{VERSION}/scored_archived.csv.gz',
                       'code/qa_scorer_snapshot.py','code/score_model_selection_batch.py']
    for relative in relative_inputs:
        path = DATA/relative
        assert sha(path)==manifest_entries[relative]['sha256'],path
        if path.suffix=='.gz':
            assert hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest()==manifest_entries[relative]['original_sha256']
        source_paths.append(path)
    source_paths += [ORIGINAL/name for name in ['retention_artikelebene.csv','retention_mittelwerte_95ci.csv',
        'vier_gepaarte_kontraste_pp_95ci.csv','qa_labels_reproduziert.csv.gz']]
    source_paths += [LENGTH/name for name in ['artikelwerte_laenge_retention.csv',
        'retention_und_drei_rangfolgen_24_gruppen.csv','vollstaendige_24_gruppen_main_100.csv']]
    source_paths += [PACKAGE/'02_Auswertung/hauptexperiment_01_qa_faktenretention.quellnotebook.json',
        PACKAGE/'02_Auswertung/hauptexperiment_02_laengeneinhaltung.quellnotebook.json',
        PACKAGE/'03_Experiment/compute_length_adjusted_retention.py',
        PACKAGE/'03_Experiment/iterative_summarize.py',PACKAGE/'README_START.md']
    sources_before = {str(p):sha(p) for p in source_paths}
    sources = pd.DataFrame([{'Datei':str(p.relative_to(PACKAGE)),'sha256':sources_before[str(p)]}
                            for p in source_paths])
    export(sources,'quellen_sha256')
    display(sources[['Datei']])

    facts = read(DATA/'shared/canonical_facts.csv.gz',strings=True)
    summaries = read(DATA/VERSION/'summaries.csv.gz',strings=True)
    qa = read(DATA/VERSION/'scored_archived.csv.gz',strings=True).rename(columns={'summary_prompt_variant':'method'})
    raw = read(DATA/VERSION/'answers.csv.gz',strings=True).rename(columns={'summary_prompt_variant':'method'})
    assert len(facts)==2000 and facts.canonical_fact_id.is_unique
    assert facts.groupby('article_id').size().eq(20).all()
    IDS = sorted(facts.article_id.unique().tolist())
    assert len(IDS)==100 and len(summaries)==600
    assert not summaries.duplicated(['article_id','method']).any()
    assert len(qa)==len(raw)==48000 and not qa.duplicated(KEY).any()
    expected = {(a,m,t,f) for a,fs in facts.groupby('article_id') for f in fs.canonical_fact_id
                for m in METHODS for t in STAGES}
    assert set(qa[KEY].itertuples(index=False,name=None))==expected
    assert set(raw[KEY].itertuples(index=False,name=None))==expected
    joined = qa.merge(raw,on=KEY,suffixes=('_arch','_raw'),validate='one_to_one')
    for field in ['question','reference_answer','answer','support','error','answer_status']:
        assert joined[field+'_arch'].eq(joined[field+'_raw']).all(),field
    fact_join = qa.merge(facts[['article_id','canonical_fact_id','question','reference_answer']],
        on=['article_id','canonical_fact_id'],validate='many_to_one',suffixes=('','_fact'))
    assert fact_join.question.eq(fact_join.question_fact).all()
    assert fact_join.reference_answer.eq(fact_join.reference_answer_fact).all()
    qa['label'] = np.select([qa.technical_missing.eq('True'),qa.retention.eq('True'),
        qa.loss.eq('True'),qa.reference_discordant_answer.eq('True')],
        ['technical_missing','retained','lost','reference_discordant'],default='INVALID')
    assert not qa.label.eq('INVALID').any()
    qa['valid'] = qa.label.ne('technical_missing')
    qa['retained_binary'] = qa.label.eq('retained').astype(int)
    assert (~qa.valid).sum()==196
    assert qa.valid.eq(qa.analysis_eligible.eq('True')).all()
    assert (~qa.valid).eq(qa.error.ne('')).all()
    stored_labels = read(ORIGINAL/'qa_labels_reproduziert.csv.gz',strings=True)
    label_check = qa[KEY+['label']].merge(stored_labels[KEY+['label']],on=KEY,
        validate='one_to_one',suffixes=('','_existing'))
    assert label_check.label.eq(label_check.label_existing).all()
    original_label_digest = hashlib.sha256(qa[KEY+['label']].to_csv(index=False).encode()).hexdigest()

    article = qa.groupby(CELL).agg(n_total=('valid','size'),n_valid=('valid','sum'),
        n_retained=('retained_binary','sum'),n_missing=('technical_missing',lambda s:s.eq('True').sum()),
        n_discordant=('label',lambda s:s.eq('reference_discordant').sum())).reset_index()
    assert len(article)==2400 and article.n_total.eq(20).all() and article.n_valid.gt(0).all()
    article['retention_observed'] = article.n_retained/article.n_valid
    article['retention_lower'] = article.n_retained/20
    reference_article = read(ORIGINAL/'retention_artikelebene.csv')
    check = article.merge(reference_article,on=CELL,validate='one_to_one',suffixes=('','_archive'))
    for field in ['n_retained','n_valid','n_missing','retention_observed','retention_lower']:
        assert np.allclose(check[field],check[field+'_archive'],rtol=0,atol=1e-12),field
    coverage = pd.DataFrame({'Einheit':['Artikel','Fakten','Zusammenfassungen','QA-Positionen','Gültige QA-Urteile','Technische Fehlwerte'],
        'Anzahl':[len(IDS),len(facts),len(summaries)*4,len(qa),int(qa.valid.sum()),int((~qa.valid).sum())]})
    display(coverage)
    export(coverage,'eingabeabdeckung')

    from analysis_statistics import estimate, means_for, contrasts_for

    original_means = means_for(article)
    original_contrasts = contrasts_for(article)
    mean_reference = read(ORIGINAL/'retention_mittelwerte_95ci.csv')
    contrast_reference = read(ORIGINAL/'vier_gepaarte_kontraste_pp_95ci.csv')
    checks = []
    for title,new,old,keys in [
        ('24 Mittelwerte und Intervalle',original_means,mean_reference,['method','stage_target_words']),
        ('16 gepaarte Differenzen und Intervalle',original_contrasts,contrast_reference,['first','second','stage_target_words'])]:
        paired = new.merge(old,on=keys,validate='one_to_one',suffixes=('_new','_reference'))
        for field in ['n_articles','mean','ci_low','ci_high']:
            delta = np.abs(paired[field+'_new']-paired[field+'_reference'])
            checks.append({'Abgleich':title,'Feld':field,'max_abs_Abweichung':float(delta.max()),
                           'bestanden':bool(delta.le(1e-12).all())})
    reproduction = pd.DataFrame(checks)
    export(reproduction,'reproduktion_originalmittelwerte_und_kontraste')
    display(reproduction)
    assert reproduction.bestanden.all(), 'Originalwerte weichen ab: vor Sensitivitätsanalyse klären.'
    export(original_means,'original_retention_mittelwerte_95ci')
    export(original_contrasts,'original_vier_kontraste_pp_95ci')
    display(original_contrasts[['contrast','stage_target_words','mean','ci_low','ci_high']].round(3))

    from summary_rules import count_words, bounds_for_target
    length_ns={'count_words':count_words,'bounds_for_target':bounds_for_target}

    length_rows=[]
    for row in summaries.itertuples():
        for stage in STAGES:
            text=getattr(row,f'summary_{stage}')
            words=length_ns['count_words'](text)
            low,high=length_ns['bounds_for_target'](stage,5.0)
            assert words==int(getattr(row,f'summary_{stage}_word_count'))
            violation=max(low-words,words-high,0)
            length_rows.append(dict(article_id=row.article_id,method=row.method,stage_target_words=stage,
                W=words,violation_words=violation,D=violation/stage,
                summary_sha256=hashlib.sha256(text.encode()).hexdigest()))
    length_values=pd.DataFrame(length_rows)
    length_archive=read(LENGTH/'artikelwerte_laenge_retention.csv')
    cross=length_values.merge(length_archive,on=CELL,validate='one_to_one',suffixes=('','_old'))
    for name in ['W','violation_words','D']:
        assert np.allclose(cross[name],cross[name+'_old'],rtol=0,atol=1e-12),name
    assert cross.summary_sha256.eq(cross.summary_sha256_old).all()
    article=article.merge(length_values,on=CELL,validate='one_to_one')
    article['R_adj']=article.retention_lower/(1+article.D)
    reference_groups=read(LENGTH/'vollstaendige_24_gruppen_main_100.csv')
    length_reproduction=[]
    for row in reference_groups.itertuples():
        sub=article[article.method.eq(row.method)&article.stage_target_words.eq(row.stage_target_words)]
        stats=estimate(sub.set_index('article_id').R_adj)
        for name in ['mean','ci_low','ci_high']:
            delta=abs(stats[name]-getattr(row,'R_adj_'+name))
            length_reproduction.append(dict(method=row.method,stage_target_words=row.stage_target_words,
                field=name,abs_difference=delta,passed=delta<=1e-12))
    length_reproduction=pd.DataFrame(length_reproduction)
    export(length_reproduction,'reproduktion_laengenadjustiert')
    assert length_reproduction.passed.all()

    def rank_scores(scores,tol=0.0):
        # Letzte Achse = Verfahren. Gleichstand erhält denselben Wettbewerbsrang.
        return 1+np.sum(scores[...,None,:] > scores[...,:,None]+tol,axis=-1)

    def tensor(column,dtype=float):
        index=pd.MultiIndex.from_product([IDS,METHODS,STAGES],names=CELL)
        return article.set_index(CELL)[column].reindex(index).to_numpy(dtype=dtype).reshape(len(IDS),6,4)

    V=tensor('n_valid',int)
    K=tensor('n_retained',int)
    DISCORD=tensor('n_discordant',int)
    PENALTY=1/(1+tensor('D'))
    ORIG_RAW=(K/V).mean(axis=0)
    ORIG_ADJ=(K/20*PENALTY).mean(axis=0)
    LCM=math.lcm(*np.unique(V).tolist())
    assert LCM*100*20 < np.iinfo(np.int64).max
    RAW_SCALE=LCM//V
    ORIG_RAW_SCORES=(K*RAW_SCALE).sum(axis=0)
    assert np.allclose(ORIG_RAW,ORIG_RAW_SCORES/(LCM*len(IDS)))
    original_ranks_raw=rank_scores(ORIG_RAW_SCORES.T).T
    original_ranks_adj=rank_scores(ORIG_ADJ.T,tol=1e-12).T
    rank_rows=[]
    for mi,method in enumerate(METHODS):
        for ti,stage in enumerate(STAGES):
            row=reference_groups[reference_groups.method.eq(method)&reference_groups.stage_target_words.eq(stage)].iloc[0]
            assert original_ranks_raw[mi,ti]==row.rank_R_obs
            assert original_ranks_adj[mi,ti]==row.rank_R_adj
            rank_rows.append(dict(method=method,stage_target_words=stage,
                raw_percent=100*ORIG_RAW[mi,ti],adjusted_percent=100*ORIG_ADJ[mi,ti],
                original_rank_raw=int(original_ranks_raw[mi,ti]),original_rank_adjusted=int(original_ranks_adj[mi,ti])))
    original_ranks=pd.DataFrame(rank_rows)
    export(original_ranks,'original_mittelwerte_und_rangfolgen')
    display(original_ranks.round(3))
    display(Markdown('**Reproduktion bestanden:** alle 24 Hauptmittelwerte, 16 gepaarten Kontraste, '
        'zugehörigen Originalintervalle sowie 24 adjustierten Mittelwerte und Intervalle '
        'stimmen innerhalb 10⁻¹² mit dem vorhandenen Projekt überein. Auch beide Originalrangfolgen stimmen.'))

    MATCHED=PACKAGE/'02_Auswertung/hauptexperiment_laengenmatched_ergebnisse/aktuell_20260930'
    extra_sources=[DATA/VERSION/'generation_states.jsonl.gz',
        MATCHED/'bootstrap_intervalle_60_eindeutige_vergleiche.csv',
        MATCHED/'paarbildung_audit_6000_artikelpaare.csv',
        PACKAGE/'02_Auswertung/build_hauptexperiment_06_laengenmatched_matrizen.py']
    for path in extra_sources:
        sources_before[str(path)]=sha(path)
        sources.loc[len(sources)]={'Datei':str(path.relative_to(PACKAGE)),'sha256':sha(path)}
    state_entry=manifest_entries[f'{VERSION}/generation_states.jsonl.gz']
    assert sha(DATA/VERSION/'generation_states.jsonl.gz')==state_entry['sha256']
    export(sources,'quellen_sha256')
    with gzip.open(DATA/VERSION/'generation_states.jsonl.gz','rt') as handle:
        states=[json.loads(line) for line in handle]
    assert len(states)==600
    lookup=summaries.set_index(['article_id','method'])
    input_rows=[]
    for record in states:
        a=int(record['article_id']);method=record['method'];row=lookup.loc[(a,method)]
        state=record['state']
        assert state['complete'] and state['source_sha256']==row.article_sha256
        assert hashlib.sha256(row.article.encode()).hexdigest()==row.article_sha256
        assert [s['target_words'] for s in state['stages']]==STAGES
        previous=row.article
        original_words=length_ns['count_words'](previous)
        for stage in state['stages']:
            target=stage['target_words']
            output=stage['selected']['response']
            assert stage['input_text']==previous and output==row[f'summary_{target}']
            count=length_ns['count_words'](previous)
            assert count==stage['input_word_count']
            input_rows.append(dict(article_id=a,method=method,stage_target_words=target,
                input_words=count,original_words=original_words,empty_input=not bool(previous.strip()),
                empty_output=not bool(output.strip())))
            previous=output
    inputs=pd.DataFrame(input_rows)
    article=article.merge(inputs,on=CELL,validate='one_to_one')
    word_reference=reference_groups[['method','stage_target_words','W_mean']]
    word_means=article.groupby(['method','stage_target_words']).W.mean().rename('W_reproduced').reset_index()
    word_check=word_means.merge(word_reference,on=['method','stage_target_words'],validate='one_to_one')
    word_check['abs_difference']=abs(word_check.W_reproduced-word_check.W_mean)
    export(word_check,'reproduktion_24_wortzahlmittelwerte')
    assert word_check.abs_difference.le(1e-12).all()
    display(Markdown('**Reproduktion abgeschlossen:** 24 QA-Mittelwerte, 16 gepaarte '
        'Differenzen samt ihren Originalintervallen, 24 Wortzahlmittelwerte sowie die '
        'vorhandene heuristische Adjustierung und Rangfolgen stimmen. Für Wortzahlmittelwerte '
        'ist in der bestehenden 24-Gruppen-Tabelle kein eigenes Konfidenzintervall archiviert.'))

    article['word_change']=article.W-article.input_words
    article['absolute_word_change']=article.word_change.abs()
    article['length_ratio']=article.W/article.input_words.replace(0,np.nan)
    article['compression_rate']=1-article.length_ratio
    article['actually_shorter']=article.W.lt(article.input_words)
    article['equal_length']=article.W.eq(article.input_words)
    article['expands']=article.W.gt(article.input_words)
    article['window_hit']=article.D.eq(0)
    article['baseline_type']=np.where(article.method.eq('lead_k'),'deterministische Präfixbaseline','generativ rekursiv')
    assert (article.actually_shorter.astype(int)+article.equal_length+article.expands).eq(1).all()
    compression=article.groupby(['method','stage_target_words'],sort=False).agg(
        n_articles=('article_id','size'),mean_words=('W','mean'),median_words=('W','median'),
        sd_words=('W','std'),q25_words=('W',lambda x:x.quantile(.25)),q75_words=('W',lambda x:x.quantile(.75)),
        mean_input_words=('input_words','mean'),mean_compression_rate=('compression_rate','mean'),
        median_compression_rate=('compression_rate','median'),n_relative_defined=('compression_rate','count'),
        n_shorter=('actually_shorter','sum'),n_equal=('equal_length','sum'),n_expands=('expands','sum'),
        n_window_hits=('window_hit','sum'),n_empty_input=('empty_input','sum'),n_empty_output=('empty_output','sum')).reset_index()
    compression['iqr_words']=compression.q75_words-compression.q25_words
    for name in ['shorter','equal','expands','window_hits']:
        compression[f'{name}_percent']=100*compression[f'n_{name}']/compression.n_articles
    export(article,'A_alle_2400_kompressions_und_retention_werte')
    export(compression,'A_tatsaechliche_kompression_24_gruppen')
    display(compression[['method','stage_target_words','mean_words','median_words','sd_words','iqr_words',
        'mean_input_words','mean_compression_rate','n_relative_defined','n_shorter','n_equal','n_expands','n_window_hits']].round(3))
    expansion300=compression[compression.stage_target_words.eq(300)]
    display(Markdown('**Expansion auf der nominellen 300-Wort-Stufe:** '+ '; '.join(
        f'{NAMES[r.method]} {int(r.n_expands)}/100 ({de(r.expands_percent,1)} %)' for r in expansion300.itertuples())+'.'))

    histogram_rows=[]
    fig,axes=plt.subplots(2,2,figsize=(13.5,9),layout='constrained')
    for ax,stage in zip(axes.flat,STAGES):
        stage_data=article[article.stage_target_words.eq(stage)]
        assert stage_data.loc[stage_data.method.eq('lead_k'),'W'].eq(stage).all()
        bin_width=max(5,int(np.ceil((stage_data.W.max()-stage_data.W.min())/18/5))*5)
        edges=np.arange(np.floor(stage_data.W.min()/bin_width)*bin_width,
                        np.ceil(stage_data.W.max()/bin_width)*bin_width+bin_width*1.01,bin_width)
        for mi,method in enumerate(GENERATIVE):
            values=stage_data.loc[stage_data.method.eq(method),'W'].to_numpy()
            counts,_=np.histogram(values,bins=edges)
            assert counts.sum()==100
            ax.hist(values,bins=edges,histtype='step',lw=1.8,color=METHOD_COLORS[method],
                    linestyle=['-','--','-.',':','-'][mi],label=NAMES[method])
            histogram_rows.extend(dict(method=method,stage_target_words=stage,lower=edges[i],upper=edges[i+1],
                n=int(n),right_edge_inclusive=i==len(counts)-1) for i,n in enumerate(counts))
        ax.axvline(stage,color='#333333',ls='--',lw=1.2,label='Zielwortzahl / Lead-k (n=100)')
        ax.set(xlabel='Tatsächliche Ausgabewortzahl',ylabel='Anzahl Artikel je Wortzahlintervall',
               title=f'Zielstufe {stage} Wörter · Klassenbreite {bin_width} Wörter')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True));ax.grid(axis='y',alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('Wortzahlverteilungen der Verfahren · je 100 Artikel und Stufe',fontsize=14)
    export(pd.DataFrame(histogram_rows),'grafik_histogramme_wortzahlen')
    savefig(fig,'03_histogramme_tatsaechlicher_wortzahlen')
    display(Markdown('**Lesart:** Überlappende Histogramme zeigen, wo Vergleiche bei ähnlicher Länge möglich '
        'sein könnten. Die anschließende Supportprüfung verlangt zusätzlich ausreichend gepaarte Artikel '
        'und lokale Beobachtungen; die bloße Überlappung einzelner Balken genügt dafür nicht.'))

    quantiles=article.groupby(['method','stage_target_words']).W.quantile([0,.05,.10,.25,.50,.75,.90,.95,1]).unstack()
    quantiles.columns=['min','p05','p10','p25','p50','p75','p90','p95','max']
    quantiles=quantiles.reset_index()
    export(quantiles,'B_wortzahlverteilungen_24_gruppen')
    display(quantiles.round(2))
    SUPPORT_DEFS={'central95':(.05,.95),'broad':(0.,1.),'central80':(.10,.90)}

    def support_for(methods,stage,definition):
        sub=article[article.stage_target_words.eq(stage)&article.method.isin(methods)]
        wide=sub.pivot(index='article_id',columns='method',values='W').reindex(index=IDS,columns=methods)
        loq,hiq=SUPPORT_DEFS[definition]
        lower=float(max(wide[m].quantile(loq) for m in methods))
        upper=float(min(wide[m].quantile(hiq) for m in methods))
        in_range=wide.ge(lower)&wide.le(upper)&wide.gt(0)
        n_by_method={m:int(in_range[m].sum()) for m in methods}
        paired=in_range.all(axis=1)
        local=paired.copy()
        for a in wide.index[paired]:
            for w in wide.loc[a].to_numpy():
                bandwidth=max(5,.1*w)
                if any(int((wide[m]-w).abs().le(bandwidth).sum())<5 for m in methods):
                    local.loc[a]=False
                    break
        reasons=[]
        if lower>=upper:reasons.append('kein gemeinsamer Bereich mit positiver Breite')
        if min(n_by_method.values())<MIN_SUPPORT:reasons.append('weniger als 20 Beobachtungen je Verfahren im Bereich')
        if int(local.sum())<MIN_SUPPORT:reasons.append('weniger als 20 lokal gestützte gepaarte Artikel')
        if any(m=='lead_k' for m in methods):reasons.append('Lead-k bleibt außerhalb der generativen Regression')
        return dict(lower=lower,upper=upper,n_by_method=n_by_method,
            n_paired_range=int(paired.sum()),n_supported=int(local.sum()),ids=wide.index[local].tolist(),
            estimable=not reasons,reason='; '.join(reasons) if reasons else 'gestützter explorativer Vergleich',
            small_group=0<int(local.sum())<30)

    support_rows=[];supports={}
    for first,second in combinations(METHODS,2):
        for stage in STAGES:
            left=article[(article.method==first)&article.stage_target_words.eq(stage)].set_index('article_id')
            right=article[(article.method==second)&article.stage_target_words.eq(stage)].set_index('article_id')
            direct=(left.W+right.W).gt(0)&(40*(left.W-right.W).abs()).le(left.W+right.W)
            for definition in SUPPORT_DEFS:
                support=support_for([first,second],stage,definition)
                supports[(first,second,stage,definition)]=support
                support_rows.append(dict(first=first,second=second,stage_target_words=stage,definition=definition,
                    common_lower=support['lower'],common_upper=support['upper'],
                    n_first_in_range=support['n_by_method'][first],n_second_in_range=support['n_by_method'][second],
                    n_paired_range=support['n_paired_range'],n_supported_articles=support['n_supported'],
                    n_direct_5pct_pairs=int(direct.sum()),estimable=support['estimable'],
                    small_group=support['small_group'],status=support['reason']))
    overlap=pd.DataFrame(support_rows)
    export(overlap,'B_ueberlappung_60_paare_drei_definitionen')

    def get_support(first,second,stage,definition='central95'):
        key=(first,second,stage,definition)
        if key not in supports:key=(second,first,stage,definition)
        return supports[key]

    core_support=pd.DataFrame([dict(first=f,second=s,stage_target_words=t,
        lower=get_support(f,s,t)['lower'],upper=get_support(f,s,t)['upper'],
        n=get_support(f,s,t)['n_supported'],estimable=get_support(f,s,t)['estimable'],
        status=get_support(f,s,t)['reason']) for f,s in CONTRASTS for t in STAGES])
    display(core_support.round(2))
    export(core_support,'B_ueberlappung_16_hauptkontraste')
    global_support={t:support_for(GENERATIVE,t,'central95') for t in STAGES}
    display(pd.DataFrame([dict(stage=t,lower=s['lower'],upper=s['upper'],n=s['n_supported'],
        estimable=s['estimable'],status=s['reason']) for t,s in global_support.items()]))

    model_data=article[article.method.isin(GENERATIVE)].sort_values(CELL).reset_index(drop=True).copy()
    assert len(model_data)==2000
    MODEL_CELLS=[(m,t) for m in GENERATIVE for t in STAGES]
    cell_map={key:i for i,key in enumerate(MODEL_CELLS)}
    article_map={a:i for i,a in enumerate(IDS)}
    N_ART=len(IDS)
    ART_INDEX=model_data.article_id.map(article_map).to_numpy()
    Y=np.column_stack([model_data.n_retained,model_data.n_valid-model_data.n_retained]).astype(float)
    SPLINE_INFO={}
    for stage in STAGES:
        w=model_data.loc[model_data.stage_target_words.eq(stage),'W'].to_numpy()/100
        spline=patsy.dmatrix("cr(w, df=3, constraints='center') - 1",{'w':w},return_type='dataframe')
        SPLINE_INFO[stage]=spline.design_info

    def design(frame,kind):
        n=len(frame)
        base=np.zeros((n,20+99),dtype=float)
        cell_idx=np.array([cell_map[(m,int(t))] for m,t in zip(frame.method,frame.stage_target_words)])
        ai=frame.article_id.map(article_map).to_numpy()
        assert np.isfinite(ai).all()
        ai=ai.astype(int)
        base[np.arange(n),cell_idx]=1
        has_dummy=ai>0
        base[np.flatnonzero(has_dummy),20+ai[has_dummy]-1]=1
        length_columns=[]
        for stage in STAGES:
            selected=frame.stage_target_words.eq(stage).to_numpy()
            if kind=='linear':
                block=np.zeros((n,1));block[selected,0]=(frame.loc[selected,'W'].to_numpy()-stage)/100
            else:
                block=np.zeros((n,3))
                if selected.any():
                    block[selected]=np.asarray(patsy.build_design_matrices([SPLINE_INFO[stage]],
                        {'w':frame.loc[selected,'W'].to_numpy()/100})[0])
            length_columns.append(block)
        return np.column_stack([base]+length_columns)

    models={};diagnostic_rows=[];coefficient_rows=[]
    for kind in ['linear','spline']:
        X=design(model_data,kind)
        names=[f'{m}:{t}' for m,t in MODEL_CELLS]+[f'article_{a}' for a in IDS[1:]]
        names += [f'length_{t}'+(f'_basis{b}' if kind=='spline' else '')
                  for t in STAGES for b in (range(1,4) if kind=='spline' else [1])]
        started=time.perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            result=sm.GLM(Y,X,family=sm.families.Binomial()).fit(maxiter=100,tol=1e-10,
                cov_type='cluster',cov_kwds={'groups':model_data.article_id.to_numpy(),'use_correction':True})
        seconds=time.perf_counter()-started
        cov=np.asarray(result.cov_params())
        eig=la.eigvalsh((cov+cov.T)/2)
        stable=bool(result.converged and np.isfinite(result.params).all() and
            np.linalg.matrix_rank(X)==X.shape[1] and np.isfinite(cov).all() and eig.min()>-1e-8)
        models[kind]=dict(X=X,result=result,beta=np.asarray(result.params),cov=cov,stable=stable,names=names)
        diagnostic_rows.append(dict(model=kind,n_rows=len(X),n_parameters=X.shape[1],design_rank=np.linalg.matrix_rank(X),
            n_article_clusters=N_ART,converged=result.converged,iterations=result.fit_history.get('iteration'),
            max_abs_coefficient=float(abs(result.params).max()),design_condition=float(np.linalg.cond(X)),
            covariance_rank=np.linalg.matrix_rank(cov),min_covariance_eigenvalue=float(eig.min()),
            max_abs_score=float(abs(result.model.score(result.params)).max()),
            pearson_ratio=float(result.pearson_chi2/result.df_resid),deviance=float(result.deviance),
            fit_seconds=seconds,warning_text=' | '.join(str(w.message) for w in captured),stable=stable))
        for i,name in enumerate(names):
            coefficient_rows.append(dict(model=kind,term=name,estimate=result.params[i],
                cluster_robust_se=float(np.sqrt(max(cov[i,i],0)))))
        influence=result.get_influence(observed=False)
        diagnostics=model_data[CELL+['W','n_retained','n_valid','retention_observed']].copy()
        diagnostics['predicted']=result.predict()
        diagnostics['pearson_residual']=result.resid_pearson
        diagnostics['leverage']=influence.hat_matrix_diag
        export(diagnostics,f'C_residuen_und_hebelwerte_{kind}')
    diagnostics_table=pd.DataFrame(diagnostic_rows)
    export(diagnostics_table,'C_modelldiagnostik')
    export(pd.DataFrame(coefficient_rows),'C_regressionskoeffizienten_clusterrobuste_se')
    display(diagnostics_table)
    assert diagnostics_table.stable.all(), 'Modellinstabilität: keine verlässlichen Standardisierungen ausgeben.'
    display(Markdown('Die robuste Kovarianz ist erwartungsgemäß rangreduziert: Scorebeiträge der '
        'artikelweisen Interzepte summieren sich innerhalb ihrer Cluster nahezu zu null. '
        'Es wird kein gemeinsamer Wald-Test aller Koeffizienten daraus abgeleitet. '
        'Die Kontrastintervalle beruhen zusätzlich auf vollständigen Artikelbootstrap-Anpassungen.'))

    binned_rows=[]
    fig,axes=plt.subplots(2,2,figsize=(15,9.4),layout='constrained')
    cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#EEEEEE')
    for ax,stage in zip(axes.flat,STAGES):
        sub=article[article.stage_target_words.eq(stage) & article.method.isin(GENERATIVE)].copy()
        width={500:50,300:50,150:50,75:25}[stage]
        edges=np.arange(np.floor(sub.W.min()/width)*width,(np.floor(sub.W.max()/width)+1)*width+width*.5,width)
        sub['word_bin']=pd.cut(sub.W,edges,right=False,include_lowest=True)
        grouped=sub.groupby(['method','word_bin'],observed=False).agg(n=('article_id','size'),retention=('retention_observed','mean'))
        bins=sub.word_bin.cat.categories
        means=grouped.retention.unstack().reindex(index=GENERATIVE,columns=bins)*100
        counts=grouped.n.unstack().reindex(index=GENERATIVE,columns=bins,fill_value=0)
        plotted=means.where(counts.ge(5))
        im=ax.imshow(np.ma.masked_invalid(plotted.to_numpy()),cmap=cmap,vmin=0,vmax=100,aspect='auto')
        for i,method in enumerate(GENERATIVE):
            for j,interval in enumerate(bins):
                n=int(counts.iloc[i,j]);mean=means.iloc[i,j]
                text=f'{mean:.0f} %\nn={n}' if n>=5 else f'n={n}'
                ax.text(j,i,text,ha='center',va='center',fontsize=8.5,color='white' if n>=5 and mean>55 else '#333333')
                binned_rows.append(dict(method=method,stage_target_words=stage,lower=interval.left,upper=interval.right,
                    n=n,mean_retention_percent=mean,displayed=n>=5))
        assert int(counts.to_numpy().sum())==500
        ax.set(xticks=range(len(bins)),xticklabels=[f'{int(b.left)}–\n<{int(b.right)}' for b in bins],
               yticks=range(5),yticklabels=[NAMES[m] for m in GENERATIVE],xlabel='Tatsächliche Ausgabewortzahl',
               title=f'Zielstufe {stage} Wörter · beobachtete Artikelwerte')
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.6,label='Mittlere beobachtete QA-Retention (%)')
    fig.suptitle('Beobachtete Retention nach Wortzahlklasse · Mittelwert nur bei n ≥ 5',fontsize=14)
    export(pd.DataFrame(binned_rows),'grafik_retention_nach_wortzahlklassen')
    savefig(fig,'04_beobachtete_retention_nach_wortzahlklasse')
    shown_bins=pd.DataFrame(binned_rows)
    display(Markdown(f'**Beobachtung:** {int(shown_bins.displayed.sum())} von {len(shown_bins)} '
        'Verfahren-Wortzahl-Zellen enthalten mindestens fünf Artikel. Die übrigen Zellen bleiben ohne Mittelwert; '
        'die Artikelzusammensetzung kann sich zwischen Verfahren und Wortzahlklassen unterscheiden.'))

    comparison_specs=[]
    for first,second in combinations(GENERATIVE,2):
        for stage in STAGES:
            for definition in SUPPORT_DEFS:
                support=get_support(first,second,stage,definition)
                if not support['estimable']:continue
                ids=support['ids']
                left=article[(article.method==first)&article.stage_target_words.eq(stage)].set_index('article_id').loc[ids]
                right=article[(article.method==second)&article.stage_target_words.eq(stage)].set_index('article_id').loc[ids]
                reference=pd.DataFrame({'article_id':np.repeat(ids,2),'W':np.column_stack([left.W,right.W]).ravel(),
                                        'stage_target_words':stage})
                comparison_specs.append(dict(first=first,second=second,stage=stage,definition=definition,
                    support=support,reference=reference,
                    reference_article_indices=reference.article_id.map(article_map).to_numpy(),
                    raw_same_articles_pp=float(100*(left.retention_observed-right.retention_observed).mean())))

    def standardized_point(spec,kind):
        ref=spec['reference']
        xa=design(ref.assign(method=spec['first']),kind)
        xb=design(ref.assign(method=spec['second']),kind)
        beta=models[kind]['beta']
        return float(100*np.mean(expit(xa@beta)-expit(xb@beta))),xa,xb

    point_rows=[]
    for spec in comparison_specs:
        for kind in models:
            point,xa,xb=standardized_point(spec,kind)
            point_rows.append(dict(first=spec['first'],second=spec['second'],stage_target_words=spec['stage'],
                definition=spec['definition'],model=kind,adjusted_difference_pp=point,
                raw_same_articles_pp=spec['raw_same_articles_pp'],n_articles=spec['support']['n_supported'],
                common_lower=spec['support']['lower'],common_upper=spec['support']['upper']))
    points=pd.DataFrame(point_rows)
    export(points,'D_gestuetzte_standardisierte_punktschaetzer')
    display(points[(points.definition=='central95')&(points.model=='linear')].round(3))

    worker_source=Path(__file__).with_name('analysis_bootstrap_worker.py').read_text()
    import analysis_bootstrap_worker as worker
    draws=np.random.default_rng(SEED).integers(0,N_ART,size=(B,N_ART))
    bootstrap_counts=np.zeros((B,N_ART),dtype=np.int16)
    np.add.at(bootstrap_counts,(np.repeat(np.arange(B),N_ART),draws.ravel()),1)
    assert bootstrap_counts.sum(axis=1).tolist()==[N_ART]*B
    boot_coefficients={};bootstrap_fit_rows=[]
    weight_validation=[]
    for kind,state in models.items():
        worker.initialize(state['X'],Y,ART_INDEX,state['beta'])
        # Häufigkeitsgewichte unabhängig gegen physisch wiederholte Artikelzeilen prüfen,
        # insbesondere die neue Referenzkodierung, wenn Originalartikel 1 nicht gezogen wurde.
        test_indices=[int(np.flatnonzero(bootstrap_counts[:,0]>0)[0]),
                      int(np.flatnonzero(bootstrap_counts[:,0]==0)[0])]
        for draw in test_indices:
            counts=bootstrap_counts[draw]
            reference=int(np.flatnonzero(counts>0)[0])
            columns=np.ones(state['X'].shape[1],dtype=bool)
            for ai in range(1,N_ART):
                if counts[ai]==0 or ai==reference:columns[20+ai-1]=False
            expanded=np.repeat(np.arange(len(model_data)),counts[ART_INDEX])
            direct_fit=sm.GLM(Y[expanded],state['X'][expanded][:,columns],family=sm.families.Binomial()).fit(
                maxiter=100,tol=1e-10)
            weighted,records_check=worker.fit_batch(counts[None,:])
            included=counts[ART_INDEX]>0
            direct_predictions=direct_fit.predict(state['X'][included][:,columns])
            weighted_predictions=expit(state['X'][included]@weighted[0])
            max_error=float(abs(direct_predictions-weighted_predictions).max())
            assert records_check[0][0] and direct_fit.converged and max_error<1e-7
            weight_validation.append(dict(model=kind,draw=draw,original_reference_absent=bool(counts[0]==0),
                n_expanded_rows=len(expanded),max_prediction_difference=max_error,passed=True))
    export(pd.DataFrame(weight_validation),'bootstrap_haufigkeitsgewichte_unabhaengig_validiert')
    for kind,state in models.items():
        fingerprint=hashlib.sha256(state['X'].tobytes()+Y.tobytes()+state['beta'].tobytes()+
            bootstrap_counts.tobytes()+worker_source.encode()+json.dumps(software,sort_keys=True).encode()).hexdigest()
        cache=OUT/f'bootstrap_10000_{kind}.npz'
        cached=False
        if USE_BOOTSTRAP_CACHE and cache.is_file() and not recompute_bootstrap:
            with np.load(cache,allow_pickle=False) as archive:
                if str(archive['fingerprint'])==fingerprint:
                    coefficients=archive['coefficients'].copy()
                    records=json.loads(str(archive['records']))
                    cached=True
        started=time.perf_counter()
        if not cached:
            if not recompute_bootstrap:
                raise RuntimeError("Bootstrap-Cache passt nicht. Explizit --recompute-regression-bootstrap verwenden.")
            # Zusätzliche Implementierungsprüfung: alle Artikel genau einmal reproduziert die Punktanpassung.
            worker.initialize(state['X'],Y,ART_INDEX,state['beta'])
            check_coef,check_rec=worker.fit_batch(np.ones((1,N_ART),dtype=np.int16))
            assert check_rec[0][0] and np.allclose(check_coef[0],state['beta'],rtol=0,atol=1e-6)
            batches=[bootstrap_counts[i:i+100] for i in range(0,B,100)]
            parts=[];records=[]
            with ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn'),
                initializer=worker.initialize,initargs=(state['X'],Y,ART_INDEX,state['beta'])) as pool:
                for idx,(coef,rec) in enumerate(pool.map(worker.fit_batch,batches)):
                    parts.append(coef);records.extend(rec)
                    if (idx+1)%20==0:print(f'{kind}: {(idx+1)*100}/{B} vollständige Bootstrap-Anpassungen')
            coefficients=np.vstack(parts)
            np.savez_compressed(cache,coefficients=coefficients,records=np.array(json.dumps(records)),
                                fingerprint=np.array(fingerprint),bootstrap_counts=bootstrap_counts)

        assert coefficients.shape==(B,state['X'].shape[1]) and len(records)==B
        success=np.array([bool(r[0]) for r in records])
        assert np.isfinite(coefficients[success]).all()
        boot_coefficients[kind]=coefficients
        bootstrap_fit_rows.append(dict(model=kind,n_requested=B,n_success=int(success.sum()),
            n_failed=int((~success).sum()),n_warning_fits=sum(bool(r[2]) for r in records),
            max_iterations=max(r[1] for r in records),seconds_this_run=time.perf_counter()-started,
            cache_used=cached,fingerprint=fingerprint))
        failures=pd.DataFrame([dict(draw=i,converged=r[0],iterations=r[1],message=r[2])
                               for i,r in enumerate(records) if not r[0] or r[2]],
                              columns=['draw','converged','iterations','message'])
        export(failures,f'bootstrap_numerische_hinweise_{kind}')
    bootstrap_fits=pd.DataFrame(bootstrap_fit_rows)
    export(bootstrap_fits,'bootstrap_anpassungsdiagnostik')
    display(bootstrap_fits.drop(columns='fingerprint'))

    bootstrap_comparisons=[]
    for spec in comparison_specs:
        for kind in models:
            point,xa,xb=standardized_point(spec,kind)
            reference_ai=spec['reference_article_indices']
            distributions=np.full(B,np.nan)
            for start in range(0,B,200):
                end=min(start+200,B)
                coefficients=boot_coefficients[kind][start:end]
                weights=bootstrap_counts[start:end,reference_ai]
                denominator=weights.sum(axis=1)
                probabilities_a=expit(coefficients@xa.T)
                probabilities_b=expit(coefficients@xb.T)
                numerator=((probabilities_a-probabilities_b)*weights).sum(axis=1)
                distributions[start:end]=np.divide(100*numerator,denominator,
                    out=np.full(end-start,np.nan),where=denominator>0)
            finite=np.isfinite(distributions)
            reliable=int(finite.sum())>=int(.99*B)
            low,high=np.nanquantile(distributions,[.025,.975]) if reliable else [np.nan,np.nan]
            bootstrap_comparisons.append(dict(first=spec['first'],second=spec['second'],stage_target_words=spec['stage'],
                definition=spec['definition'],model=kind,adjusted_difference_pp=point if reliable else np.nan,
                ci_low_pp=low,ci_high_pp=high,n_bootstrap_valid=int(finite.sum()),n_bootstrap_missing=int((~finite).sum()),
                reliable=reliable,n_articles=spec['support']['n_supported'],n_reference_points=len(spec['reference']),
                common_lower=spec['support']['lower'],common_upper=spec['support']['upper'],
                raw_same_articles_pp=spec['raw_same_articles_pp']))
    controlled=pd.DataFrame(bootstrap_comparisons)
    export(controlled,'D_alle_gestuetzten_kontraste_95ci')

    def oriented_result(first,second,stage,definition='central95',kind='linear'):
        rows=controlled[(controlled['first']==first)&(controlled.second==second)&
            controlled.stage_target_words.eq(stage)&controlled.definition.eq(definition)&controlled.model.eq(kind)]
        if len(rows):return rows.iloc[0].copy()
        rows=controlled[(controlled['first']==second)&(controlled.second==first)&
            controlled.stage_target_words.eq(stage)&controlled.definition.eq(definition)&controlled.model.eq(kind)]
        if not len(rows):return None
        row=rows.iloc[0].copy()
        row['first'],row['second']=first,second
        row['adjusted_difference_pp']=-row.adjusted_difference_pp
        row['raw_same_articles_pp']=-row.raw_same_articles_pp
        row['ci_low_pp'],row['ci_high_pp']=-row.ci_high_pp,-row.ci_low_pp
        return row

    core_rows=[]
    for original in original_contrasts.itertuples():
        first,second,stage=original.first,original.second,original.stage_target_words
        support=get_support(first,second,stage)
        adjusted=oriented_result(first,second,stage)
        reliable=adjusted is not None and bool(adjusted.reliable)
        core_rows.append(dict(first=first,second=second,contrast=original.contrast,stage_target_words=stage,
            original_difference_pp=original.mean,original_ci_low_pp=original.ci_low,original_ci_high_pp=original.ci_high,
            controlled_difference_pp=adjusted.adjusted_difference_pp if reliable else np.nan,
            controlled_ci_low_pp=adjusted.ci_low_pp if reliable else np.nan,
            controlled_ci_high_pp=adjusted.ci_high_pp if reliable else np.nan,
            raw_same_articles_pp=adjusted.raw_same_articles_pp if reliable else np.nan,
            common_lower=support['lower'],common_upper=support['upper'],
            n_first_in_range=support['n_by_method'][first],n_second_in_range=support['n_by_method'][second],
            n_supported_articles=support['n_supported'],n_reference_points=2*support['n_supported'],
            reliable=reliable,status=('kleine Gruppe (<30 Artikel)' if support['small_group'] else 'gestützt')
                if reliable else 'nicht zuverlässig schätzbar: '+support['reason']))
    core_table=pd.DataFrame(core_rows)
    export(core_table,'haupttabelle_16_originale_und_laengenkontrollierte_kontraste')
    display(core_table.drop(columns=['first','second']).round(3))

    curve_rows=[]
    state=models['linear']
    for stage in STAGES:
        for method in GENERATIVE:
            actual=article[(article.method==method)&article.stage_target_words.eq(stage)].W
            lower,upper=actual.quantile([.05,.95])
            grid=np.linspace(lower,upper,70)
            ref=pd.DataFrame({'article_id':np.tile(IDS,len(grid)),'W':np.repeat(grid,len(IDS)),
                              'method':method,'stage_target_words':stage})
            X=design(ref,'linear')
            mu=expit(X@state['beta'])
            mean=mu.reshape(len(grid),len(IDS)).mean(axis=1)
            gradients=(X*(mu*(1-mu))[:,None]).reshape(len(grid),len(IDS),X.shape[1]).mean(axis=1)
            variance=np.einsum('ij,jk,ik->i',gradients,state['cov'],gradients)
            assert variance.min()>-1e-10
            se=np.sqrt(np.maximum(variance,0))/(mean*(1-mean))
            logit=np.log(mean/(1-mean))
            critical=student_t.ppf(.975,99)
            local=np.array([actual.sub(w).abs().le(max(5,.1*w)).sum()>=5 for w in grid])
            for w,p,low,high,ok in zip(grid,mean,expit(logit-critical*se),expit(logit+critical*se),local):
                curve_rows.append(dict(method=method,stage_target_words=stage,words=w,p=100*p,
                    ci_low=100*low,ci_high=100*high,local_support=bool(ok)))
    curves=pd.DataFrame(curve_rows)
    export(curves,'abbildung_1_kurvenwerte_und_unterstuetzung')
    fig=plt.figure(figsize=(13.2,10.2),layout='constrained')
    gs=fig.add_gridspec(4,2,height_ratios=[3.2,1,3.2,1])
    for index,stage in enumerate(STAGES):
        row,col=2*(index//2),index%2
        ax=fig.add_subplot(gs[row,col]);rug=fig.add_subplot(gs[row+1,col],sharex=ax)
        for mi,method in enumerate(METHODS):
            sub=article[(article.method==method)&article.stage_target_words.eq(stage)]
            color=METHOD_COLORS[method]
            if method=='lead_k':
                ax.scatter(sub.W.mean(),100*sub.retention_observed.mean(),marker='D',s=55,color=color,zorder=5)
            else:
                ax.scatter(sub.W,100*sub.retention_observed,s=9,alpha=.20,color=color)
                curve=curves[(curves.method==method)&curves.stage_target_words.eq(stage)]
                y=curve.p.where(curve.local_support)
                ax.plot(curve.words,y,color=color,lw=1.7)
                ax.fill_between(curve.words,curve.ci_low.where(curve.local_support),
                    curve.ci_high.where(curve.local_support),color=color,alpha=.12)
            rug.scatter(sub.W,np.full(len(sub),mi),marker='|',s=45,alpha=.4,color=color)
        ax.set(ylim=(0,100),ylabel='QA-Retention (%)',title=f'Zielstufe {stage} Wörter')
        ax.tick_params(labelbottom=False)
        rug.set(yticks=range(6),yticklabels=['Lead','P','ZS','FS','Auto','AM'],xlabel='Tatsächliche Ausgabewortzahl',ylim=(-.6,5.6))
        rug.tick_params(axis='y',length=0,labelsize=8)
        ax.grid(axis='y',alpha=.15)
    handles=[plt.Line2D([0],[0],color=METHOD_COLORS[m],marker='D' if m=='lead_k' else 'o',label=NAMES[m]) for m in METHODS]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('Faktenretention und tatsächliche Wortlänge\nKurven nur bei eigener empirischer Unterstützung; Lead-k ohne Regressionskurve',fontsize=14)
    savefig(fig,'01_faktenretention_und_tatsaechliche_wortlaenge')

    word_contrast_rows=[]
    for contrast_index,(first,second) in enumerate(CONTRASTS,1):
        fig,axes=plt.subplots(2,2,figsize=(12.5,8.5),layout='constrained')
        for ax,stage in zip(axes.flat,STAGES):
            support=get_support(first,second,stage)
            ax.axhline(0,color='#444444',ls='--',lw=1)
            ax.set(xlabel='Gleiche tatsächliche Ausgabewortzahl',ylabel='QA-Retention: Differenz (Prozentpunkte)',
                   title=f'Zielstufe {stage} Wörter · n = {support["n_supported"]} gestützte Artikel')
            if not support['estimable']:
                ax.set(xticks=[],xlabel='Für diese Stufe nicht schätzbar')
                ax.text(.5,.5,'Kein ausreichend gestützter Vergleich\n'+support['reason'].replace('; ',';\n'),
                        transform=ax.transAxes,ha='center',va='center',fontsize=10)
                word_contrast_rows.append(dict(first=first,second=second,stage_target_words=stage,
                    words=np.nan,n_articles=support['n_supported'],local_support=False,status=support['reason']))
                continue
            ids=support['ids']; ai=np.array([article_map[a] for a in ids])
            grid=np.linspace(support['lower'],support['upper'],31)
            coef=boot_coefficients['linear']
            weights=bootstrap_counts[:,ai]
            denominator=weights.sum(axis=1)
            article_effect=np.zeros((len(coef),len(ids)))
            for j,index in enumerate(ai):
                if index:
                    article_effect[:,j]=coef[:,20+index-1]
            ci_a=cell_map[(first,stage)];ci_b=cell_map[(second,stage)]
            length_index=119+STAGES.index(stage)
            plotted=[]
            for w in grid:
                local_n=[int(model_data.loc[model_data.method.eq(m)&model_data.stage_target_words.eq(stage),'W']
                             .sub(w).abs().le(max(5,.1*w)).sum()) for m in [first,second]]
                local=min(local_n)>=5
                row=dict(first=first,second=second,stage_target_words=stage,words=w,n_articles=len(ids),
                    n_local_first=local_n[0],n_local_second=local_n[1],local_support=local,
                    difference_pp=np.nan,ci_low_pp=np.nan,ci_high_pp=np.nan,n_bootstrap_valid=0,status='zu wenig lokaler Support')
                if local:
                    ref=pd.DataFrame({'article_id':ids,'method':first,'stage_target_words':stage,'W':w})
                    xa=design(ref,'linear');xb=design(ref.assign(method=second),'linear')
                    beta=models['linear']['beta']
                    point=100*np.mean(expit(xa@beta)-expit(xb@beta))
                    offset=(w-stage)/100
                    pa=expit(article_effect+coef[:,ci_a,None]+offset*coef[:,length_index,None])
                    pb=expit(article_effect+coef[:,ci_b,None]+offset*coef[:,length_index,None])
                    samples=np.divide(100*((pa-pb)*weights).sum(axis=1),denominator,
                        out=np.full(len(coef),np.nan),where=denominator>0)
                    valid=np.isfinite(samples)
                    reliable=int(valid.sum())>=int(.99*B)
                    if reliable:
                        low,high=np.quantile(samples[valid],[.025,.975])
                        row.update(difference_pp=point,ci_low_pp=low,ci_high_pp=high,
                                   n_bootstrap_valid=int(valid.sum()),status='gestützter punktweiser Vergleich')
                    # Unabhängige Kontrolle der optimierten Berechnung gegen die Modellmatrix.
                    assert np.allclose(pa[:3],expit(coef[:3]@xa.T),rtol=0,atol=1e-12,equal_nan=True)
                    assert np.allclose(pb[:3],expit(coef[:3]@xb.T),rtol=0,atol=1e-12,equal_nan=True)
                plotted.append(row);word_contrast_rows.append(row)
            values=pd.DataFrame(plotted)
            ax.plot(values.words,values.difference_pp,color='#0072B2',lw=2)
            ax.fill_between(values.words,values.ci_low_pp,values.ci_high_pp,color='#0072B2',alpha=.18)
            ax.grid(axis='y',alpha=.15)
        # Gleiche Skala innerhalb eines Vergleichs, mit sichtbarer Nulllinie.
        own=[r for r in word_contrast_rows if r['first']==first and r['second']==second and pd.notna(r.get('difference_pp'))]
        if own:
            lo=min(0,min(r['ci_low_pp'] for r in own));hi=max(0,max(r['ci_high_pp'] for r in own))
            pad=max(1,.08*(hi-lo))
            for ax in axes.flat: ax.set_ylim(lo-pad,hi+pad)
        fig.suptitle(f'{NAMES[first]} − {NAMES[second]} bei gleicher Wortzahl\n'
                     'Punktweise 95-%-Artikelbootstrap-Intervalle · lineares binomiales Modell',fontsize=14)
        savefig(fig,f'05{contrast_index}_wortzahlabhaengiger_kontrast_{first}_vs_{second}')
        ranges=[]
        for stage in STAGES:
            valid_rows=[r for r in own if r['stage_target_words']==stage]
            if valid_rows:
                ranges.append(f"Zielstufe {stage}: {min(r['words'] for r in valid_rows):.0f}–{max(r['words'] for r in valid_rows):.0f} tatsächliche Wörter (n={valid_rows[0]['n_articles']})")
        display(Markdown('**Gezeigte Vergleichsbereiche:** '+('; '.join(ranges) if ranges else 'Kein gestützter Bereich')+
            '. Die Bereiche gelten jeweils für dieses Methodenpaar; daraus ergibt sich keine gemeinsame Rangfolge aller Verfahren.'))
    export(pd.DataFrame(word_contrast_rows),'grafik_wortzahlabhaengige_kontraste_95ci')

    fig,(ax,bx)=plt.subplots(1,2,figsize=(13.4,8.2),sharey=True,
        gridspec_kw={'width_ratios':[3.4,1.25]},layout='constrained')
    for y,row in enumerate(core_table.itertuples()):
        ax.errorbar(row.original_difference_pp,y-.12,
            xerr=[[row.original_difference_pp-row.original_ci_low_pp],[row.original_ci_high_pp-row.original_difference_pp]],
            fmt='o',color='#777777',mfc='white',capsize=3)
        if row.reliable:
            ax.errorbar(row.controlled_difference_pp,y+.12,
                xerr=[[row.controlled_difference_pp-row.controlled_ci_low_pp],[row.controlled_ci_high_pp-row.controlled_difference_pp]],
                fmt='s',color='#0072B2',capsize=3)
            bx.text(0,y,f'n = {row.n_supported_articles}'+(' (klein)' if row.n_supported_articles<30 else ''),va='center',fontsize=10)
        else:
            ax.axhspan(y-.45,y+.45,color='#EEEEEE',zorder=-2)
            bx.text(0,y,f'n.z.s. (n = {row.n_supported_articles})',va='center',fontsize=10,color='#666666')
    ax.axvline(0,color='#555555',ls='--',lw=1)
    ax.set(yticks=range(16),yticklabels=[f'{r.contrast} | {r.stage_target_words}' for r in core_table.itertuples()],
        xlabel='Mittlere Differenz (Prozentpunkte); punktweise 95-%-Intervalle')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    bx.set(xlim=(0,1),xticks=[]);bx.tick_params(left=False,labelleft=False)
    for spine in bx.spines.values():spine.set_visible(False)
    ax.legend(handles=[plt.Line2D([0],[0],marker='o',mfc='white',color='#777777',label='Original: 100 Artikel'),
        plt.Line2D([0],[0],marker='s',color='#0072B2',label='Längenkontrolliert: gestützte Artikel')],loc='lower right',frameon=False,fontsize=9)
    bx.set_title('Support\nn.z.s. = nicht zuverlässig schätzbar',fontsize=10)
    fig.suptitle('Originale versus längenkontrollierte Methodenunterschiede',fontsize=14)
    savefig(fig,'02_originale_und_laengenkontrollierte_kontraste')

    direct_rows=[]
    for stage in STAGES:
        for i,j in combinations(range(6),2):
            first,second=METHODS[i],METHODS[j]
            left=article[(article.method==first)&article.stage_target_words.eq(stage)].set_index('article_id').loc[IDS]
            right=article[(article.method==second)&article.stage_target_words.eq(stage)].set_index('article_id').loc[IDS]
            selected=(left.W+right.W).gt(0)&(40*(left.W-right.W).abs()).le(left.W+right.W)
            differences=(100*(left.retention_observed-right.retention_observed))[selected].to_numpy()
            n=len(differences)
            lo=hi=mean=np.nan
            if n:mean=float(differences.mean())
            if n>=2:
                rng=np.random.default_rng(np.random.SeedSequence([SEED,stage,i,j]))
                means=differences[rng.integers(0,n,size=(B,n))].mean(axis=1)
                lo,hi=np.quantile(means,[.025,.975],method='linear')
            direct_rows.append(dict(first=first,second=second,stage_target_words=stage,n=n,
                mean_delta_pp=mean,ci95_low_pp=lo,ci95_high_pp=hi))
    direct=pd.DataFrame(direct_rows)
    direct_archive=pd.read_csv(MATCHED/'bootstrap_intervalle_60_eindeutige_vergleiche.csv')
    direct_check=direct.merge(direct_archive.rename(columns={'row_method':'first','column_method':'second'}),
        on=['first','second','stage_target_words'],suffixes=('','_old'),validate='one_to_one')
    assert direct_check.n.eq(direct_check.n_old).all()
    for field in ['mean_delta_pp','ci95_low_pp','ci95_high_pp']:
        assert np.allclose(direct_check[field],direct_check[field+'_old'],rtol=0,atol=1e-12,equal_nan=True),field
    export(direct,'E_reproduzierte_direkte_5prozent_paarungen')
    direct_model_rows=[]
    for r in core_table.itertuples():
        selected=direct[(direct['first']==r.first)&(direct.second==r.second)&direct.stage_target_words.eq(r.stage_target_words)]
        orientation=1
        if selected.empty:
            selected=direct[(direct['first']==r.second)&(direct.second==r.first)&direct.stage_target_words.eq(r.stage_target_words)]
            orientation=-1
        d=selected.iloc[0]
        direct_mean=orientation*d.mean_delta_pp
        direct_low=d.ci95_low_pp if orientation==1 else -d.ci95_high_pp
        direct_high=d.ci95_high_pp if orientation==1 else -d.ci95_low_pp
        direct_model_rows.append(dict(contrast=r.contrast,stage_target_words=r.stage_target_words,
            model_difference_pp=r.controlled_difference_pp,model_ci_low_pp=r.controlled_ci_low_pp,
            model_ci_high_pp=r.controlled_ci_high_pp,model_n_articles=r.n_supported_articles,model_reliable=r.reliable,
            raw_on_model_support_pp=r.raw_same_articles_pp,direct_5pct_difference_pp=direct_mean,
            direct_5pct_ci_low_pp=direct_low,direct_5pct_ci_high_pp=direct_high,direct_5pct_n=int(d.n),
            direct_5pct_small_below10=bool(0<d.n<10),
            same_point_direction=bool(direct_mean*r.controlled_difference_pp>0)
                if pd.notna(direct_mean) and pd.notna(r.controlled_difference_pp) else np.nan))
    direct_model=pd.DataFrame(direct_model_rows)
    export(direct_model,'E_modell_gegen_direkte_5prozent_paarung_16_kontraste')
    display(direct_model.round(3))
    sensitivity_rows=[]
    for first,second in CONTRASTS:
        for stage in STAGES:
            for definition in SUPPORT_DEFS:
                support=get_support(first,second,stage,definition)
                for kind in models:
                    result=oriented_result(first,second,stage,definition,kind)
                    sensitivity_rows.append(dict(first=first,second=second,stage_target_words=stage,
                        definition=definition,model=kind,n=support['n_supported'],
                        reliable=result is not None and bool(result.reliable),
                        difference_pp=result.adjusted_difference_pp if result is not None else np.nan,
                        ci_low_pp=result.ci_low_pp if result is not None else np.nan,
                        ci_high_pp=result.ci_high_pp if result is not None else np.nan))
    sensitivity=pd.DataFrame(sensitivity_rows)
    export(sensitivity,'E_sensitivitaet_16_kontraste_modelle_und_support')
    model_sensitivity=core_table[['first','second','stage_target_words','controlled_difference_pp']].copy()
    flex=[]
    for row in model_sensitivity.itertuples():
        result=oriented_result(row.first,row.second,row.stage_target_words,kind='spline')
        flex.append(result.adjusted_difference_pp if result is not None else np.nan)
    model_sensitivity['spline_difference_pp']=flex
    model_sensitivity['spline_minus_linear_pp']=model_sensitivity.spline_difference_pp-model_sensitivity.controlled_difference_pp
    model_sensitivity['model_sign_change']=(model_sensitivity.spline_difference_pp*model_sensitivity.controlled_difference_pp)<0
    export(model_sensitivity,'E_linear_gegen_spline_prim_support')
    display(model_sensitivity.round(3))
    comparable=model_sensitivity.dropna(subset=['controlled_difference_pp','spline_difference_pp'])
    display(Markdown(f'Die flexible Form ändert {int(comparable.model_sign_change.sum())}/{len(comparable)} '
        'gestützte Kontrastvorzeichen gegenüber der linearen Form. Dies wird als Formunsicherheit '
        'ausgewiesen; ein kleineres Residualmaß allein ersetzt keine inhaltliche oder unabhängige Validierung.'))
    display(sensitivity.pivot(index=['first','second','stage_target_words'],columns=['definition','model'],values='difference_pp').round(3))
    display(Markdown('**Keine automatische Modellselektion:** Die dokumentierten Unterschiede in Devianz, '
        'numerischer Stabilität und standardisierten Effekten dienen der Formdiagnostik. '
        'Das lineare Hauptmodell bleibt unabhängig davon bestehen, welches Verfahren dadurch günstiger erscheint.'))

    rank_comparison=original_ranks.copy()
    rank_comparison['model_based_rank']=np.nan
    rank_comparison['model_rank_status']='keine ausreichend gestützte gemeinsame Fünfmethoden-Standardisierung'
    global_rows=[]
    for stage,support in global_support.items():
        global_rows.append(dict(stage_target_words=stage,common_lower=support['lower'],common_upper=support['upper'],
            n_supported=support['n_supported'],estimable=support['estimable'],status=support['reason']))
        if support['estimable']:
            # Alle Verfahren auf identischem empirischem Artikel-Wortzahl-Panel standardisieren.
            ids=support['ids']
            observed=article[article.stage_target_words.eq(stage)&article.method.isin(GENERATIVE)&article.article_id.isin(ids)]
            reference=observed[['article_id','W','stage_target_words']].copy()
            values=[]
            for method in GENERATIVE:
                values.append(expit(design(reference.assign(method=method),'linear')@models['linear']['beta']).mean())
            ranks=rank_scores(np.asarray(values),tol=1e-12)
            for method,rank in zip(GENERATIVE,ranks):
                mask=rank_comparison.method.eq(method)&rank_comparison.stage_target_words.eq(stage)
                rank_comparison.loc[mask,'model_based_rank']=int(rank)
                rank_comparison.loc[mask,'model_rank_status']='gemeinsam standardisiert, generative Verfahren'
    export(rank_comparison,'F_originale_heuristische_und_zulaessige_modellraenge')
    export(pd.DataFrame(global_rows),'F_gemeinsamer_fuenfmethoden_support')
    display(rank_comparison[['method','stage_target_words','original_rank_raw','original_rank_adjusted','model_based_rank','model_rank_status']])

    auto_rows=[]
    for other in GENERATIVE:
        if other=='auto_cot':continue
        support=get_support('auto_cot',other,75)
        result=oriented_result('auto_cot',other,75)
        auto_rows.append(dict(comparison='Auto-CoT − '+NAMES[other],n_articles=support['n_supported'],
            reliable=result is not None and bool(result.reliable),difference_pp=result.adjusted_difference_pp if result is not None else np.nan,
            ci_low_pp=result.ci_low_pp if result is not None else np.nan,
            ci_high_pp=result.ci_high_pp if result is not None else np.nan,status=support['reason']))
    auto75=pd.DataFrame(auto_rows)
    export(auto75,'F_auto_cot_75_gegen_alle_generativen_verfahren')
    display(auto75.round(3))
    auto_sensitivity_rows=[]
    for other in GENERATIVE:
        if other=='auto_cot':continue
        for definition in SUPPORT_DEFS:
            for kind in models:
                result=oriented_result('auto_cot',other,75,definition,kind)
                support=get_support('auto_cot',other,75,definition)
                auto_sensitivity_rows.append(dict(comparison='Auto-CoT − '+NAMES[other],definition=definition,
                    model=kind,n_articles=support['n_supported'],reliable=result is not None and bool(result.reliable),
                    difference_pp=result.adjusted_difference_pp if result is not None else np.nan,
                    ci_low_pp=result.ci_low_pp if result is not None else np.nan,
                    ci_high_pp=result.ci_high_pp if result is not None else np.nan))
    auto_sensitivity=pd.DataFrame(auto_sensitivity_rows)
    export(auto_sensitivity,'F_auto_cot_75_support_und_modellsensitivitaet')
    display(auto_sensitivity[auto_sensitivity.comparison.eq('Auto-CoT − Zero-Shot-CoT')].round(3))
    display(Markdown('**Einordnung der 75-Wörter-Frage:** Für Auto-CoT gegenüber Plain bleiben die '
        'modellierten Unterschiede positiv; die vollständigen Modell-/Supportvarianten sind exportiert. '
        'Der Vergleich zu Few-Shot-CoT überdeckt im primären Intervall null. Die Intervallgrenze zu '
        'Automate-CoT liegt sehr nahe null und ist explorativ zu lesen. Zu Zero-Shot-CoT reicht der '
        'primäre gepaarte Support nicht aus; auch die berechenbare breitere Variante ist keine '
        'Bestätigung einer generellen Überlegenheit.'))

    research_questions=[]
    for first,second,stages in [('zero_shot_cot','plain',[500,300]),('automate_cot','auto_cot',STAGES)]:
        for stage in stages:
            row=core_table[(core_table['first']==first)&(core_table.second==second)&core_table.stage_target_words.eq(stage)].iloc[0]
            research_questions.append(dict(question=f'{NAMES[first]} − {NAMES[second]} bei {stage}',
                original_pp=row.original_difference_pp,controlled_pp=row.controlled_difference_pp,
                ci_low_pp=row.controlled_ci_low_pp,ci_high_pp=row.controlled_ci_high_pp,status=row.status))
    export(pd.DataFrame(research_questions),'F_antworten_auf_zentrale_methodenfragen')
    display(pd.DataFrame(research_questions).round(3))

    supported=core_table[core_table.reliable].copy()
    supported['sign_reversal']=supported.original_difference_pp*supported.controlled_difference_pp<0
    supported['smaller_absolute']=supported.controlled_difference_pp.abs()<supported.original_difference_pp.abs()
    summary_rows=[]
    for row in core_table.itertuples():
        summary_rows.append(dict(contrast=row.contrast,stage=row.stage_target_words,
            original_pp=row.original_difference_pp,
            controlled_with_ci=(f'{de(row.controlled_difference_pp)} [{de(row.controlled_ci_low_pp)}; '
                f'{de(row.controlled_ci_high_pp)}]; n={row.n_supported_articles}') if row.reliable else 'nicht zuverlässig schätzbar',
            status=row.status))
    compact=pd.DataFrame(summary_rows).pivot(index='contrast',columns='stage',values='controlled_with_ci').reindex(
        index=[f'{NAMES[f]} − {NAMES[s]}' for f,s in CONTRASTS],columns=STAGES)
    compact.to_csv(OUT/'haupttext_kompakttabelle_4x4.csv',encoding='utf-8-sig')
    display(compact)
    auto_text=[]
    for r in auto75.itertuples():
        if r.reliable:
            auto_text.append(f'{r.comparison}: {de(r.difference_pp)} pp [{de(r.ci_low_pp)}; {de(r.ci_high_pp)}], n={r.n_articles}')
        else:auto_text.append(f'{r.comparison}: nicht zuverlässig schätzbar (n={r.n_articles})')
    comparison_auto=core_table[(core_table['first']=='automate_cot')&(core_table.second=='auto_cot')]
    auto_autom_text='; '.join(f'{r.stage_target_words}: {de(r.controlled_difference_pp)} pp '
        f'[{de(r.controlled_ci_low_pp)}; {de(r.controlled_ci_high_pp)}]' for r in comparison_auto.itertuples() if r.reliable)
    comparison500=comparison_auto[comparison_auto.stage_target_words.eq(500)].iloc[0]
    chapter=(
        'Die längenkontrollierte Auswertung reproduzierte die ursprünglichen Retentions- und Wortzahlmittelwerte '
        'sowie die vordefinierten Kontraste. Die tatsächliche Länge wurde in einem binomialen Modell mit '
        'Methode-Stufe-Zellen und festen Artikeleffekten berücksichtigt; Unsicherheitsintervalle beruhen auf '
        '10.000 vollständigen Artikelbootstrap-Anpassungen je Modell.\n\n'
        f'Für {int(core_table.reliable.sum())}/16 Hauptvergleiche bestand nach den dokumentierten '
        'Überlappungsregeln eine ausreichende Grundlage. Insbesondere Zero-Shot-CoT gegenüber Plain bei '
        '500 und 300 Zielwörtern war nicht zuverlässig längenkontrolliert schätzbar. Die dort beobachteten '
        'Vorteile lassen sich damit weder als längenunabhängig bestätigen noch durch diese Analyse widerlegen. '
        f'Unter den gestützten Vergleichen wurden {int(supported.smaller_absolute.sum())} absolute Unterschiede '
        f'kleiner; {int(supported.sign_reversal.sum())} wechselten gegenüber dem Gesamtrohwert das Vorzeichen. '
        'Dabei ändern sich auch die ausgewählten Artikelpopulationen.\n\n'
        f'Bei Automate-CoT gegenüber Auto-CoT auf Stufe 500 war die Rohdifferenz in der '
        f'Supportstichprobe bereits {de(comparison500.raw_same_articles_pp)} pp; der Vorzeichenwechsel '
        'ist daher nicht allein der rechnerischen Längenkontrolle zuzuschreiben. '
        f'Der Spline änderte {int(comparable.model_sign_change.sum())}/{len(comparable)} gestützte '
        'Kontrastvorzeichen gegenüber der linearen Form.\n\n'
        f'Automate-CoT minus Auto-CoT ergab nach Längenkontrolle: {auto_autom_text}. '
        f'Für Auto-CoT bei 75 Wörtern: '+ '; '.join(auto_text)+'.\n\n'
        f'Eine gemeinsame gestützte Fünfmethoden-Rangfolge war auf '
        f'{sum(s["estimable"] for s in global_support.values())}/4 Stufen möglich. '
        'Die Ergebnisse begrenzen daher pauschale Überlegenheitsaussagen. Die heuristische Längenadjustierung '
        'bleibt eine andere, strafbasierte Kennzahl; Standardisierung ersetzt sie nicht. '
        'Wortlänge ist selbst ein Methodenergebnis, weshalb die Unterschiede explorative bedingte '
        'Zusammenhänge und keine kausalen Effekte darstellen. QA-Messfehler bleiben bestehen.'
    )
    assert len(chapter.split())<=300,len(chapter.split())
    (OUT/'Wissenschaftliche_Zusammenfassung_max300_Woerter.md').write_text(chapter+'\n',encoding='utf-8')
    display(Markdown('### Wissenschaftliche Ergebniszusammenfassung\n\n'+chapter))
    display(Markdown(f'*Automatisch erzeugt: {len(chapter.split())} Wörter.*'))

    for title,frame in [
        ('Alle 24 Kompressionsgruppen',compression),
        ('Alle 60 Methodenpaare × drei Überlappungsdefinitionen',overlap),
        ('Alle gestützten standardisierten Paarvergleiche mit Bootstrap-Intervallen',controlled),
        ('Alle Hauptkontraste nach Modell und Supportdefinition',sensitivity),
        ('Alle direkten 5-%-Paarungen',direct),
        ('Direkte 5-%-Paarung versus Modell: alle 16 Hauptkontraste',direct_model),
        ('Auto-CoT bei 75 Wörtern: alle Modell- und Supportvarianten',auto_sensitivity),
        ('Vollständige Modellkoeffizienten und artikelrobuste Standardfehler',pd.DataFrame(coefficient_rows)),
        ('Modellkonvergenz und numerische Diagnostik',diagnostics_table)]:
        display(HTML('<details><summary>'+html.escape(title)+f' ({len(frame)} Zeilen)</summary>'
            +'<div style="overflow-x:auto">'+frame.to_html(index=False,float_format=lambda x:f'{x:.5g}')
            +'</div></details>'))

    assert reproduction.bestanden.all() and length_reproduction.passed.all()
    assert word_check.abs_difference.le(1e-12).all()
    assert hashlib.sha256(qa[KEY+['label']].to_csv(index=False).encode()).hexdigest()==original_label_digest
    assert (~qa.valid).sum()==196
    assert bootstrap_fits.n_requested.eq(10000).all()
    for row in core_table.itertuples():
        if not row.reliable:assert pd.isna(row.controlled_difference_pp)
        else:assert row.n_supported_articles>=MIN_SUPPORT and row.common_lower<row.common_upper
    for source,digest in sources_before.items():assert sha(source)==digest,source
    run=dict(data_version=VERSION,n_articles=100,n_summaries=2400,n_qa=48000,n_technical_missing=196,
        source_package=str(PACKAGE),source_files_unchanged=True,original_labels_unchanged=True,
        new_llm_calls=0,new_human_ratings=0,summary_generation=False,software=software,
        original_results_reproduced=True,model='grouped binomial GLM, fixed article effects, method x stage, stage-specific word length',
        primary_length_function='linear raw word count per 100 words',sensitivity_length_function='stage-specific centered natural cubic spline df=3',
        bootstrap_samples=10000,bootstrap_full_refits_per_model=True,bootstrap_unit='article',seed=SEED,
        support_selection_fixed_in_bootstrap=True,standardization_distribution_resampled=True,
        primary_support='intersection p05-p95 plus local support and at least 20 paired articles',
        n_supported_core_contrasts=int(core_table.reliable.sum()),n_unreliable_core_contrasts=int((~core_table.reliable).sum()),
        n_core_sign_reversals=int(supported.sign_reversal.sum()),
        n_global_generativestage_rankings=sum(s['estimable'] for s in global_support.values()))
    (OUT/'ausfuehrungsprotokoll.json').write_text(json.dumps(run,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    display(Markdown('**Abgeschlossen:** Originalwerte reproduziert, Eingabe-/Ausgabelängen und '
        'QA-Nenner geprüft, Modell- und Bootstrapdiagnostik dokumentiert, alle verwendeten '
        'Quellenhashes unverändert. Keine ursprünglichen Daten oder Klassifikationen bearbeitet.'))
    display(pd.DataFrame([{'Datei':str(p.relative_to(ROOT)),'Bytes':p.stat().st_size}
                         for p in sorted(OUT.rglob('*')) if p.is_file()]))

if __name__ == '__main__':
    from offline_guard import install
    install(allow_statistics_workers='--recompute-regression-bootstrap' in sys.argv)
    main(sys.argv[1], '--recompute-regression-bootstrap' in sys.argv, int(sys.argv[2]))
