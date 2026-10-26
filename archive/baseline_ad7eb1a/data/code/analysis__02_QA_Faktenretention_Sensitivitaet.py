# Statische Originalauswertung; Eingaben/Ausgaben relativ zur übergebenen Arbeitskopie.
# Keine LLM-Aufrufe. Standard: vorhandene Regressionscaches verwenden.
from pathlib import Path
import os,sys

def main(work, recompute_bootstrap=False, workers=1):
    WORK=__import__('pathlib').Path(work).resolve()
    __import__('os').chdir(WORK/('supplement' if '02_QA_Faktenretention_Sensitivitaet'[0].isdigit() else 'project/02_Auswertung'))
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
    ROOT = BUNDLE/'ergebnisse/02_sensitivitaet'
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
    SIM_SEED, REPEATS, BATCH = 32026, 5_000, 500
    RATES = [.005,.01,.02,.03,.05,.10]
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
                for m in ['numpy','pandas','matplotlib','nbformat','nbclient','nbconvert','ipykernel']}}
    display(pd.Series(software,name='Version').to_frame())

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

    qa_candidates=qa.merge(article[CELL+['n_valid']],on=CELL,validate='many_to_one').copy()
    qa_candidates['method_order']=qa_candidates.method.map(dict(zip(METHODS,range(6))))
    article_lookup=article.set_index(CELL)

    def exact_difference(first,second,stage):
        total=Fraction(0)
        for a in IDS:
            left=article_lookup.loc[(a,first,stage)]
            right=article_lookup.loc[(a,second,stage)]
            total+=Fraction(int(left.n_retained),int(left.n_valid))-Fraction(int(right.n_retained),int(right.n_valid))
        return total*Fraction(100,len(IDS))

    def threshold_count(weights,gap,strict=False):
        cumulative=Fraction(0)
        for i,w in enumerate(weights,1):
            cumulative+=w
            if cumulative>gap if strict else cumulative>=gap:
                return i
        return None

    def tied_solution(ordered,k,strategy):
        # Nur die Reihenfolge innerhalb exakt gleich gewichteter Tiers ändert sich.
        selected=[]
        use=Counter()
        remaining=k
        for denominator,tier in ordered.groupby('n_valid',sort=True):
            take=min(remaining,len(tier))
            if take==0: break
            if strategy=='canonical':
                chosen=tier.head(take).index.tolist()
            else:
                groups={a:g.index.tolist() for a,g in tier.groupby('article_id',sort=True)}
                chosen=[]
                for _ in range(take):
                    available=[a for a,indices in groups.items() if indices]
                    if strategy=='compact':
                        a=min(available,key=lambda a:(not bool(use[a]),-len(groups[a]),a))
                    else:
                        a=min(available,key=lambda a:(use[a],a))
                    chosen.append(groups[a].pop(0))
                    use[a]+=1
            selected.extend(chosen)
            if strategy=='canonical': use.update(ordered.loc[chosen,'article_id'])
            remaining-=take
        assert remaining==0
        return ordered.loc[selected].copy()

    attack_rows=[]
    attack_plans=[]
    budget_rows=[]
    for first,second in CONTRASTS:
        for stage in STAGES:
            diff=exact_difference(first,second,stage)
            assert diff!=0, 'Für Originalgleichstand muss die Zieldefinition separat festgelegt werden.'
            sign=1 if diff>0 else -1
            pool=qa_candidates[qa_candidates.method.isin([first,second]) &
                 qa_candidates.stage_target_words.eq(stage) & qa_candidates.valid].copy()
            pool['hypothetical_binary']=np.where(pool.method.eq(first),0 if sign>0 else 1,1 if sign>0 else 0)
            pool=pool[pool.retained_binary.ne(pool.hypothetical_binary)].copy()
            pool=pool.sort_values(['n_valid','article_id','method_order','canonical_fact_id'])
            weights=[Fraction(100,len(IDS)*int(v)) for v in pool.n_valid]
            gap=abs(diff)
            k_nonpositive=threshold_count(weights,gap)
            k_reverse=threshold_count(weights,gap,strict=True)
            assert k_nonpositive is not None and k_reverse is not None
            assert sum(weights[:k_nonpositive],Fraction(0))>=gap
            assert sum(weights[:max(0,k_nonpositive-1)],Fraction(0))<gap
            assert sum(weights[:k_reverse],Fraction(0))>gap
            assert sum(weights[:max(0,k_reverse-1)],Fraction(0))<=gap
            denominator=int(article[article.method.isin([first,second])&article.stage_target_words.eq(stage)].n_valid.sum())
            article_counts=[]
            combined=None
            for strategy in ['canonical','compact','spread']:
                solution=tied_solution(pool,k_reverse,strategy)
                impact=sum((Fraction(100,len(IDS)*int(v)) for v in solution.n_valid),Fraction(0))
                after=diff-sign*impact
                assert after*diff<0
                # Lösung auf separater Kopie der Artikelzähler anwenden und exakt nachrechnen.
                deltas=solution.assign(change=solution.hypothetical_binary-solution.retained_binary).groupby(['article_id','method']).change.sum()
                after_direct=Fraction(0)
                for a in IDS:
                    l=article_lookup.loc[(a,first,stage)]
                    r=article_lookup.loc[(a,second,stage)]
                    kl=int(l.n_retained)+int(deltas.get((a,first),0))
                    kr=int(r.n_retained)+int(deltas.get((a,second),0))
                    assert 0<=kl<=l.n_valid and 0<=kr<=r.n_valid
                    after_direct+=Fraction(kl,int(l.n_valid))-Fraction(kr,int(r.n_valid))
                assert after_direct*Fraction(100,len(IDS))==after
                article_counts.append(solution.article_id.nunique())
                plan=solution[KEY+['label','retained_binary','hypothetical_binary','n_valid','question','reference_answer']].copy()
                plan['contrast_first']=first; plan['contrast_second']=second; plan['strategy']=strategy
                plan['change_number']=np.arange(1,len(plan)+1)
                plan['effect_pp']=100/(len(IDS)*plan.n_valid)
                plan['original_difference_pp']=float(diff)
                attack_plans.append(plan)
                if strategy=='canonical':
                    combined=solution
                    canonical_after=after
            first_weights=[Fraction(100,len(IDS)*int(v)) for v in pool.loc[pool.method.eq(first),'n_valid']]
            second_weights=[Fraction(100,len(IDS)*int(v)) for v in pool.loc[pool.method.eq(second),'n_valid']]
            attack_rows.append(dict(first=first,second=second,contrast=f'{NAMES[first]} − {NAMES[second]}',
                stage_target_words=stage,original_difference_pp=float(diff),original_difference_exact=str(diff),
                min_tie_or_cross=k_nonpositive,min_strict_reverse=k_reverse,
                exact_tie_in_optimal_prefix=sum(weights[:k_nonpositive],Fraction(0))==gap,
                difference_after_min_tie_or_cross_pp=float(diff-sign*sum(weights[:k_nonpositive],Fraction(0))),
                difference_after_strict_reverse_pp=float(canonical_after),
                valid_comparison_judgments=denominator,min_reverse_percent_valid=pct(k_reverse,denominator),
                n_changes_first=int(combined.method.eq(first).sum()),n_changes_second=int(combined.method.eq(second).sum()),
                n_articles_canonical=article_counts[0],n_articles_compact=article_counts[1],n_articles_spread=article_counts[2],
                n_articles_min_found=min(article_counts),n_articles_max_found=max(article_counts),
                min_reverse_only_first=threshold_count(first_weights,gap,strict=True),
                min_reverse_only_second=threshold_count(second_weights,gap,strict=True)))
            for budget in [10,20,50,100]:
                actual=min(budget,len(weights))
                impact=sum(weights[:actual],Fraction(0))
                budget_rows.append(dict(first=first,second=second,stage_target_words=stage,budget=budget,
                    used_changes=actual,removed_gap_pp=float(impact),
                    original_gap_closed_percent=min(100,float(100*impact/gap)),
                    difference_after_pp=float(diff-sign*impact),strict_reverse=impact>gap))
    attacks=pd.DataFrame(attack_rows)
    plans=pd.concat(attack_plans,ignore_index=True)
    budgets=pd.DataFrame(budget_rows)
    export(attacks,'A_minimale_adversariale_aenderungen')
    export(plans,'A_konkrete_hypothetische_aenderungsplaene')
    export(budgets,'A_wirkung_erster_10_20_50_100_aenderungen')
    display(attacks[['contrast','stage_target_words','original_difference_pp','min_tie_or_cross','min_strict_reverse',
        'min_reverse_percent_valid','n_changes_first','n_changes_second','n_articles_compact','n_articles_spread']].round(3))
    display(Markdown('**Nenner:** jeweils alle gültigen QA-Urteile beider Verfahren auf der betreffenden Stufe. '
        'Prozentanteile sind Worst-Case-Änderungsbudgets, keine geschätzten Fehlerraten. '
        'Die vier Budgetauswertungen deckeln den aufgehobenen Anteil der ursprünglichen Differenz bei 100 %; '
        'die zusätzliche Spalte `difference_after_pp` zeigt ein etwaiges Überschreiten.'))
    widest=attacks.assign(article_span=attacks.n_articles_max_found-attacks.n_articles_min_found).sort_values('article_span',ascending=False).iloc[0]
    display(Markdown(f'**Alternative gleich minimale Lösungen:** Für {widest.contrast} bei '
        f'{int(widest.stage_target_words)} Wörtern betreffen die {int(widest.min_strict_reverse)} Änderungen '
        f'je nach Gleichstandsauswahl {int(widest.n_articles_min_found)} bis {int(widest.n_articles_max_found)} Artikel. '
        'Diese gefundenen Beispiele zeigen, dass die minimale Urteilszahl allein keine eindeutige Zahl '
        'betroffener Artikel festlegt.'))

    plot_rows=attacks.merge(original_contrasts[['first','second','stage_target_words','ci_low','ci_high']],
        on=['first','second','stage_target_words'],validate='one_to_one')
    labels=[f'{NAMES[r.first]} − {NAMES[r.second]} | {r.stage_target_words}' for r in plot_rows.itertuples()]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(13,8.2),sharey=True,gridspec_kw={'width_ratios':[1.15,1]},layout='constrained')
    ys=np.arange(len(plot_rows))
    for y,r in zip(ys,plot_rows.itertuples()):
        ax.errorbar(r.original_difference_pp,y,xerr=[[r.original_difference_pp-r.ci_low],[r.ci_high-r.original_difference_pp]],
            fmt='o',color=STAGE_COLORS[r.stage_target_words],capsize=3)
        bx.barh(y,r.min_strict_reverse,color=STAGE_COLORS[r.stage_target_words],height=.65)
        bx.text(r.min_strict_reverse+3,y,f'{r.min_strict_reverse} ({de(r.min_reverse_percent_valid)} %)',va='center',fontsize=9)
    ax.axvline(0,color='#555555',lw=1,ls='--')
    ax.set(yticks=ys,yticklabels=labels,xlabel='Originaldifferenz (pp); 95-%-Artikelbootstrap')
    ax.invert_yaxis()
    ax.set_title('Beobachtete Methodenunterschiede')
    bx.set(xlabel='Minimale gezielte Änderungen zur strikten Umkehr',xlim=(0,plot_rows.min_strict_reverse.max()*1.42))
    bx.set_title('Adversarialer Worst Case\nKlammer: Anteil an gültigen Urteilen beider Verfahren')
    fig.suptitle('Robustheit der 16 vordefinierten Vergleichsergebnisse',fontsize=14)
    savefig(fig,'01_originaldifferenzen_und_minimale_aenderungen')

    scenarios=[]
    def add_scenario(sid,family,p_down,p_up,dependence='independent',focal='',extra=0.0):
        down=np.broadcast_to(np.asarray(p_down,dtype=float),(6,)).copy()
        up=np.broadcast_to(np.asarray(p_up,dtype=float),(6,)).copy()
        sid_seed=int.from_bytes(hashlib.sha256(sid.encode()).digest()[:4],'little')
        seed=int(np.random.SeedSequence([SIM_SEED,sid_seed]).generate_state(1)[0])
        scenarios.append(dict(id=sid,family=family,down=down,up=up,dependence=dependence,
            focal=focal,extra=extra,seed=seed))
    for p in RATES:
        add_scenario(f'sym_{p:.3f}','symmetrisch',p,p)
    for down,up in [(.01,.03),(.03,.01),(.02,.05),(.05,.02),(0,.05),(.05,0)]:
        add_scenario(f'asym_{down:.3f}_{up:.3f}','asymmetrisch',down,up)
    for focal in METHODS:
        for extra in [.01,.02,.03]:
            down=np.full(6,.02);down[METHODS.index(focal)]+=extra
            add_scenario(f'method_{focal}_{extra:.2f}','verfahrensabhaengig',down,.02,focal=focal,extra=extra)
    for p in RATES:
        add_scenario(f'article_{p:.3f}','artikelkorreliert',p,p,dependence='article')
    scenario_by_id={s['id']:s for s in scenarios}
    assert len(scenarios)==36
    scenario_rows=[]
    for s in scenarios:
        for m,down,up in zip(METHODS,s['down'],s['up']):
            scenario_rows.append(dict(scenario=s['id'],family=s['family'],method=m,
                p_R_to_N=down,p_N_to_R=up,dependence=s['dependence'],focal_method=s['focal'],
                extra_adverse_pp=100*s['extra'],seed=s['seed'],n_repeats=REPEATS))
    scenario_table=pd.DataFrame(scenario_rows)
    export(scenario_table,'B_szenariodefinitionen_und_seeds')
    display(scenario_table.drop_duplicates('scenario')[['scenario','family','dependence','focal_method','extra_adverse_pp','seed','n_repeats']])

    def simulate(scenario,repeats=REPEATS):
        rng=np.random.default_rng(scenario['seed'])
        down=scenario['down'][None,None,:,None]
        up=scenario['up'][None,None,:,None]
        raw_scores=np.empty((repeats,6,4),dtype=np.int64)
        adj_means=np.empty((repeats,6,4),dtype=float)
        down_totals=np.empty(repeats,dtype=np.int64)
        up_totals=np.empty(repeats,dtype=np.int64)
        for start in range(0,repeats,BATCH):
            stop=min(start+BATCH,repeats)
            size=stop-start
            if scenario['dependence']=='article':
                assert np.all(scenario['down']==scenario['down'][0]) and np.all(scenario['up']==scenario['down'][0])
                flipped=(rng.random((size,len(IDS),1,1))<scenario['down'][0])
                n_down=flipped*K[None,:,:,:]
                n_up=flipped*(V-K)[None,:,:,:]
            else:
                n_down=rng.binomial(K[None,:,:,:],down,size=(size,len(IDS),6,4))
                n_up=rng.binomial((V-K)[None,:,:,:],up,size=(size,len(IDS),6,4))
            new_counts=K[None,:,:,:]-n_down+n_up
            assert np.all(new_counts>=0) and np.all(new_counts<=V[None,:,:,:])
            raw_scores[start:stop]=(new_counts*RAW_SCALE).sum(axis=1)
            adj_means[start:stop]=(new_counts/20*PENALTY).mean(axis=1)
            down_totals[start:stop]=n_down.sum(axis=(1,2,3))
            up_totals[start:stop]=n_up.sum(axis=(1,2,3))
        return dict(raw_scores=raw_scores,adjusted=adj_means,n_down=down_totals,n_up=up_totals)

    # Aussagekräftige Randprüfungen: keine Änderungen bzw. vollständiges Umkehren aller gültigen Positionen.
    for probability in [0.,1.]:
        boundary=dict(id='boundary',down=np.full(6,probability),up=np.full(6,probability),dependence='independent',seed=123)
        result=simulate(boundary,repeats=4)
        expected_counts=K if probability==0 else V-K
        assert np.array_equal(result['raw_scores'],np.broadcast_to((expected_counts*RAW_SCALE).sum(axis=0),(4,6,4)))
        assert np.allclose(result['adjusted'],(expected_counts/20*PENALTY).mean(axis=0))
        assert np.all(result['n_down']+result['n_up']==(0 if probability==0 else qa.valid.sum()))

    simulations={}
    mc_check_rows=[]
    for s in scenarios:
        result=simulate(s)
        simulations[s['id']]=result
        expected_counts=K*(1-s['down'][None,:,None])+(V-K)*s['up'][None,:,None]
        expected_raw=(expected_counts/V).mean(axis=0)
        expected_adj=(expected_counts/20*PENALTY).mean(axis=0)
        for metric,draws,expectation in [('raw',result['raw_scores']/(LCM*len(IDS)),expected_raw),
                                         ('adjusted',result['adjusted'],expected_adj)]:
            standard_error=draws.std(axis=0,ddof=1)/np.sqrt(REPEATS)
            difference=np.abs(draws.mean(axis=0)-expectation)
            z=np.divide(difference,standard_error,out=np.zeros_like(difference),where=standard_error>0)
            mc_check_rows.append(dict(scenario=s['id'],metric=metric,
                max_abs_mean_deviation_pp=float(100*difference.max()),max_MC_standard_errors=float(z.max()),
                analytic_expectation_check=bool(np.all(difference<=6*standard_error+1e-12))))
    mc_checks=pd.DataFrame(mc_check_rows)
    export(mc_checks,'B_abgleich_analytische_erwartungen')
    assert mc_checks.analytic_expectation_check.all(), 'Monte-Carlo-Mittelwerte unerwartet weit vom analytischen Erwartungswert.'
    scenario_realization=pd.DataFrame([dict(scenario=s['id'],n_repeats=REPEATS,
        n_valid_fixed=int(qa.valid.sum()),mean_flips=float((simulations[s['id']]['n_down']+simulations[s['id']]['n_up']).mean()),
        mean_flip_percent=pct((simulations[s['id']]['n_down']+simulations[s['id']]['n_up']).mean(),int(qa.valid.sum())),
        mean_R_to_N=float(simulations[s['id']]['n_down'].mean()),mean_N_to_R=float(simulations[s['id']]['n_up'].mean()))
        for s in scenarios])
    export(scenario_realization,'B_tatsaechlich_simulierte_aenderungszahlen')
    pass  # Große Bootstrap-Zwischendatei wird nicht benötigt.
    display(Markdown(f'**{len(scenarios)} Szenarien × {REPEATS:,} Wiederholungen abgeschlossen.** '
        'Alle Änderungen betreffen nur gültige temporäre Binärwerte. Randfälle und analytische Erwartungen geprüft.'))

    def direction(x):
        return np.where(np.abs(x)<=1e-10,0,np.sign(x)).astype(int)

    contrast_sim_rows=[]
    group_sim_rows=[]
    for s in scenarios:
        sid=s['id'];result=simulations[sid]
        for metric,draws,original in [('raw',result['raw_scores']/(LCM*len(IDS)),ORIG_RAW),
                                       ('adjusted',result['adjusted'],ORIG_ADJ)]:
            for mi,m in enumerate(METHODS):
                for ti,t in enumerate(STAGES):
                    values=100*draws[:,mi,ti]
                    group_sim_rows.append(dict(scenario=sid,family=s['family'],metric=metric,method=m,
                        stage_target_words=t,original_percent=100*original[mi,ti],simulation_mean_percent=float(values.mean()),
                        sim_q025=float(np.quantile(values,.025)),sim_q975=float(np.quantile(values,.975)),
                        mean_MC_se_pp=float(values.std(ddof=1)/np.sqrt(REPEATS)),n_repeats=REPEATS))
            for first,second in CONTRASTS:
                i,j=METHODS.index(first),METHODS.index(second)
                for ti,t in enumerate(STAGES):
                    values=100*(draws[:,i,ti]-draws[:,j,ti])
                    orig=100*(original[i,ti]-original[j,ti])
                    dirs=direction(values);sign=int(direction(orig))
                    n_rev=int(np.sum(dirs*sign<0));n_tie=int(np.sum(dirs==0));n_same=int(np.sum(dirs==sign))
                    assert n_rev+n_tie+n_same==REPEATS
                    p=n_rev/REPEATS
                    contrast_sim_rows.append(dict(scenario=sid,family=s['family'],metric=metric,first=first,second=second,
                        stage_target_words=t,original_difference_pp=orig,simulation_mean_difference_pp=float(values.mean()),
                        sim_q025_pp=float(np.quantile(values,.025)),sim_q975_pp=float(np.quantile(values,.975)),
                        mean_MC_se_pp=float(values.std(ddof=1)/np.sqrt(REPEATS)),
                        n_reversals=n_rev,n_ties=n_tie,n_same_direction=n_same,n_repeats=REPEATS,
                        reversal_frequency_percent=100*p,tie_frequency_percent=pct(n_tie,REPEATS),
                        reversal_frequency_MC_se_pp=100*np.sqrt(p*(1-p)/REPEATS)))
    sim_contrasts=pd.DataFrame(contrast_sim_rows)
    sim_groups=pd.DataFrame(group_sim_rows)
    export(sim_contrasts,'B_simulierte_kontraste_und_vorzeichenwechsel')
    export(sim_groups,'B_simulierte_retention_mittelwerte')
    selected_table=sim_contrasts[sim_contrasts.metric.eq('raw') & sim_contrasts.scenario.isin(
        ['sym_0.050','article_0.050','method_auto_cot_0.03'])]
    display(selected_table[['scenario','first','second','stage_target_words','original_difference_pp',
        'simulation_mean_difference_pp','sim_q025_pp','sim_q975_pp','reversal_frequency_percent']].round(3))

    fig,axes=plt.subplots(2,2,figsize=(12.8,8.2),layout='constrained')
    for ax,(first,second) in zip(axes.flat,CONTRASTS):
        for stage in STAGES:
            rows=sim_contrasts[(sim_contrasts.metric=='raw') & (sim_contrasts.family=='symmetrisch') &
                (sim_contrasts['first']==first) & (sim_contrasts.second==second) & sim_contrasts.stage_target_words.eq(stage)].copy()
            rows['rate']=[100*scenario_by_id[sid]['down'][0] for sid in rows.scenario]
            rows=rows.sort_values('rate')
            original=float(rows.original_difference_pp.iloc[0])
            ax.plot([0]+rows.rate.tolist(),[original]+rows.simulation_mean_difference_pp.tolist(),
                marker='o',ms=4,color=STAGE_COLORS[stage],label=f'{stage} Wörter')
            ax.fill_between(rows.rate.to_numpy(),rows.sim_q025_pp.to_numpy(),rows.sim_q975_pp.to_numpy(),
                color=STAGE_COLORS[stage],alpha=.12)
            ax.scatter([0],[original],color=STAGE_COLORS[stage],s=60,marker='D',zorder=3)
        ax.axhline(0,color='#555555',lw=1,ls='--')
        ax.set(xlabel='Hypothetische Änderungsrate je Richtung (%)',ylabel='Mittlere Differenz (pp)',xticks=[0,1,2,3,5,10])
        ax.set_title(f'{NAMES[first]} − {NAMES[second]}')
        ax.grid(axis='y',alpha=.15)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle('Beobachteter Ausgangspunkt (◆) und hypothetische symmetrische Änderungen\n'
                 'Kurven: Monte-Carlo-Mittel; Bänder: 95-%-Simulationsbereich, 5.000 Wiederholungen',fontsize=13)
    savefig(fig,'02_symmetrische_szenarien_methodendifferenzen')

    scenario_labels = {
        'sym_0.010':'Symmetrisch 1 %', 'sym_0.050':'Symmetrisch 5 %', 'sym_0.100':'Symmetrisch 10 %',
        'article_0.050':'Artikelkorreliert 5 %',
        **{f'method_{m}_0.03':f'Zusatz R→N: {NAMES[m]}' for m in METHODS[1:]}}
    fig, axes = plt.subplots(2,2,figsize=(14,11),layout='constrained')
    scenario_plot_rows=[]
    for ax,(first,second) in zip(axes.flat, CONTRASTS):
        sub=sim_contrasts[sim_contrasts.metric.eq('raw') & sim_contrasts['first'].eq(first) & sim_contrasts.second.eq(second)]
        matrix=sub.pivot(index='scenario',columns='stage_target_words',values='reversal_frequency_percent').reindex(
            index=list(scenario_labels),columns=STAGES)
        assert matrix.notna().all().all()
        im=ax.imshow(matrix.to_numpy(),cmap='Blues',vmin=0,vmax=100,aspect='auto')
        for i,sid in enumerate(matrix.index):
            for j,stage in enumerate(STAGES):
                value=matrix.iloc[i,j]
                ax.text(j,i,f'{value:.1f} %',ha='center',va='center',color='white' if value>55 else '#222222',fontsize=10)
                scenario_plot_rows.append(dict(first=first,second=second,scenario=sid,stage_target_words=stage,
                    reversal_frequency_percent=value,n_repeats=REPEATS))
        ax.set(xticks=range(4),xticklabels=STAGES,yticks=range(len(matrix)),yticklabels=list(scenario_labels.values()),
               xlabel='Zielwortzahl',title=f'{NAMES[first]} − {NAMES[second]}')
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.6,label='Strikte Vorzeichenumkehr in Simulationen (%)')
    fig.suptitle('Vorzeichenrobustheit im Vergleich hypothetischer Szenarien · Hauptmetrik',fontsize=14)
    export(pd.DataFrame(scenario_plot_rows),'grafik_szenarienvergleich_vorzeichenumkehr')
    savefig(fig,'04_szenarienvergleich_vorzeichenumkehr')
    worst=pd.DataFrame(scenario_plot_rows).sort_values('reversal_frequency_percent',ascending=False).iloc[0]
    display(Markdown(f'**Beobachtung:** Der höchste Anteil strikter Vorzeichenumkehr in dieser Szenarioauswahl beträgt '
        f'{de(worst.reversal_frequency_percent,1)} % bei {NAMES[worst["first"]]} − {NAMES[worst.second]} '
        f'auf der {worst.stage_target_words}-Wort-Stufe ({scenario_labels[worst.scenario]}). '
        'Das Ergebnis gilt nur unter der bezeichneten hypothetischen Änderungsannahme.'))

    rank_frequency_rows=[]
    leader_rows=[]
    length_rank_rows=[]
    length_pair_rows=[]
    for s in scenarios:
        sid=s['id'];result=simulations[sid]
        ranks={
            'raw':rank_scores(result['raw_scores'].transpose(0,2,1)).transpose(0,2,1),
            'adjusted':rank_scores(result['adjusted'].transpose(0,2,1),tol=1e-12).transpose(0,2,1)}
        for metric,original_scores,original_rank in [('raw',ORIG_RAW,original_ranks_raw),('adjusted',ORIG_ADJ,original_ranks_adj)]:
            sim_rank=ranks[metric]
            for mi,m in enumerate(METHODS):
                for ti,t in enumerate(STAGES):
                    for rank in range(1,7):
                        n=int(np.sum(sim_rank[:,mi,ti]==rank))
                        rank_frequency_rows.append(dict(scenario=sid,family=s['family'],metric=metric,method=m,
                            stage_target_words=t,rank=rank,n=n,n_repeats=REPEATS,frequency_percent=pct(n,REPEATS)))
            for ti,t in enumerate(STAGES):
                leader_indices=np.flatnonzero(original_rank[:,ti]==1)
                unique=(sim_rank[:,:,ti]==1).sum(axis=1)==1
                for mi in leader_indices:
                    kept=sim_rank[:,mi,ti]==1
                    leader_rows.append(dict(scenario=sid,family=s['family'],metric=metric,stage_target_words=t,
                        original_leader=METHODS[mi],n_repeats=REPEATS,
                        rank1_kept_percent=pct(kept.sum(),REPEATS),
                        sole_rank1_kept_percent=pct((kept&unique).sum(),REPEATS),
                        full_original_rank_vector_percent=pct(np.all(sim_rank[:,:,ti]==original_rank[:,ti],axis=1).sum(),REPEATS)))
        for mi,m in enumerate(METHODS):
            for ti,t in enumerate(STAGES):
                original_change=int(np.sign(original_ranks_adj[mi,ti]-original_ranks_raw[mi,ti]))
                sim_change=np.sign(ranks['adjusted'][:,mi,ti]-ranks['raw'][:,mi,ti])
                length_rank_rows.append(dict(scenario=sid,family=s['family'],method=m,stage_target_words=t,
                    original_raw_rank=int(original_ranks_raw[mi,ti]),original_adjusted_rank=int(original_ranks_adj[mi,ti]),
                    original_rank_change_direction=original_change,
                    same_adjustment_direction_percent=pct(np.sum(sim_change==original_change),REPEATS),
                    same_original_rank_pair_percent=pct(np.sum((ranks['raw'][:,mi,ti]==original_ranks_raw[mi,ti]) &
                        (ranks['adjusted'][:,mi,ti]==original_ranks_adj[mi,ti])),REPEATS)))
        for i,j in combinations(range(6),2):
            for ti,t in enumerate(STAGES):
                raw0=int(direction(ORIG_RAW[i,ti]-ORIG_RAW[j,ti]))
                adj0=int(direction(ORIG_ADJ[i,ti]-ORIG_ADJ[j,ti]))
                raw_dir=direction(result['raw_scores'][:,i,ti]-result['raw_scores'][:,j,ti])
                adj_dir=direction(result['adjusted'][:,i,ti]-result['adjusted'][:,j,ti])
                length_pair_rows.append(dict(scenario=sid,family=s['family'],first=METHODS[i],second=METHODS[j],
                    stage_target_words=t,original_raw_direction=raw0,original_adjusted_direction=adj0,
                    originally_inverted_by_adjustment=raw0*adj0<0,
                    simulation_inverted_by_adjustment_percent=pct(np.sum(raw_dir*adj_dir<0),REPEATS),
                    both_original_directions_kept_percent=pct(np.sum((raw_dir==raw0)&(adj_dir==adj0)),REPEATS),
                    n_repeats=REPEATS))
    rank_frequencies=pd.DataFrame(rank_frequency_rows)
    leaders=pd.DataFrame(leader_rows)
    length_rank_stability=pd.DataFrame(length_rank_rows)
    length_pair_stability=pd.DataFrame(length_pair_rows)
    assert rank_frequencies.groupby(['scenario','metric','method','stage_target_words']).n.sum().eq(REPEATS).all()
    export(rank_frequencies,'C_ranghaeufigkeiten_alle_szenarien')
    export(leaders,'C_originalfuehrung_und_gesamtrangfolge')
    export(length_rank_stability,'C_laengenadjustierung_rangaenderungen')
    export(length_pair_stability,'C_laengenadjustierung_alle_15_paarordnungen')
    display(leaders[leaders.scenario.isin(['sym_0.050','article_0.050','method_auto_cot_0.03'])][
        ['scenario','metric','stage_target_words','original_leader','rank1_kept_percent','sole_rank1_kept_percent','full_original_rank_vector_percent']].round(2))
    display(length_pair_stability[length_pair_stability.originally_inverted_by_adjustment &
        length_pair_stability.scenario.eq('sym_0.050')][['first','second','stage_target_words',
            'both_original_directions_kept_percent','simulation_inverted_by_adjustment_percent']].round(2))
    adjusted75=leaders[(leaders.scenario=='sym_0.050')&(leaders.metric=='adjusted')&leaders.stage_target_words.eq(75)].iloc[0]
    raw75=leaders[(leaders.scenario=='sym_0.050')&(leaders.metric=='raw')&leaders.stage_target_words.eq(75)].iloc[0]
    display(Markdown(f'**Metrikabhängige Rangstabilität:** Bei 75 Wörtern behält '
        f'{NAMES[adjusted75.original_leader]} im unabhängigen symmetrischen 5-%-Szenario '
        f'die adjustierte Führung in {de(adjusted75.rank1_kept_percent)} % der Simulationen; '
        f'bei der Hauptmetrik sind es {de(raw75.rank1_kept_percent)} %. '
        'Die Differenz hängt auch mit dem verfahrensabhängigen additiven Erwartungsterm der '
        'adjustierten Kennzahl zusammen und darf nicht als beobachteter Scorer-Bias interpretiert werden.'))

    SELECTED=['sym_0.050','article_0.050','method_auto_cot_0.03']
    SCENARIO_TITLES=['Unabhängig: 5 % je Richtung','Artikelkorreliert: 5 %',
                     'Auto-CoT zusätzlich benachteiligt\nBasis 2 %, zusätzlich R→N +3 pp']
    def plot_rank_heatmaps(metric,name):
        fig,axes=plt.subplots(4,3,figsize=(13.5,13.5),layout='constrained')
        for ti,t in enumerate(STAGES):
            for si,sid in enumerate(SELECTED):
                ax=axes[ti,si]
                matrix=rank_frequencies[(rank_frequencies.scenario==sid)&(rank_frequencies.metric==metric)&
                    rank_frequencies.stage_target_words.eq(t)].pivot(index='method',columns='rank',values='frequency_percent').reindex(METHODS).to_numpy()
                image=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=100,aspect='auto')
                for i in range(6):
                    for j in range(6):
                        value=matrix[i,j]
                        text='·' if value==0 else ('<0,1' if value<.1 else de(value,1))
                        ax.text(j,i,text,ha='center',va='center',fontsize=8.5,color='white' if value>=55 else '#222222')
                ax.set(xticks=range(6),xticklabels=range(1,7),yticks=range(6),
                    yticklabels=[NAMES[m] for m in METHODS] if si==0 else [])
                ax.tick_params(length=0)
                ax.set_title(f'{t} Wörter'+ ('\n'+SCENARIO_TITLES[si] if ti==0 else ''),fontsize=10)
                if ti==3: ax.set_xlabel('Simulierter Wettbewerbsrang')
        fig.colorbar(image,ax=axes.ravel().tolist(),shrink=.5,label='Anteil der 5.000 Simulationen (%)')
        title='QA-Retention mit gültigem Nenner' if metric=='raw' else 'Längenadjustierte Retention mit unveränderter Strafe'
        fig.suptitle(title+'\nRanghäufigkeiten unter hypothetischen Labeländerungen; keine Überlegenheitswahrscheinlichkeiten',fontsize=13)
        savefig(fig,name)
    plot_rank_heatmaps('raw','03a_rangrobustheit_qa_retention')
    plot_rank_heatmaps('adjusted','03b_rangrobustheit_laengenadjustiert')

    PERMISSIVE_RAW=((K+DISCORD)/V).mean(axis=0)
    PERMISSIVE_ADJ=((K+DISCORD)/20*PENALTY).mean(axis=0)
    assert np.all(K+DISCORD<=V)
    deterministic_rows=[]
    deterministic_contrasts=[]
    for metric,conservative,permissive in [('raw',ORIG_RAW,PERMISSIVE_RAW),('adjusted',ORIG_ADJ,PERMISSIVE_ADJ)]:
        conservative_ranks=rank_scores(conservative.T,tol=1e-12).T
        permissive_ranks=rank_scores(permissive.T,tol=1e-12).T
        for mi,m in enumerate(METHODS):
            for ti,t in enumerate(STAGES):
                deterministic_rows.append(dict(metric=metric,method=m,stage_target_words=t,
                    conservative_percent=100*conservative[mi,ti],permissive_percent=100*permissive[mi,ti],
                    change_pp=100*(permissive[mi,ti]-conservative[mi,ti]),
                    conservative_rank=int(conservative_ranks[mi,ti]),permissive_rank=int(permissive_ranks[mi,ti]),
                    n_discordant=int(DISCORD[:,mi,ti].sum()),n_valid_fixed=int(V[:,mi,ti].sum())))
        for first,second in CONTRASTS:
            i,j=METHODS.index(first),METHODS.index(second)
            for ti,t in enumerate(STAGES):
                old=100*(conservative[i,ti]-conservative[j,ti])
                new=100*(permissive[i,ti]-permissive[j,ti])
                deterministic_contrasts.append(dict(metric=metric,first=first,second=second,stage_target_words=t,
                    conservative_difference_pp=old,permissive_difference_pp=new,change_pp=new-old,
                    strict_sign_reversal=bool(direction(old)*direction(new)<0),tie_after=bool(direction(new)==0)))
    deterministic=pd.DataFrame(deterministic_rows)
    permissive_contrasts=pd.DataFrame(deterministic_contrasts)
    export(deterministic,'D_permissive_mittelwerte_und_raenge')
    export(permissive_contrasts,'D_permissive_gepaarte_differenzen')
    display(deterministic[deterministic.metric.eq('raw')][['method','stage_target_words','n_discordant','n_valid_fixed',
        'conservative_percent','permissive_percent','change_pp','conservative_rank','permissive_rank']].round(3))
    display(permissive_contrasts[permissive_contrasts.metric.eq('raw')].round(3))
    changed_ranks=deterministic[(deterministic.metric=='raw') &
        deterministic.conservative_rank.ne(deterministic.permissive_rank)]
    display(Markdown(f'**Hauptmetrik:** '
        f'{int(permissive_contrasts[permissive_contrasts.metric.eq("raw")].strict_sign_reversal.sum())}/16 '
        f'vordefinierte Kontraste wechseln das Vorzeichen, aber {len(changed_ranks)}/24 '
        'Methoden-Stufen-Rangplätze ändern sich. Die vier Kontraste decken nicht alle 15 '
        'Methodenpaare ab; stabile Kontrastvorzeichen bedeuten daher keine vollständig stabile Rangfolge.'))

    fig,axes=plt.subplots(2,2,figsize=(13,8.8),sharex=True,sharey=True,layout='constrained')
    for ax,stage in zip(axes.flat,STAGES):
        sub=deterministic[deterministic.metric.eq('raw') & deterministic.stage_target_words.eq(stage)].set_index('method').loc[METHODS]
        for i,(method,row) in enumerate(sub.iterrows()):
            ax.plot([row.conservative_percent,row.permissive_percent],[i,i],color='#777777',lw=2)
            ax.scatter(row.conservative_percent,i,facecolors='white',edgecolors='#0072B2',s=48,zorder=3)
            ax.scatter(row.permissive_percent,i,color='#0072B2',s=40,zorder=3)
            ax.annotate(f'+{de(row.change_pp,1)} PP',(row.permissive_percent,i),xytext=(7,0),textcoords='offset points',va='center',fontsize=9)
        ax.set(yticks=range(6),yticklabels=[NAMES[m] for m in METHODS],xlim=(0,113),
               xticks=range(0,101,20),xlabel='Mittlere QA-Retention (%)',title=f'Zielstufe {stage} Wörter')
        ax.grid(axis='x',alpha=.15)
    axes[0,0].invert_yaxis()
    fig.legend(handles=[plt.Line2D([0],[0],marker='o',mfc='white',color='#0072B2',ls='',label='Konservative Regel'),
                        plt.Line2D([0],[0],marker='o',color='#0072B2',ls='',label='Permissive Regel')],
               loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('Sensitivität gegenüber der Behandlung von Referenzabweichungen · 100 Artikel je Gruppe',fontsize=14)
    savefig(fig,'05_permissive_referenzregel_methodenvergleich')
    shown=deterministic[deterministic.metric.eq('raw')]
    display(Markdown(f'**Beobachtung:** Die permissive Regel erhöht die Retention je Verfahren und Stufe um '
        f'{de(shown.change_pp.min(),1)} bis {de(shown.change_pp.max(),1)} Prozentpunkte. '
        'Die ungleichen Verschiebungen erklären, weshalb sich Rangplätze trotz stabiler Vorzeichen einzelner Kontraste ändern können.'))

    main_table=attacks[['first','second','contrast','stage_target_words','original_difference_pp',
        'min_strict_reverse','min_reverse_percent_valid','n_articles_compact','n_articles_spread']].copy()
    for sid,label in [('sym_0.050','sym5_reversal_percent'),('article_0.050','article5_reversal_percent')]:
        subset=sim_contrasts[(sim_contrasts.scenario==sid)&sim_contrasts.metric.eq('raw')][
            ['first','second','stage_target_words','reversal_frequency_percent']].rename(columns={'reversal_frequency_percent':label})
        main_table=main_table.merge(subset,on=['first','second','stage_target_words'],validate='one_to_one')
    main_table=main_table.merge(permissive_contrasts[permissive_contrasts.metric.eq('raw')][
        ['first','second','stage_target_words','permissive_difference_pp','strict_sign_reversal']],
        on=['first','second','stage_target_words'],validate='one_to_one')
    export(main_table,'haupttext_robustheit_16_vergleiche')
    display(main_table.drop(columns=['first','second']).round(3))
    compact_rows=pd.concat([main_table.nsmallest(2,'min_strict_reverse'),
                           main_table.nlargest(2,'min_strict_reverse')]).drop_duplicates(['first','second','stage_target_words'])
    compact_table=compact_rows[['contrast','stage_target_words','original_difference_pp','min_strict_reverse','min_reverse_percent_valid']]
    export(compact_table,'haupttext_kompakttabelle_4_befunde')
    display(Markdown('**Platzsparende Haupttextauswahl:** die zwei kleinsten und zwei größten '
        'gezielten Änderungsbudgets; Auswahl deskriptiv nach der vollständig gezeigten 16-Zellen-Tabelle.'))
    display(compact_table.round(3))
    family_summary=sim_contrasts[sim_contrasts.metric.eq('raw')].groupby(['family','first','second','stage_target_words']).agg(
        min_scenario_reversal_percent=('reversal_frequency_percent','min'),
        max_scenario_reversal_percent=('reversal_frequency_percent','max'),n_scenarios=('scenario','nunique')).reset_index()
    export(family_summary,'zusammenfassung_szenariofamilien_getrennt')
    display(Markdown('Die Minima und Maxima je Szenariofamilie sind deskriptive Spannweiten über verschiedene Annahmen. '
        'Sie werden nicht zu einer gemeinsamen Fehler- oder Überlegenheitswahrscheinlichkeit gemittelt.'))

    fragile=attacks.sort_values(['min_strict_reverse','stage_target_words']).head(2)
    robust=attacks.sort_values('min_strict_reverse',ascending=False).iloc[0]
    fragile_text='; '.join(f'{r.contrast} bei {r.stage_target_words} Wörtern: {r.min_strict_reverse} Änderungen '
        f'({de(r.min_reverse_percent_valid)} %)' for r in fragile.itertuples())
    n_permissive_reversed=int(permissive_contrasts[permissive_contrasts.metric.eq('raw')].strict_sign_reversal.sum())
    sym5=sim_contrasts[(sim_contrasts.metric=='raw')&(sim_contrasts.scenario=='sym_0.050')]
    dep5=sim_contrasts[(sim_contrasts.metric=='raw')&(sim_contrasts.scenario=='article_0.050')]
    leader5=leaders[(leaders.metric=='raw')&(leaders.scenario=='sym_0.050')]
    worst_leader=leader5.sort_values('rank1_kept_percent').iloc[0]
    adverse300=sim_contrasts[(sim_contrasts.scenario=='method_automate_cot_0.03') &
        (sim_contrasts.metric=='raw') & (sim_contrasts['first']=='automate_cot') &
        (sim_contrasts.second=='auto_cot') & sim_contrasts.stage_target_words.eq(300)].iloc[0]
    large=attacks[attacks.original_difference_pp.abs().ge(5)]
    all_scenarios_for_large=sim_contrasts[sim_contrasts.metric.eq('raw')].merge(
        large[['first','second','stage_target_words']],on=['first','second','stage_target_words'],validate='many_to_one')
    stable_count=int((all_scenarios_for_large.groupby(['first','second','stage_target_words']).reversal_frequency_percent.max()<=1).sum())
    chapter=(
        'Die Sensitivitätsanalyse reproduzierte alle ursprünglichen Retentionsmittelwerte, '
        'gepaarten Unterschiede und zugehörigen Artikelbootstrap-Intervalle sowie die längenadjustierten Ergebnisse. '
        'Technische Fehlwerte und Originalklassifikationen blieben unverändert.\n\n'
        f'Für eine gezielte strikte Vorzeichenumkehr waren je Vergleich und Stufe mindestens '
        f'{int(attacks.min_strict_reverse.min())} bis {int(attacks.min_strict_reverse.max())} Änderungen erforderlich. '
        f'Besonders empfindlich waren {fragile_text}. '
        f'Der größte Mindestbedarf betraf {robust.contrast} bei {robust.stage_target_words} Wörtern '
        f'({robust.min_strict_reverse} Änderungen). Dies sind adversariale Änderungsbudgets, keine Fehlerraten.\n\n'
        f'Unter unabhängigen symmetrischen 5-%-Änderungen lagen die simulierten Vorzeichenumkehranteile '
        f'zwischen {de(sym5.reversal_frequency_percent.min())} und {de(sym5.reversal_frequency_percent.max())} %. '
        f'Bei artikelkorrelierten 5-%-Änderungen reichte die Spannweite bis '
        f'{de(dep5.reversal_frequency_percent.max())} %. Die ursprüngliche Führung bei '
        f'{int(worst_leader.stage_target_words)} Wörtern blieb im unabhängigen 5-%-Szenario in '
        f'{de(worst_leader.rank1_kept_percent)} % der Simulationen erhalten. '
        'Rangfolgenstabilität hängt damit von Änderungsrichtung und Abhängigkeitsannahme ab; '
        'Ranghäufigkeiten sind keine Wahrscheinlichkeiten tatsächlicher Überlegenheit.\n\n'
        f'Bei zusätzlich 3 pp nachteiligen Änderungen für Automate-CoT kehrte sich dessen '
        f'300-Wörter-Vergleich zu Auto-CoT in {de(adverse300.reversal_frequency_percent)} % der Simulationen um. '
        f'Die adjustierte Auto-CoT-Führung bei 75 Wörtern blieb im unabhängigen 5-%-Szenario nur '
        f'in {de(adjusted75.rank1_kept_percent)} % erhalten.\n\n'
        f'Die maximal permissive Behandlung von reference_discordant kehrte '
        f'{n_permissive_reversed}/16 ursprünglichen Kontrastvorzeichen um. '
        'Die 500-Wörter-Führung wechselte dabei von Lead-k zu Automate-CoT. '
        'Große deskriptive Abstände sind unter den untersuchten Annahmen belastbarer als knappe '
        'Rangpositionen; eine allgemeine Verfahrensüberlegenheit oder Scorer-Genauigkeit folgt daraus nicht. '
        'Empfohlen wird ein Kurzabschnitt in Kapitel 4.2 mit der kompakten Vier-Zeilen-Tabelle. '
        'Die eigenständige Robustheitsabbildung 1 und Szenariotabellen gehören aus Platzgründen '
        'in den Anhang, die unbekannte reale Fehlerrate in die Limitationen.'
    )
    assert len(chapter.split())<=250,len(chapter.split())
    (OUT/'Ergebniszusammenfassung_Masterarbeit_max250_Woerter.md').write_text(chapter+'\n',encoding='utf-8')
    display(Markdown(chapter))
    display(Markdown(f'*Automatisch erzeugt: {len(chapter.split())} Wörter.*'))

    assert hashlib.sha256(qa[KEY+['label']].to_csv(index=False).encode()).hexdigest()==original_label_digest
    assert (~qa.valid).sum()==196
    assert reproduction.bestanden.all() and length_reproduction.passed.all()
    assert mc_checks.analytic_expectation_check.all()
    assert np.all(attacks.min_strict_reverse>=attacks.min_tie_or_cross)
    assert np.all(attacks.n_changes_first+attacks.n_changes_second==attacks.min_strict_reverse)
    # Unabhängiger Abgleich der Minima aus Artikelzählungen und ganzzahligen kgV-Gewichten.
    # Diese Prüfung braucht weder die Einzelurteil-Sortierung noch die Fraction-Präfixliste.
    for row in attacks.itertuples():
        i,j=METHODS.index(row.first),METHODS.index(row.second)
        t=STAGES.index(row.stage_target_words)
        integer_gap=int(ORIG_RAW_SCORES[i,t]-ORIG_RAW_SCORES[j,t])
        available_first=K[:,i,t] if integer_gap>0 else V[:,i,t]-K[:,i,t]
        available_second=V[:,j,t]-K[:,j,t] if integer_gap>0 else K[:,j,t]
        possible_weights=np.concatenate([np.repeat(RAW_SCALE[:,i,t],available_first),
                                         np.repeat(RAW_SCALE[:,j,t],available_second)])
        cumulative=np.cumsum(np.sort(possible_weights)[::-1])
        assert np.searchsorted(cumulative,abs(integer_gap),side='left')+1==row.min_tie_or_cross
        assert np.searchsorted(cumulative,abs(integer_gap),side='right')+1==row.min_strict_reverse
    assert rank_frequencies.groupby(['scenario','metric','method','stage_target_words']).n.sum().eq(REPEATS).all()
    for source,digest in sources_before.items():
        assert sha(source)==digest,source
    manifest_out=dict(data_version=VERSION,source_package=str(PACKAGE),software=software,
        n_articles=len(IDS),n_facts=len(facts),n_qa=len(qa),n_valid=int(qa.valid.sum()),n_missing=int((~qa.valid).sum()),
        original_sources_unchanged=True,original_labels_unchanged=True,new_llm_calls=0,new_human_ratings=0,
        original_results_reproduced=True,bootstrap_unit='article',bootstrap_seed=SEED,bootstrap_draws=B,
        simulation_unit='conditional binary flips; independent binomial by observed class, or shared article flip',
        n_scenarios=len(scenarios),repetitions_per_scenario=REPEATS,simulation_seed_base=SIM_SEED,
        simulation_uncertainty='separate from article sampling uncertainty; no combined probabilities',
        original_label_digest=original_label_digest,raw_rank_integer_lcm=LCM,
        minimum_target='strict sign reversal; tie-or-cross also separately calculated',
        minimum_solution_ties='same optimal weight multiset; canonical, compact heuristic, spread heuristic',
        original_metric='equal article mean K/V',length_adjusted_metric='equal article mean (K/20)/(1+D)',
        minimum_changes_range=[int(attacks.min_strict_reverse.min()),int(attacks.min_strict_reverse.max())],
        permissive_raw_sign_reversals=n_permissive_reversed)
    (OUT/'ausfuehrungsprotokoll.json').write_text(json.dumps(manifest_out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    display(Markdown('**Alle Abschlussprüfungen erfolgreich.** Originaldaten, Klassifikationen, Scorer und bestehende '
        'Notebooks sind unverändert. Tabellen, konkrete hypothetische Änderungspläne, Simulationswerte und '
        'Abbildungen wurden ausschließlich in den neuen Ergebnisordner exportiert.'))
    display(pd.DataFrame([{'Datei':str(p.relative_to(ROOT)),'Bytes':p.stat().st_size}
                         for p in sorted(OUT.rglob('*')) if p.is_file()]))

if __name__ == '__main__':
    from offline_guard import install
    install(allow_statistics_workers='--recompute-regression-bootstrap' in sys.argv)
    main(sys.argv[1], '--recompute-regression-bootstrap' in sys.argv, int(sys.argv[2]))
