# Statische Originalauswertung; Eingaben/Ausgaben relativ zur übergebenen Arbeitskopie.
# Keine LLM-Aufrufe. Standard: vorhandene Regressionscaches verwenden.
from pathlib import Path
import os,sys

def main(work, recompute_bootstrap=False, workers=1):
    WORK=__import__('pathlib').Path(work).resolve()
    __import__('os').chdir(WORK/('supplement' if '04_LLM_Scorervergleich'[0].isdigit() else 'project/02_Auswertung'))
    from pathlib import Path
    from collections import Counter, defaultdict
    from itertools import combinations
    import csv, hashlib, json, math, random, sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from IPython.display import display, Markdown

    sys.dont_write_bytecode = True
    BUNDLE = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p/'daten/paket_manifest.json').is_file()), None)
    assert BUNDLE is not None, 'Notebook aus dem entpackten Analysepaket starten.'
    DATA = BUNDLE/'daten/llm_vergleich'
    OUT = BUNDLE/'ergebnisse/04_llm_scorervergleich'
    OUT.mkdir(parents=True, exist_ok=True)
    LABELS = ('retained', 'lost', 'unsupported')
    HUMAN_LABELS = (*LABELS, 'special')  # historischer Variablenname, hier ausschließlich LLM-Urteile
    SCORERS = ('calibrated_scorer_label', 'original_scorer_label')
    SEED, BOOTSTRAPS = 32, 10000
    COLORS = {'retained':'#0072B2','lost':'#E69F00','unsupported':'#CC79A7','special':'#999999'}
    plt.rcParams.update({'figure.dpi':110,'savefig.dpi':180,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    pd.set_option('display.max_colwidth', 95)
    def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def save_csv(rows, name): pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
    def save_figure(fig, name):
        fig.savefig(OUT/name,bbox_inches='tight')
        plt.show()
        plt.close(fig)

    def strict_json(path):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result: raise ValueError('Doppelter JSON-Schlüssel: '+key)
                result[key] = value
            return result
        return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=unique)

    manifest = strict_json(DATA/'human_validation_manifest.json')
    reference = strict_json(DATA/'referenz_auswertung.json')
    cases = strict_json(DATA/'evaluation_cases.json')
    expected_ids = {c['case_id'] for c in cases}
    assert len(cases) == len(expected_ids) == 200
    assert manifest['dataset_id'] == reference['dataset_id']
    assert sha(DATA/'evaluation_cases.json') == manifest['file_sha256']['intern/evaluation_cases.json']
    files = sorted((DATA/'eingaben').glob('*.json'))
    assert len(files) == 3
    grouped, source_rows = {}, []
    for path in files:
        obj = strict_json(path)
        pid = obj['participant_id']
        cfg = strict_json(DATA/(pid+'.json'))
        assert obj['dataset_id'] == manifest['dataset_id']
        assert pid not in grouped and obj['order'] == cfg['case_ids']
        responses = obj['responses']
        assert len(responses) == 200 and set(responses) == expected_ids
        expected_source = next(s for s in reference['sources'] if s['rater_id'] == pid)
        assert sha(path) == expected_source['sha256']
        for cid, row in responses.items():
            assert row['case_id'] == cid and row['participant_id'] == pid
            assert row['case_order'] == obj['order'].index(cid)+1
            assert row['human_label'] in HUMAN_LABELS
            if 'label' in row:
                assert ('special' if row['label']=='unclear' else row['label']) == row['human_label']
        grouped[pid] = responses
        source_rows.append({'Rater':pid,'Modellangabe':expected_source['model_name_as_supplied'],
            'Urteile':len(responses),'Kommentare':sum(bool(r.get('comment')) for r in responses.values()),
            'SHA-256':sha(path)})
    pids = sorted(grouped)
    assert sum(len(g) for g in grouped.values()) == 600
    source_hashes = {str(p.relative_to(BUNDLE)):sha(p) for p in [*files, DATA/'evaluation_cases.json']}
    display(pd.DataFrame(source_rows).drop(columns='SHA-256'))
    display(pd.DataFrame([{k:c[k] for k in ['case_id','article_id','method','stage','question','reference_answer','generated_answer']} for c in cases[:3]]))

    def consensus(votes):
        counts = Counter(votes)
        winners = [k for k,v in counts.items() if v == max(counts.values())]
        unique = len(winners) == 1
        label = winners[0] if unique else None
        majority = unique and counts[label] > len(votes)/2
        return dict(**{f'votes_{k}':counts[k] for k in HUMAN_LABELS},
                    absolute_majority=majority, unique_plurality=unique,
                    plurality_without_majority=unique and not majority,
                    tie=not unique, consensus_label=label,
                    three_class_consensus=majority and label in LABELS)

    def alpha_nominal(units):
        units = [list(u) for u in units if len(u) >= 2]
        counts = Counter(x for unit in units for x in unit)
        n = sum(counts.values())
        if n < 2:
            return None
        observed = sum((len(u)**2-sum(v*v for v in Counter(u).values()))/(len(u)-1) for u in units)/n
        expected = (n*n-sum(v*v for v in counts.values()))/(n*(n-1))
        return 1-observed/expected if expected else None

    def reliability(grouped):
        pids = sorted(grouped)
        ids = sorted(grouped[pids[0]])
        units = [[grouped[p][i]['human_label'] for p in pids] for i in ids]
        pairs = []
        for a,b in combinations(pids,2):
            n = sum(grouped[a][i]['human_label']==grouped[b][i]['human_label'] for i in ids)
            pairs.append(dict(participant_a=a,participant_b=b,agreements=n,n=len(ids),agreement=n/len(ids)))
        dist = Counter(v for u in units for v in u)
        return dict(n_cases=len(ids),n_ratings=sum(dist.values()),
            full_agreement_cases=sum(len(set(u))==1 for u in units),
            full_agreement=sum(len(set(u))==1 for u in units)/len(units),
            pairwise_agreement=sum(r['agreements'] for r in pairs)/(3*len(ids)),
            pairs=pairs,krippendorff_alpha_nominal=alpha_nominal(units),
            label_counts={k:dist[k] for k in HUMAN_LABELS},
            label_counts_by_participant={p:{k:sum(grouped[p][i]['human_label']==k for i in ids) for k in HUMAN_LABELS} for p in pids},
            special_rate=dist['special']/(3*len(ids)),
            cases_with_any_special=sum('special' in u for u in units))

    def confusion(truth, pred):
        cm = [[0]*3 for _ in range(3)]
        for a,b in zip(truth,pred):
            cm[LABELS.index(a)][LABELS.index(b)] += 1
        return cm

    def metrics_cm(cm):
        n = sum(map(sum,cm))
        correct = sum(cm[i][i] for i in range(3))
        per = []
        for i,label in enumerate(LABELS):
            support, predicted, tp = sum(cm[i]),sum(r[i] for r in cm),cm[i][i]
            p = tp/predicted if predicted else 0.0
            r = tp/support if support else None
            f = 2*tp/(support+predicted) if support+predicted else 0.0
            per.append(dict(label=label,support=support,predicted=predicted,precision=p,recall=r,f1=f))
        expected = sum(sum(cm[i])*sum(r[i] for r in cm) for i in range(3))/n**2 if n else None
        accuracy = correct/n if n else None
        return dict(n=n,correct=correct,errors=n-correct,error_rate=1-accuracy if n else None,
            accuracy=accuracy,macro_f1=sum(r['f1'] for r in per)/3 if n else None,
            cohen_kappa=(accuracy-expected)/(1-expected) if n and expected<1 else None,
            class_order=LABELS,confusion_matrix=cm,per_class=per,
            absent_truth_classes=[r['label'] for r in per if not r['support']])

    def metrics(truth,pred):
        return metrics_cm(confusion(truth,pred))

    def flat_metrics(m):
        out={k:m[k] for k in ('accuracy','macro_f1','cohen_kappa','error_rate')}
        for r in m['per_class']:
            for k in ('precision','recall','f1'):
                out[r['label']+'.'+k]=r[k]
        return out

    def percentile(values,q):
        values=sorted(values)
        if not values:
            return None
        k=(len(values)-1)*q
        lo=math.floor(k)
        return values[lo]+(values[math.ceil(k)]-values[lo])*(k-lo)

    def interval(values,attempts):
        values=[v for v in values if v is not None]
        return dict(lower=percentile(values,.025),upper=percentile(values,.975),
                    valid_replicates=len(values),undefined_replicates=attempts-len(values))

    joined = []
    for c in cases:
        votes = [grouped[p][c['case_id']]['human_label'] for p in pids]
        joined.append({**c, **consensus(votes), **{p+'_llm_label':v for p,v in zip(pids,votes)}})
    primary = [r for r in joined if r['three_class_consensus']]
    excluded = [r for r in joined if not r['three_class_consensus']]
    agreement = reliability(grouped)
    estimates = {s:metrics([r['consensus_label'] for r in primary],[r[s] for r in primary]) for s in SCORERS}
    consensus_counts = Counter(r['consensus_label'] or 'tie' for r in joined)
    assert len(primary) == reference['consensus']['main_class_majority_cases']
    assert dict(consensus_counts) == reference['consensus']['all_four_class_counts']
    for s,m in estimates.items():
        for key in ['n','correct','accuracy','macro_f1','cohen_kappa']:
            assert abs(m[key]-reference['scorers_against_llm_majority'][s][key]) < 1e-12
    assert abs(agreement['krippendorff_alpha_nominal']-reference['agreement']['krippendorff_alpha_nominal']) < 1e-12
    display(pd.DataFrame([{'Konsens':k,'Fälle':consensus_counts.get(k,0)} for k in [*HUMAN_LABELS,'tie']]))
    save_csv(joined,'konsens_alle_200.csv')
    save_csv(excluded,'ausgeschlossen_sonderfall_oder_gleichstand.csv')

    distribution = pd.DataFrame([{k:sum(r['human_label']==k for r in grouped[p].values()) for k in HUMAN_LABELS} for p in pids],index=pids)
    distribution.index = [next(r['Modellangabe'] for r in source_rows if r['Rater']==p).replace('_medium','').replace('_',' ') for p in pids]
    display(distribution.rename(columns={'special':'Sonderfall'}))
    fig, ax = plt.subplots(figsize=(10,4))
    left = np.zeros(3)
    for label in HUMAN_LABELS:
        values=distribution[label].to_numpy()
        ax.barh(distribution.index,values,left=left,color=COLORS[label],label=label)
        for i,v in enumerate(values):
            if v >= 7: ax.text(left[i]+v/2,i,str(v),ha='center',va='center',fontsize=10,color='#111111')
        left+=values
    ax.set_xlim(0,200); ax.set_xlabel('Anzahl LLM-Urteile je Modell (n = 200)')
    ax.set_title('Verteilung der vier Antwortklassen')
    ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(0.5,-0.22),frameon=False)
    fig.tight_layout(); save_figure(fig,'01_llm_klassenverteilung.png')
    display(pd.DataFrame(agreement['pairs']).rename(columns={'participant_a':'Rater A','participant_b':'Rater B','agreements':'Übereinstimmungen','agreement':'Anteil'}))
    display(Markdown(f"**Alle drei gleich:** {agreement['full_agreement_cases']}/200 ({agreement['full_agreement']:.1%}). **Paarweise gesamt:** {agreement['pairwise_agreement']:.1%}. **Nominales Alpha:** {agreement['krippendorff_alpha_nominal']:.3f}."))

    def cluster_intervals(rows,repetitions,seed=32):
        if not rows:
            return dict(status='NO_MAIN_CLASS_LLM_MAJORITY')
        articles=sorted({r['article_id'] for r in rows})
        blocks={a:{s:confusion([r['consensus_label'] for r in rows if r['article_id']==a],
                                  [r[s] for r in rows if r['article_id']==a]) for s in SCORERS} for a in articles}
        samples={s:defaultdict(list) for s in SCORERS}; differences=[]
        rng=random.Random(seed)
        for _ in range(repetitions):
            weights=Counter(rng.choices(articles,k=len(articles)));point={}
            for scorer in SCORERS:
                cm=[[sum(w*blocks[a][scorer][i][j] for a,w in weights.items()) for j in range(3)] for i in range(3)]
                point[scorer]=flat_metrics(metrics_cm(cm))
                for k,v in point[scorer].items():samples[scorer][k].append(v)
            differences.append(point[SCORERS[0]]['accuracy']-point[SCORERS[1]]['accuracy'])
        return dict(seed=seed,repetitions=repetitions,article_clusters=len(articles),
            method='95% percentile article-cluster bootstrap; paired predictions travel together; conditional on selected cases, observed LLM majority and these three model outputs.',
            scorers={s:{k:interval(v,repetitions) for k,v in vals.items()} for s,vals in samples.items()},
            calibrated_minus_original_accuracy=interval(differences,repetitions))

    bootstrap = cluster_intervals(primary, BOOTSTRAPS, SEED)
    rows=[]
    for scorer in SCORERS:
        m=estimates[scorer]; ci=bootstrap['scorers'][scorer]['accuracy']
        expected=reference['article_cluster_bootstrap']['scorers'][scorer]['accuracy']
        assert abs(ci['lower']-expected['lower']) < 1e-12 and abs(ci['upper']-expected['upper']) < 1e-12
        rows.append({'Scorer':'Kalibriert' if scorer.startswith('calibrated') else 'Ursprüngliche Regel',
            'Richtig':m['correct'],'Auswertbar':m['n'],'Accuracy':m['accuracy'],
            '95% unten':ci['lower'],'95% oben':ci['upper'],'Macro-F1':m['macro_f1'],'Kappa':m['cohen_kappa']})
    score_table=pd.DataFrame(rows)
    display(score_table.round(4))
    fig,ax=plt.subplots(figsize=(9,3.4))
    for i,row in score_table.iterrows():
        ax.errorbar(row['Accuracy']*100,i,xerr=[[100*(row['Accuracy']-row['95% unten'])],[100*(row['95% oben']-row['Accuracy'])]],fmt='o',color='#0072B2',capsize=5)
        ax.text(row['95% oben']*100+2,i,f"{row['Richtig']}/{row['Auswertbar']} ({row['Accuracy']:.1%})",va='center')
    ax.set_yticks(range(2),score_table.Scorer);ax.set_xlim(0,105);ax.set_ylim(-.6,1.6)
    ax.set_xlabel('Übereinstimmung mit der LLM-Mehrheit (%)')
    ax.set_title('Accuracy auf derselben Auswahl von 138 Konsensfällen')
    fig.text(.01,-.03,'95%-Perzentilintervalle: 10.000 Artikel-Cluster-Bootstraps, Seed 32; keine Populationseffekt-Schätzung.',fontsize=9)
    fig.tight_layout();save_figure(fig,'02_accuracy_artikelbootstrap.png')
    save_csv(rows,'scorervergleich_mit_intervallen.csv')

    cal=estimates['calibrated_scorer_label']
    display(pd.DataFrame(cal['per_class']).round(3))
    cm=np.array(cal['confusion_matrix'])
    fig,ax=plt.subplots(figsize=(6.8,5))
    im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=cm.max())
    for i in range(3):
        for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>.55*cm.max() else '#182c36',fontsize=13)
    ax.set_xticks(range(3),LABELS);ax.set_yticks(range(3),LABELS)
    ax.set_xlabel('Kalibrierter Scorer');ax.set_ylabel('LLM-Mehrheit')
    ax.set_title('Konfusionsmatrix: 138 Hauptklassen-Konsensfälle')
    fig.colorbar(im,ax=ax,label='Anzahl Fälle');fig.tight_layout();save_figure(fig,'03_konfusionsmatrix.png')
    errors=[r for r in primary if r['calibrated_scorer_label']!=r['consensus_label']]
    patterns=Counter((r['consensus_label'],r['calibrated_scorer_label']) for r in errors)
    assert patterns=={('retained','unsupported'):23,('unsupported','retained'):2}
    display(pd.DataFrame([{'LLM-Mehrheit':a,'Scorer':b,'Fälle':n} for (a,b),n in patterns.items()]))
    display(pd.DataFrame([{k:r[k] for k in ['case_id','question','reference_answer','generated_answer','consensus_label','calibrated_scorer_label']} for r in errors[:10]]))
    save_csv(errors,'abweichende_faelle_mit_texten.csv')

    sensitivity_rows=[]
    for title,subset in [('Hauptklassenmehrheit',primary),
                         ('Einstimmige Hauptklassen',[r for r in primary if any(r['votes_'+k]==3 for k in LABELS)]),
                         ('Keine Sonderfallstimme',[r for r in primary if r['votes_special']==0])]:
        m=metrics([r['consensus_label'] for r in subset],[r['calibrated_scorer_label'] for r in subset])
        sensitivity_rows.append({'Auswahl':title,'Fälle':m['n'],'Richtig':m['correct'],'Accuracy':m['accuracy'],'Macro-F1':m['macro_f1']})
    display(pd.DataFrame(sensitivity_rows).round(4))
    save_csv(sensitivity_rows,'konsens_sensitivitaet.csv')
    assert len(primary)==138 and len(errors)==25 and len(excluded)==62
    assert all(sha(BUNDLE/path)==h for path,h in source_hashes.items())
    result={'status':'LLM_RATER_COMPARISON_NOT_HUMAN_VALIDATION','cases':200,'llm_judgments':600,
            'consensus_counts':dict(consensus_counts),'agreement':agreement,'scorers':estimates,
            'article_bootstrap':bootstrap,'sensitivities':sensitivity_rows,'source_sha256':source_hashes}
    (OUT/'notebook_ergebnisse.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    display(Markdown('**Prüfstatus:** alle 600 Urteile verbunden; keine fehlenden, fremden oder doppelten Fall-IDs; Kennzahlen und Accuracy-Intervalle gegen den vorhandenen Lauf reproduziert; gelesene Eingaben unverändert.'))

    fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout='constrained')
    vmax=max(np.array(estimates[s]['confusion_matrix']).max() for s in SCORERS)
    comparison_matrix_rows=[]
    for ax,scorer in zip(axes,SCORERS):
        matrix=np.asarray(estimates[scorer]['confusion_matrix'])
        im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=vmax)
        for i,truth in enumerate(LABELS):
            for j,pred in enumerate(LABELS):
                n=int(matrix[i,j])
                ax.text(j,i,str(n),ha='center',va='center',fontsize=13,color='white' if n>.55*vmax else '#222222')
                comparison_matrix_rows.append(dict(scorer=scorer,llm_majority=truth,scorer_label=pred,n=n))
        assert matrix.sum()==len(primary)
        ax.set(xticks=range(3),xticklabels=LABELS,yticks=range(3),yticklabels=LABELS,
               xlabel='Scorer-Label',ylabel='LLM-Mehrheit',
               title=('Kalibrierter Scorer' if scorer.startswith('calibrated') else 'Ursprüngliche Regel')+f' · n = {len(primary)}')
    fig.colorbar(im,ax=axes,shrink=.7,label='Anzahl Konsensfälle')
    fig.suptitle('Fehlermuster beider Scorer auf derselben LLM-Konsensauswahl',fontsize=14)
    save_csv(comparison_matrix_rows,'grafik_konfusionsmatrizen_beide_scorer.csv')
    save_figure(fig,'04_konfusionsmatrizen_beide_scorer.png')
    retained_index=LABELS.index('retained')
    display(Markdown('**Beobachtung:** Von den '+str(sum(estimates[SCORERS[0]]['confusion_matrix'][retained_index]))+
        ' Retained-Konsensfällen erkennt der kalibrierte Scorer '+str(estimates[SCORERS[0]]['confusion_matrix'][retained_index][retained_index])+
        ', die ursprüngliche Regel '+str(estimates[SCORERS[1]]['confusion_matrix'][retained_index][retained_index])+
        ' ebenfalls als Retained. Dies beschreibt die Übereinstimmung mit der LLM-Mehrheit.'))

    fig,axes=plt.subplots(1,2,figsize=(12.5,4.8),sharex=True,sharey=True,layout='constrained')
    class_plot_rows=[]
    for ax,metric,title in zip(axes,['recall','precision'],['Recall je Mehrheitsklasse','Precision je Scorer-Klasse']):
        for si,scorer in enumerate(SCORERS):
            color=['#0072B2','#D55E00'][si];marker=['o','s'][si]
            for i,item in enumerate(estimates[scorer]['per_class']):
                point=item[metric];ci=bootstrap['scorers'][scorer][item['label']+'.'+metric]
                y=i+[-.12,.12][si]
                if point is not None:
                    ax.plot([ci['lower']*100,ci['upper']*100],[y,y],color=color,lw=1.5)
                    ax.scatter(100*point,y,color=color,marker=marker,s=40,zorder=3)
                class_plot_rows.append(dict(scorer=scorer,label=item['label'],metric=metric,estimate=point,
                    support=item['support'],predicted=item['predicted'],**ci))
        labels=[f'{item["label"]} (n={item["support"]})' for item in estimates[SCORERS[0]]['per_class']]
        ax.set(yticks=range(3),yticklabels=labels,xlim=(-3,103),xticks=range(0,101,20),
               xlabel='Anteil (%) · punktweises 95-%-Intervall',title=title)
        ax.grid(axis='x',alpha=.15)
    axes[0].invert_yaxis()
    fig.legend(handles=[plt.Line2D([0],[0],marker='o',color='#0072B2',ls='',label='Kalibriert'),
                        plt.Line2D([0],[0],marker='s',color='#D55E00',ls='',label='Ursprüngliche Regel')],
               loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('Klassenweise Übereinstimmung mit der LLM-Mehrheit · 138 Konsensfälle',fontsize=14)
    save_csv(class_plot_rows,'grafik_klassenmetriken_mit_intervallen.csv')
    save_figure(fig,'05_precision_recall_beide_scorer.png')
    display(Markdown('**Einordnung:** Die Klasse Unsupported umfasst nur '+
        str(estimates[SCORERS[0]]['per_class'][LABELS.index('unsupported')]['support'])+
        ' Referenzfälle. Insbesondere deren Recall und die Precision seltener Scorer-Urteile '
        'sind zusammen mit den breiten Intervallen und dem Klassenumfang zu lesen.'))
    # Ergänzende CSV-Exporte aus dem ursprünglichen LLM-Auswertungsskript.
    # Die Metrik- und Konsensfunktionen sind bereits unverändert im Referenznotebook.
    for row in joined:
        row.update({p+'_comment':grouped[p][row['case_id']]['comment'] for p in pids})
    exported=[]
    for row in joined:
        out={k:v for k,v in row.items() if k!='three_class_consensus'}
        out['llm_main_class_majority']=row['three_class_consensus']
        for scorer in SCORERS:
            out[scorer+'_matches_llm_majority']=row[scorer]==row['consensus_label'] if row['three_class_consensus'] else ''
        exported.append(out)
    save_csv(exported,'all_200_cases_llm_consensus.csv')
    save_csv([r for r in exported if not r['llm_main_class_majority']],'excluded_special_or_tie.csv')
    save_csv([r for r in exported if r['llm_main_class_majority'] and not r['calibrated_scorer_label_matches_llm_majority']],'calibrated_disagreements.csv')
    common=[r for r in cases if all(grouped[p][r['case_id']]['human_label'] in LABELS for p in pids)]
    individual=[]
    for p in pids:
        for set_name,population in [('own_main_labels',[r for r in cases if grouped[p][r['case_id']]['human_label'] in LABELS]),('all_three_raters_main_labels',common)]:
            for scorer in SCORERS:
                m=metrics([grouped[p][r['case_id']]['human_label'] for r in population],[r[scorer] for r in population])
                individual.append(dict(rater_id=p,comparison_set=set_name,scorer=scorer,**{k:m[k] for k in ('n','correct','errors','accuracy','macro_f1','cohen_kappa')}))
    save_csv(individual,'individual_scorer_comparisons.csv')
    by_group=[]
    for dimension in ('method','stage'):
        for value in sorted({r[dimension] for r in joined},key=str):
            sub=[r for r in primary if r[dimension]==value]
            for scorer in SCORERS:
                m=metrics([r['consensus_label'] for r in sub],[r[scorer] for r in sub])
                by_group.append(dict(dimension=dimension,value=value,scorer=scorer,**{k:m[k] for k in ('n','correct','errors','accuracy','macro_f1','cohen_kappa')}))
    save_csv(by_group,'by_method_stage.csv')
    save_csv([dict(rater_id=p,case_id=cid,llm_label=r['human_label'],comment=r['comment'],presentation_version=r.get('presentation_version','unrecorded')) for p in pids for cid,r in grouped[p].items()],'all_600_llm_judgments.csv')
    save_csv([dict(rater_id=p,model_name_as_supplied=next(s['Modellangabe'] for s in source_rows if s['Rater']==p),**{k:sum(r['human_label']==k for r in grouped[p].values()) for k in HUMAN_LABELS}) for p in pids],'class_distributions.csv')

if __name__ == '__main__':
    from offline_guard import install
    install(allow_statistics_workers='--recompute-regression-bootstrap' in sys.argv)
    main(sys.argv[1], '--recompute-regression-bootstrap' in sys.argv, int(sys.argv[2]))
