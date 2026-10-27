# Zweck: Originale Fakten-, Summary-, QA- und Metrikphasen verbinden.
# Eingaben: vorbereitete Artikel, finale Promptbank, ausdrücklich aktivierte Modelle.
# Ausgaben: LAUF/data/*.csv, Generierungsstände, Anfragebudget und Metriken.
# LLM-Aufrufe: nur --test oder --full-experiment. Beispiel: python 03_generate.py
"""Explizite Mini- oder Vollausführung. Ohne Aktivierungsflag nur Offline-Datenprüfung."""
from pathlib import Path
import argparse
from faktenretention.pipeline import (
    METHODS, prepare_articles, start_run, load_resources,
    generate_facts, generate_summaries, generate_qa, compute_metrics,
)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--test',action='store_true')
    mode.add_argument('--full-experiment',action='store_true')
    parser.add_argument('--run',type=Path)
    parser.add_argument('--articles',type=Path)
    parser.add_argument('--method',choices=METHODS,default='plain')
    parser.add_argument('--article-limit',type=int,default=1)
    parser.add_argument('--stages',type=int,nargs='+')
    parser.add_argument('--max-requests',type=int)
    parser.add_argument('--summary-only',action='store_true')
    parser.add_argument('--resources',type=Path)
    parser.add_argument('--without-model-metrics',action='store_true')
    parser.add_argument('--bank',type=Path)
    parser.add_argument('--host',default='127.0.0.1:11435')
    a=parser.parse_args()
    if not(a.test or a.full_experiment):
        from faktenretention.data import verify_sources
        print(str(verify_sources())+" Quellen geprüft.");return
    if not a.run:parser.error('--run ist erforderlich')
    if a.full_experiment and a.max_requests is None:parser.error('Vollausführung benötigt ein ausdrückliches --max-requests Budget')
    if a.articles:prepare_articles(a.articles,a.run)
    config=start_run(a.run,mode='full' if a.full_experiment else 'mini',method=a.method,article_limit=a.article_limit,
        stages=a.stages,max_requests=a.max_requests if a.max_requests is not None else 30,host=a.host,
        summary_only=a.summary_only,resources=load_resources(a.resources),semantic_metrics=a.full_experiment and not a.without_model_metrics,bank=a.bank)
    if not a.summary_only:generate_facts(a.run,config)
    generate_summaries(a.run,config,a.bank)
    if not a.summary_only:generate_qa(a.run,config)
    compute_metrics(a.run,config)

if __name__=='__main__':main()
