# Zweck: Offline-Bestand prüfen oder neue Artikel nach Originalregeln vorbereiten.
# Eingaben: eingefrorene Daten oder ausdrücklich übergebene Artikel-CSV.
# Ausgaben: results/datenpruefung.json oder LAUF/data/articles.csv.
# LLM-Aufrufe: keine. Beispiel: python 01_prepare.py
"""Offline-Bestandsprüfung oder Vorbereitung einer ausdrücklich angegebenen CSV."""
from pathlib import Path
import argparse
from faktenretention.frozen_data import verify_data

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--articles',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.articles:
        from faktenretention.pipeline import prepare_articles
        if not args.output:parser.error('--articles benötigt --output')
        print(prepare_articles(args.articles,args.output))
    else:verify_data()

if __name__=='__main__':main()
