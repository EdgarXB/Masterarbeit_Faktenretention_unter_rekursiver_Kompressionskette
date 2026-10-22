# Zweck: Offline-Bestand prüfen oder neue Artikel nach Originalregeln vorbereiten.
# Eingaben: eingefrorene Daten oder ausdrücklich übergebene Artikel-CSV.
# Ausgaben: results/datenpruefung.json oder LAUF/data/articles.csv.
# LLM-Aufrufe: keine. Beispiel: python 01_prepare.py
"""Offline-Bestandsprüfung oder Vorbereitung einer ausdrücklich angegebenen CSV."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'data/code'))
import argparse
from dataset_store import verify_data, extract_working_copy
from research_pipeline import prepare_articles

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--articles',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--working-copy',type=Path)
    args=parser.parse_args()
    if args.articles:
        if not args.output:parser.error('--articles benötigt --output')
        print(prepare_articles(args.articles,args.output))
    else:verify_data()
    if args.working_copy:
        if args.working_copy.exists() and any(args.working_copy.iterdir()):parser.error('Arbeitskopie muss leer sein.')
        extract_working_copy(args.working_copy.resolve())

if __name__=='__main__':main()
