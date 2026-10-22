# Zweck: Originale QA-Klassifikation; Modellphasen ausdrücklich fortsetzen.
# Eingaben: eingefrorene Antworten oder Antworten/Konfiguration eines neuen Laufs.
# Ausgaben: QA-Labels, optional QA-Antworten oder modellbasierte Metriken.
# LLM-Aufrufe: nur --generate-qa --execute. Beispiel: python 04_evaluate.py
"""Originaler QA-Scorer; neue Modellantworten nur mit --generate-qa --execute."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'data/code'))
import argparse
import json
from research_pipeline import generate_qa,compute_metrics,read_csv,write_csv,PACKAGE
from request_budget import activate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path)
    p.add_argument('--generate-qa',action='store_true')
    p.add_argument('--model-metrics',action='store_true')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.generate_qa or a.model_metrics:
        if not(a.run and a.execute):p.error('Modellphasen benötigen --run und --execute')
        config=json.loads((a.run/'execution.json').read_text())
        activate(a.run,config)
        if a.generate_qa:generate_qa(a.run,config)
        if a.model_metrics:compute_metrics(a.run,config,semantic=True)
    else:
        from offline_guard import install
        install()
        from analyze_run import label_answer
        source=a.run/'data/answers.csv' if a.run else PACKAGE/'data/qa_antworten.csv.gz'
        output=a.run/'evaluation' if a.run else PACKAGE/'results/qa_neuberechnet'
        rows=read_csv(source)
        for row in rows:row['evaluation_label']=label_answer(row)
        write_csv(output/'qa_labels.csv',rows)
        print(str(len(rows))+' Antworten mit Original-Scorer klassifiziert.')

if __name__=='__main__':main()
