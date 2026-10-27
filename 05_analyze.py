"""Gemeinsame Offline-Eingaben bereitstellen; Analysen laufen direkt in Notebooks."""
import argparse
from faktenretention.data import prepare

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='Die sechs erforderlichen Rohdateien nach data/prepared entpacken')
    args=parser.parse_args()
    if args.prepare:print(prepare())
    else:parser.print_help()
