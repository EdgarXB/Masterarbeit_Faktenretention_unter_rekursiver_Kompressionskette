"""Vergleiche eingefrorene Ergebnistabellen mit einem unabhängigen Ausgabelauf."""
import argparse
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


RUNTIME_COLUMNS = {
    '06_laengenregression/C_modelldiagnostik.csv': {'fit_seconds'},
    '06_laengenregression/bootstrap_anpassungsdiagnostik.csv': {'seconds_this_run'},
}


def compare(reference: Path, generated: Path) -> tuple[int, list[str]]:
    """Vergleiche Schema, Zeilenfolge und Werte mit expliziten Laufzeitausnahmen."""
    if not reference.is_dir() or not generated.is_dir():
        raise ValueError('Beide Ergebnisverzeichnisse müssen existieren.')
    paths = sorted(reference.rglob('*.csv'))
    if not paths:
        raise ValueError('Keine Referenztabellen gefunden.')
    failures = []
    for source in paths:
        relative = source.relative_to(reference)
        target = generated / relative
        if not target.is_file():
            failures.append(f'{relative}: Ergebnistabelle fehlt')
            continue
        try:
            left, right = pd.read_csv(source), pd.read_csv(target)
            ignored = RUNTIME_COLUMNS.get(relative.as_posix(), set())
            if not ignored.issubset(left.columns) or not ignored.issubset(right.columns):
                raise AssertionError('Dokumentierte Laufzeitspalte fehlt')
            assert_frame_equal(left.drop(columns=list(ignored)), right.drop(columns=list(ignored)),
                               check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)
        except (AssertionError, ValueError, pd.errors.ParserError) as exc:
            failures.append(f'{relative}: {exc}')
    return len(paths), failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--generated', type=Path, required=True)
    args = parser.parse_args()
    count, failures = compare(args.reference, args.generated)
    for failure in failures:
        print(f'ABWEICHUNG: {failure}')
    print(f'{count - len(failures)}/{count} Ergebnistabellen stimmen überein.')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
