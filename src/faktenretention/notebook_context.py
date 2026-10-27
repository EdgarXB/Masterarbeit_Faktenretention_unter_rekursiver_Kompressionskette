"""Small notebook setup: explicit paths and offline inline plotting, no run state."""
from pathlib import Path
import os
from faktenretention.io import safe_output


def display_figure(figure):
    """Embed a bounded PNG even when the original Agg backend is retained."""
    from io import BytesIO
    from IPython.display import Image, display
    buffer = BytesIO()
    figure.savefig(buffer, format='png', dpi=110, bbox_inches='tight')
    display(Image(data=buffer.getvalue()))


def setup_notebook(data_dir, output_dir):
    data_dir = Path(data_dir).expanduser().resolve()
    output_dir = safe_output(output_dir)
    if data_dir == output_dir or data_dir in output_dir.parents:
        raise ValueError('Daten und neue Ausgaben brauchen getrennte Verzeichnisse.')
    if not (data_dir / 'qa/source_manifest.json').is_file():
        raise FileNotFoundError(
            f'Vorbereitete Eingaben fehlen in {data_dir}. '
            'Einmal im Repository ausführen: python 05_analyze.py --prepare '
            '(für einen eigenen Datenordner: --data-dir /absoluter/pfad).')
    from faktenretention.offline_guard import install
    install()
    for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[name] = '1'
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    import matplotlib.pyplot as plt

    def show(*args, **kwargs):
        for number in plt.get_fignums():
            figure = plt.figure(number)
            display_figure(figure)
            plt.close(figure)

    plt.show = show
