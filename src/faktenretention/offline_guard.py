"""Schutz ausschließlich für Offline-Analysen; kein Modellimport, kein Netzwerk."""
import os
import sys


def install(allow_statistics_workers=False):
    def audit(event, args):
        if event == 'subprocess.Popen' and os.path.basename(str(args[0])) == 'fc-list':
            return
        if event in ['socket.connect','socket.getaddrinfo','os.system']:
            raise RuntimeError('Offline-Analyse: Netzwerk/Systembefehle gesperrt: '+event)
        if event == 'subprocess.Popen' and not allow_statistics_workers:
            raise RuntimeError('Statistik-Worker nur mit ausdrücklichem Bootstrap-Neuberechnungsschalter.')
        if event == 'import' and args[0].split('.')[0] in ['torch','transformers','sentence_transformers','summac','ollama','requests','httpx']:
            raise RuntimeError('Modellbibliothek im Offline-Modus gesperrt: '+args[0])
    sys.addaudithook(audit)
    sys.dont_write_bytecode = True
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.show = lambda *args, **kwargs: None
