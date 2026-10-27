"""Eindeutige Artikelschlüssel prüfen, bevor Modelle oder Caches verwendet werden."""


def validate_article_keys(rows, *, require_article_ids=True):
    seen_indices = set()
    seen_ids = set()
    for position, row in enumerate(rows, 1):
        try:
            index = int(str(row['dataset_index']))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'Zeile {position}: dataset_index muss eine nichtnegative Ganzzahl sein.') from error
        if index < 0:
            raise ValueError(f'Zeile {position}: dataset_index muss nichtnegativ sein.')
        key = (str(row.get('split', '')), index)
        if key in seen_indices:
            raise ValueError(f'Doppelter dataset_index {index} im Split {key[0]!r}. Keine Generierung gestartet.')
        seen_indices.add(key)
        if require_article_ids:
            article_id = str(row.get('article_id', '')).strip()
            if not article_id or article_id in seen_ids:
                raise ValueError(f'Leere oder doppelte article_id in Zeile {position}. Keine Generierung gestartet.')
            seen_ids.add(article_id)


def validate_candidate_keys(rows):
    seen = set()
    seen_sources = set()
    source_articles = {}
    for row in rows:
        candidate_id = str(row.get('candidate_id', ''))
        if not candidate_id or candidate_id in seen:
            raise ValueError('Leere oder doppelte candidate_id. Keine Modellauswertung gestartet.')
        seen.add(candidate_id)
        if all(key in row for key in ['split', 'dataset_index', 'target_words']):
            source = (str(row['split']), int(row['dataset_index']))
            stage_key = (*source, int(row['target_words']))
            if stage_key in seen_sources:
                raise ValueError('Doppelter Artikel-/Stufenschlüssel in Demonstrationskandidaten.')
            seen_sources.add(stage_key)
            article_hash = row.get('article_sha256')
            if article_hash:
                if source in source_articles and source_articles[source] != article_hash:
                    raise ValueError('Ein dataset_index verweist auf unterschiedliche Kandidatenartikel.')
                source_articles[source] = article_hash
