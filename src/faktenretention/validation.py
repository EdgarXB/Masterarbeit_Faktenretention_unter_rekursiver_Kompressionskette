"""Explicit historical comparison policy; absolute tolerance remains 1e-12."""
from pathlib import Path
import gzip
import io

# Exact provenance exclusions from the earlier runner, without substring rules.
# n_warning_fits and warning_text are now compared as scientific diagnostics.
PROVENANCE_COLUMNS = {
    'source_state_file','source_state_sha256','input_sha256','output_sha256',
    'prompt_sha256','source','file','sha256','summary_sha256','highlight_sha256',
    'summary_sha256_43','Datei','SHA256','fingerprint','qa_fingerprint','cache_used',
}
RUNTIME_COLUMNS = {'fit_seconds','seconds_this_run'}
INTERMEDIATE_ARRAYS = {'bootstrap_kendall_10000_seed32.npz','bootstrap_summac_referenzwechsel.npz',
    'bootstrap_paare_10000_seed32.npz','bootstrap_gruppen_10000_seed32.npz'}

def compare_table(new_path, reference_bytes, label):
    import numpy as np
    import pandas as pd
    new_path=Path(new_path)
    old_bytes=gzip.decompress(reference_bytes) if new_path.name.endswith('.gz') else reference_bytes
    new_bytes=gzip.decompress(new_path.read_bytes()) if new_path.name.endswith('.gz') else new_path.read_bytes()
    frames=[]
    for content in (old_bytes,new_bytes):
        try:frames.append(pd.read_csv(io.BytesIO(content),keep_default_na=False,low_memory=False))
        except pd.errors.EmptyDataError:frames.append(None)
    old,new=frames
    if old is None or new is None:
        return dict(file=label,passed=old is None and new is None and old_bytes.strip()==new_bytes.strip(),kind='empty_table')
    if new_path.name=='export_manifest.csv':
        missing=set(old['file'])-set(new['file']);added=set(new['file'])-set(old['file'])
        return dict(file=label,kind='artifact_inventory',passed=not added and (not missing or missing==INTERMEDIATE_ARRAYS),
            removed_intermediate_arrays=sorted(missing),added_files=sorted(added))
    if list(old.columns)!=list(new.columns) or len(old)!=len(new):
        return dict(file=label,passed=False,reason='shape_or_columns',old_shape=list(old.shape),new_shape=list(new.shape))
    excluded=[c for c in old if c in PROVENANCE_COLUMNS or c in RUNTIME_COLUMNS]
    identifiers={'article_id','canonical_fact_id','dataset_index','method','stage_target_words',
                 'first','second','contrast','policy','entity_type','model','metric','term','draw'}
    keys=[c for c in old if c in identifiers and c not in excluded]
    keys += [c for c in old if c not in keys and c not in excluded and
             pd.api.types.is_string_dtype(old[c]) and pd.api.types.is_string_dtype(new[c])]
    if keys:
        old=old.sort_values(keys,kind='stable').reset_index(drop=True)
        new=new.sort_values(keys,kind='stable').reset_index(drop=True)
    checked=[];problems=[];max_error=0.0
    for column in old:
        if column in excluded:continue
        left=pd.to_numeric(old[column],errors='coerce').astype(float)
        right=pd.to_numeric(new[column],errors='coerce').astype(float)
        if left.notna().any() or right.notna().any():
            ok=np.allclose(left,right,rtol=0,atol=1e-12,equal_nan=True)
            neither_numeric=left.isna() & right.isna()
            ok=ok and old.loc[neither_numeric,column].astype(str).equals(new.loc[neither_numeric,column].astype(str))
            error=(left-right).abs().max()
            if pd.notna(error):max_error=max(max_error,float(error))
        else:ok=old[column].astype(str).equals(new[column].astype(str))
        checked.append(column)
        if not ok:problems.append(column)
    return dict(file=label,passed=not problems,row_count=len(new),key_columns=keys,checked_columns=checked,
        excluded_provenance_or_runtime_columns=excluded,differences=problems,max_absolute_error=max_error)

def compare_png(new_path,reference_bytes,label):
    import numpy as np
    from PIL import Image
    with Image.open(new_path) as img:new=np.asarray(img.convert('RGBA'))
    with Image.open(io.BytesIO(reference_bytes)) as img:old=np.asarray(img.convert('RGBA'))
    same=old.shape==new.shape;exact=same and np.array_equal(old,new)
    return dict(file=label,passed=exact,kind='figure',status='identical_pixels' if exact else 'visual_review_required',
        reference_shape=list(old.shape),new_shape=list(new.shape),
        changed_pixel_fraction=float(np.any(old!=new,axis=2).mean()) if same else None,
        mean_absolute_channel_error=float(np.abs(old.astype(float)-new).mean()) if same else None)

def require_comparisons(comparisons):
    if not comparisons:raise ValueError('Leere Vergleichsliste: kein erfolgreicher Reproduktionslauf möglich.')
    return all(item['passed'] for item in comparisons)
