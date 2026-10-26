"""Shared ordering; statistical procedures retain their own seeds and estimands."""
METHODS = ['lead_k','plain','zero_shot_cot','few_shot_cot','auto_cot','automate_cot']
STAGES = [500,300,150,75]
LABELS = dict(zip(METHODS,['Lead-k','Plain','Zero-Shot-CoT','Few-Shot-CoT','Auto-CoT','Automate-CoT']))
CONTRASTS = [('zero_shot_cot','plain'),('few_shot_cot','plain'),('auto_cot','few_shot_cot'),('automate_cot','auto_cot')]
