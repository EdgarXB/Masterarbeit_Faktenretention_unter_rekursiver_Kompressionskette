#!/usr/bin/env python3
"""Select five stage-specific Auto-CoT demonstrations by input clustering."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def word_chunks(text: str, chunk_words: int) -> list[str]:
    words = (text or "").split()
    return [" ".join(words[index : index + chunk_words]) for index in range(0, len(words), chunk_words)]


def embed_documents(
    texts: list[str], model: SentenceTransformer, chunk_words: int, batch_size: int
) -> np.ndarray:
    chunks: list[str] = []
    owners: list[int] = []
    for owner, text in enumerate(texts):
        for chunk in word_chunks(text, chunk_words):
            chunks.append(chunk)
            owners.append(owner)
    chunk_embeddings = np.asarray(
        model.encode(
            chunks,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    document_embeddings = np.zeros((len(texts), chunk_embeddings.shape[1]), dtype=np.float32)
    counts = Counter(owners)
    for owner, embedding in zip(owners, chunk_embeddings):
        document_embeddings[owner] += embedding
    for owner in range(len(texts)):
        document_embeddings[owner] /= counts[owner]
        norm = float(np.linalg.norm(document_embeddings[owner]))
        if norm:
            document_embeddings[owner] /= norm
    return document_embeddings


def numeric_tokens(text: str) -> set[str]:
    normalized: set[str] = set()
    for token in re.findall(r"(?:[$£€]\s*)?\d[\d,]*(?:\.\d+)?%?", text or ""):
        value = re.sub(r"[$£€,%\s]", "", token.lower())
        if value:
            normalized.add(value)
    return normalized


def quality_eligible(row: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not str(row.get("input_text") or "").strip():
        reasons.append("empty_input")
    if not str(row.get("thinking") or "").strip():
        reasons.append("empty_rationale")
    if int(row.get("substantive_thinking_word_count") or 0) < 25:
        reasons.append("rationale_has_fewer_than_25_substantive_words")
    if int(row.get("thinking_word_count") or len(str(row.get("thinking") or "").split())) > 512:
        reasons.append("rationale_exceeds_512_word_upper_bound")
    if not str(row.get("response") or "").strip():
        reasons.append("empty_response")
    if not bool(row.get("word_window_success")):
        reasons.append("outside_word_window")
    if not bool(row.get("ends_with_complete_sentence")):
        reasons.append("incomplete_final_sentence")
    expected_input = {500: 800, 300: 500, 150: 300, 75: 150}[int(row["target_words"])]
    input_low, input_high = (
        (795, 805) if int(row["target_words"]) == 500
        else (round(expected_input * 0.95), round(expected_input * 1.05))
    )
    input_words = int(row.get("input_word_count") or 0)
    if not input_low <= input_words <= input_high:
        reasons.append(f"input_outside_expected_stage_window:{input_words}_not_{input_low}-{input_high}")
    return not reasons, reasons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--chunk-words", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=32)
    parser.add_argument("--n-init", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_jsonl(args.candidates)
    from faktenretention.input_checks import validate_candidate_keys
    validate_candidate_keys(candidates)
    model = SentenceTransformer(args.model_path, local_files_only=True, device=args.device)
    selected_all: dict[str, list[dict[str, Any]]] = {}
    report: dict[str, Any] = {
        "created_at_utc": now_utc(),
        "method": "stage-specific Auto-CoT input clustering",
        "embedding_model_path": args.model_path,
        "embedding_model_revision": MODEL_REVISION,
        "embedding_device": args.device,
        "document_embedding": f"mean of normalized embeddings over non-overlapping {args.chunk_words}-word chunks, then L2 normalize",
        "kmeans": {
            "clusters_per_stage": args.clusters,
            "random_state": args.random_state,
            "n_init": args.n_init,
        },
        "stages": {},
    }
    for target in (500, 300, 150, 75):
        stage_rows = [row for row in candidates if int(row["target_words"]) == target]
        if len(stage_rows) < args.clusters:
            raise RuntimeError(f"Only {len(stage_rows)} rows for target={target}")
        embeddings = embed_documents(
            [str(row["input_text"]) for row in stage_rows],
            model,
            args.chunk_words,
            args.batch_size,
        )
        kmeans = KMeans(
            n_clusters=args.clusters,
            random_state=args.random_state,
            n_init=args.n_init,
        )
        labels = kmeans.fit_predict(embeddings)
        distances = kmeans.transform(embeddings)
        stage_selected: list[dict[str, Any]] = []
        cluster_reports: list[dict[str, Any]] = []
        for cluster in range(args.clusters):
            member_indices = np.where(labels == cluster)[0].tolist()
            member_indices.sort(key=lambda index: (float(distances[index, cluster]), stage_rows[index]["candidate_id"]))
            audit: list[dict[str, Any]] = []
            chosen_index: int | None = None
            for index in member_indices:
                eligible, reasons = quality_eligible(stage_rows[index])
                audit.append(
                    {
                        "candidate_id": stage_rows[index]["candidate_id"],
                        "distance_to_centroid": float(distances[index, cluster]),
                        "eligible": eligible,
                        "rejection_reasons": reasons,
                    }
                )
                if eligible:
                    chosen_index = index
                    break
            if chosen_index is None:
                raise RuntimeError(f"No eligible Auto-CoT candidate for target={target}, cluster={cluster}")
            chosen = dict(stage_rows[chosen_index])
            chosen["cluster"] = cluster
            chosen["distance_to_centroid"] = float(distances[chosen_index, cluster])
            chosen["cluster_size"] = len(member_indices)
            chosen["auto_cot_selection"] = "nearest eligible input to KMeans centroid"
            stage_selected.append(chosen)
            cluster_reports.append(
                {
                    "cluster": cluster,
                    "size": len(member_indices),
                    "chosen_candidate_id": chosen["candidate_id"],
                    "ordered_centroid_audit": audit,
                }
            )
        stage_selected.sort(key=lambda row: int(row["cluster"]))
        selected_all[str(target)] = stage_selected
        report["stages"][str(target)] = {
            "candidate_count": len(stage_rows),
            "silhouette_score_cosine": (
                float(silhouette_score(embeddings, labels, metric="cosine"))
                if len(set(labels)) > 1
                else math.nan
            ),
            "clusters": cluster_reports,
        }

    (args.output_dir / "auto_cot_demonstrations.json").write_text(
        json.dumps(selected_all, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "auto_cot_selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
