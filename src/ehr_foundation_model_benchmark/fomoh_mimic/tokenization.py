from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class TokenizationSummary:
    sampled_events: int
    mapped_events: int
    unique_raw_codes: int
    unique_mapped_codes: int
    map_fraction: float
    common_unmapped_codes: list[tuple[str, int]]


def load_token_codes(tokenizer_config: Path) -> set[str]:
    payload = json.loads(tokenizer_config.read_text())
    tokens = payload.get("tokens", [])
    return {token["code"] for token in tokens if token.get("type") == "code" and token.get("code")}


def normalize_code_candidates(code: str) -> tuple[str, ...]:
    candidates = [code]
    if "//" in code:
        candidates.append(code.replace("//", "/", 1))
    if code.endswith("//start") or code.endswith("//end"):
        base = code.rsplit("//", 1)[0]
        candidates.append(base)
        if "//" in base:
            candidates.append(base.replace("//", "/", 1))
    return tuple(dict.fromkeys(candidates))


def summarize_tokenization_coverage(
    meds_dir: Path,
    tokenizer_config: Path,
    *,
    max_events: int = 20000,
) -> TokenizationSummary:
    token_codes = load_token_codes(tokenizer_config)
    files = sorted((meds_dir / "data" / "train").glob("*.parquet"))[:8]
    rows = (
        pl.scan_parquet([str(path) for path in files])
        .select("code")
        .drop_nulls()
        .limit(max_events)
        .collect()["code"]
        .to_list()
    )
    mapped = []
    unmapped = Counter()
    for code in rows:
        mapped_code = next(
            (candidate for candidate in normalize_code_candidates(code) if candidate in token_codes),
            None,
        )
        if mapped_code is None:
            unmapped[code] += 1
        else:
            mapped.append(mapped_code)
    return TokenizationSummary(
        sampled_events=len(rows),
        mapped_events=len(mapped),
        unique_raw_codes=len(set(rows)),
        unique_mapped_codes=len(set(mapped)),
        map_fraction=(len(mapped) / len(rows)) if rows else 0.0,
        common_unmapped_codes=unmapped.most_common(10),
    )

