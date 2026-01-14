# run_batch.py
import os
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
from dotenv import load_dotenv

from orchestrator.observability import setup_observability, get_langfuse_callbacks

DEFAULT_INPUT_CSV = "Prompts.csv"
DEFAULT_FIXED_CSV = "Prompts_fixed.csv"
DEFAULT_OUTPUT_CSV = "batch_results.csv"
SLEEP_BETWEEN_CALLS_SEC = 0.2


def _read_bytes(path: str, n: int = 8192) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


def _pick_encodings(sample: bytes) -> list[str]:
    # null bytes -> often UTF-16
    if b"\x00" in sample:
        return ["utf-16", "utf-16-le", "utf-8-sig", "utf-8", "cp1252", "latin1"]
    return ["utf-8-sig", "utf-8", "cp1252", "latin1"]


def _try_read_csv(path: str, encoding: str) -> pd.DataFrame:
    # sep=None sniffs delimiter: comma/semicolon/tab
    return pd.read_csv(path, encoding=encoding, sep=None, engine="python")


def _looks_like_index_series(s: pd.Series) -> bool:
    """
    Detect Excel 'Unnamed: 0' index col: 0..n or 1..n numeric monotonic.
    """
    try:
        # drop NaNs
        s2 = s.dropna()
        if s2.empty:
            return False
        # all numeric?
        nums = pd.to_numeric(s2, errors="coerce")
        if nums.isna().any():
            return False
        nums = nums.astype(int)

        # monotonic increasing and close to range?
        if not nums.is_monotonic_increasing:
            return False

        # allow 0..n-1 OR 1..n
        values = nums.to_list()
        n = len(values)
        if values == list(range(0, n)):
            return True
        if values == list(range(1, n + 1)):
            return True
        return False
    except Exception:
        return False


def _stitch_fragments(values: list[str]) -> str:
    """
    Stitch fragments that got split across columns.
    e.g. 'temp' + 'erature?' -> 'temperature?'
         'Sp' + 'alte' -> 'Spalte'
    """
    out = ""
    prev = ""
    for frag in values:
        frag = str(frag).strip()
        if not frag:
            continue

        if not out:
            out = frag
            prev = frag
            continue

        glue = False
        # glue if previous is short and both parts look like one word
        if prev and prev[-1].isalnum() and frag and frag[0].isalpha() and len(prev) <= 4:
            glue = True
        # glue if previous is single char
        if len(prev) == 1 and prev.isalpha() and frag and frag[0].isalpha():
            glue = True

        out = out + ("" if glue else " ") + frag
        prev = frag

    return " ".join(out.split()).strip()


def load_prompts_csv(path: str) -> pd.DataFrame:
    """
    Robust prompt loader:
    - tries multiple encodings
    - keeps 'Unnamed' unless it's truly an index
    - always reconstructs prompt by joining all non-empty cells per row
    """
    sample = _read_bytes(path)
    encodings = _pick_encodings(sample)

    df = None
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            df = _try_read_csv(path, enc)
            break
        except Exception as e:
            last_err = e
            df = None

    if df is None:
        raise ValueError(f"CSV konnte nicht gelesen werden. Letzter Fehler: {last_err!r}")

    # normalize col names
    df.columns = [str(c).strip() for c in df.columns]

    # drop fully empty columns
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)

    # drop 'Unnamed' ONLY if it looks like an index
    unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    for c in unnamed_cols:
        if _looks_like_index_series(df[c]):
            df = df.drop(columns=[c])

    if df.shape[1] == 0:
        raise ValueError("CSV hat nach Cleanup keine Spalten mehr.")

    prompts: list[str] = []
    for _, row in df.iterrows():
        vals = []
        for v in row.tolist():
            if pd.isna(v):
                continue
            s = str(v).strip()
            if not s:
                continue
            vals.append(s)

        p = _stitch_fragments(vals)
        if p:
            prompts.append(p)

    out = pd.DataFrame({"prompt": prompts})
    out = out[out["prompt"].astype(str).str.strip() != ""].reset_index(drop=True)
    return out


def repair_csv(input_csv: str, fixed_csv: str) -> int:
    df = load_prompts_csv(input_csv)
    df.to_csv(fixed_csv, index=False, encoding="utf-8", lineterminator="\n")
    print(f"✅ Repaired prompts written to: {fixed_csv}")
    print(f"✅ Rows: {len(df)} (column: 'prompt')")

    print("\nPreview (first 5):")
    for i, p in enumerate(df["prompt"].head(5).tolist(), start=1):
        print(f"{i:02d}: {p}")
    return 0


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.getenv("BATCH_INPUT_CSV", DEFAULT_INPUT_CSV))
    parser.add_argument("--fixed", default=os.getenv("BATCH_FIXED_CSV", DEFAULT_FIXED_CSV))
    parser.add_argument("--output", default=os.getenv("BATCH_OUTPUT_CSV", DEFAULT_OUTPUT_CSV))
    parser.add_argument("--sleep", type=float, default=float(os.getenv("BATCH_SLEEP", str(SLEEP_BETWEEN_CALLS_SEC))))
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()

    load_dotenv()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV nicht gefunden: {args.input}")

    setup_observability(os.getenv("LANGSMITH_PROJECT", "agent_fzi_test"))
    _ = get_langfuse_callbacks()

    if args.repair:
        raise SystemExit(repair_csv(args.input, args.fixed))

    df = load_prompts_csv(args.input)
    print(f"✅ Loaded {len(df)} prompts from: {args.input}")
    print(f"➡️  Writing results to: {args.output}\n")

    # Import only here to avoid circular import issues
    from orchestrator.router import run_router

    results: List[Dict[str, Any]] = []
    try:
        for i, prompt in enumerate(df["prompt"].tolist(), start=1):
            started = time.time()
            try:
                answer = run_router(prompt)
                status = "ok"
                error = ""
            except Exception as e:
                answer = ""
                status = "error"
                error = repr(e)

            dur_ms = int((time.time() - started) * 1000)
            results.append(
                {
                    "idx": i,
                    "timestamp": now_iso(),
                    "prompt": prompt,
                    "status": status,
                    "duration_ms": dur_ms,
                    "answer": answer,
                    "error": error,
                }
            )

            print(f"[{i:03d}/{len(df):03d}] {status} ({dur_ms} ms) - {prompt[:90]}")
            if args.sleep > 0:
                time.sleep(args.sleep)

    except KeyboardInterrupt:
        print("\n🛑 Abgebrochen per Ctrl+C – schreibe Ergebnisse bis hierhin...")

    pd.DataFrame(results).to_csv(args.output, index=False, encoding="utf-8", lineterminator="\n")
    print("\n✅ Done.")
    print(f"📄 Results saved to: {args.output}")


if __name__ == "__main__":
    main()