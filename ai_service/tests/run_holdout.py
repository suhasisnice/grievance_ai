"""
Scores classify_complaint() against the HELD-OUT test set.

    python -m ai_service.tests.run_holdout

This is the real accuracy number. run_examples.py grades the model on the same
15 examples that are baked into its own prompt — open book, which is why it
scores 15/15. This scores it on 30 complaints it has never seen.

Options:
    --normalize     Run the full production path (normalize_text -> classify)
                    instead of classifying the raw text directly. Doubles the
                    API calls and therefore the runtime. See "WHICH PATH AM I
                    TESTING?" below.
    --start N       Skip the first N cases (for resuming a chunked run)
    --limit N       Only run N cases
    --delay N       Seconds between API calls (default 15, for free-tier limits)

WHICH PATH AM I TESTING?
    By default this calls classify_complaint(raw_text) directly, which is what
    run_examples.py does. But production actually runs normalize_text() first
    to translate regional-language input to English, THEN classifies. For the
    Kannada/Tamil/Telugu/Hindi cases in the holdout set those are meaningfully
    different tests. Run both and compare — if --normalize scores better, the
    translation step is earning its API call; if it scores worse, it's losing
    detail in translation and worth a look.

READING THE OUTPUT:
    The headline number is EXACT MATCH (category AND priority both right).
    But the number that actually matters for the product is in the CONFIDENCE
    CALIBRATION section: this system routes anything below the human-review
    threshold to a person. So a wrong answer with low confidence is a caught
    error, not a failure. A wrong answer with HIGH confidence is the dangerous
    kind — that's a misrouted ticket nobody double-checks.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

from ai_service.categories import HUMAN_REVIEW_CONFIDENCE_THRESHOLD
from ai_service.service import classify_complaint, normalize_text
from ai_service.tests.holdout_set import HOLDOUT_SET

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "holdout_results")


def _accepted_categories(case: dict) -> list:
    return [case["expected_category"]] + list(case.get("also_accept_category", []))


def _accepted_priorities(case: dict) -> list:
    return [case["expected_priority"]] + list(case.get("also_accept_priority", []))


def _truncate(text: str, length: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= length else text[: length - 1] + "…"


def run_case(case: dict, use_normalize: bool) -> dict:
    """Classifies one holdout case and grades it. Never raises — a case that
    blows up is recorded as an error row rather than killing the whole run,
    because a 25-minute run dying on case 28 is miserable."""
    text = case["text"]
    language = None
    try:
        if use_normalize:
            normalized = normalize_text(text)
            text = normalized.get("canonical_text") or text
            language = normalized.get("language")
        result = classify_complaint(text)
    except Exception as exc:  # noqa: BLE001 — a scorer must never crash mid-run
        return {
            "text": case["text"],
            "error": f"{type(exc).__name__}: {exc}",
            "source": "error",
        }

    got_category = result.get("category")
    got_priority = result.get("priority")
    category_ok = got_category in _accepted_categories(case)
    priority_ok = got_priority in _accepted_priorities(case)

    return {
        "text": case["text"],
        "normalized_text": text if use_normalize else None,
        "detected_language": language,
        "expected_category": case["expected_category"],
        "expected_priority": case["expected_priority"],
        "got_category": got_category,
        "got_priority": got_priority,
        "category_ok": category_ok,
        "priority_ok": priority_ok,
        "exact_ok": category_ok and priority_ok,
        "confidence": result.get("confidence", 0.0),
        "source": result.get("source", "?"),
        "summary": result.get("summary"),
        "location_text": result.get("location_text"),
        "borderline": bool(case.get("borderline")),
        "note": case.get("note"),
    }


def report(rows: list, use_normalize: bool) -> dict:
    """Prints the scorecard and returns the summary dict that gets saved."""
    errors = [r for r in rows if r.get("source") == "error"]
    fallbacks = [r for r in rows if r.get("source") == "fallback"]

    # Rows the AI layer never actually answered can't be scored as wrong —
    # they'd understate real accuracy. They're reported separately instead.
    scored = [r for r in rows if r.get("source") in ("gemini", "groq")]

    print("\n" + "=" * 78)
    print(f"HOLDOUT SCORECARD  ({'normalize -> classify' if use_normalize else 'classify on raw text'})")
    print("=" * 78)

    if not scored:
        print("\nNothing was scored — every call fell back or errored.")
        print("Check your API key and rate limits, then re-run.")
        return {"scored": 0, "errors": len(errors), "fallbacks": len(fallbacks)}

    n = len(scored)
    cat_hits = sum(1 for r in scored if r["category_ok"])
    pri_hits = sum(1 for r in scored if r["priority_ok"])
    exact_hits = sum(1 for r in scored if r["exact_ok"])

    print(f"\nScored {n} of {len(rows)} cases"
          f"   (excluded: {len(fallbacks)} fallback, {len(errors)} error)")
    print(f"\n  Category correct    {cat_hits:>3}/{n}   {cat_hits / n:6.1%}")
    print(f"  Priority correct    {pri_hits:>3}/{n}   {pri_hits / n:6.1%}")
    print(f"  EXACT MATCH         {exact_hits:>3}/{n}   {exact_hits / n:6.1%}   <-- headline")

    # Strict score: borderline cases counted against their primary label only.
    strict = [r for r in scored if not r["borderline"]]
    if strict:
        strict_exact = sum(1 for r in strict if r["exact_ok"])
        print(f"\n  Excluding the {len(scored) - len(strict)} borderline cases:"
              f"   {strict_exact}/{len(strict)}   {strict_exact / len(strict):6.1%}")

    # ---- where it went wrong -------------------------------------------
    cat_misses = [r for r in scored if not r["category_ok"]]
    if cat_misses:
        print("\n" + "-" * 78)
        print("CATEGORY MISSES")
        print("-" * 78)
        pairs = Counter((r["expected_category"], r["got_category"]) for r in cat_misses)
        for (want, got), count in pairs.most_common():
            print(f"  {want:>14}  ->  {got:<14}  x{count}")

    pri_misses = [r for r in scored if not r["priority_ok"]]
    if pri_misses:
        print("\n" + "-" * 78)
        print("PRIORITY MISSES")
        print("-" * 78)
        pairs = Counter((r["expected_priority"], r["got_priority"]) for r in pri_misses)
        for (want, got), count in pairs.most_common():
            direction = "OVER-escalated" if _severity(got) > _severity(want) else "UNDER-escalated"
            print(f"  {want:>10}  ->  {got:<10}  x{count}   ({direction})")

        over = sum(1 for r in pri_misses if _severity(r["got_priority"]) > _severity(r["expected_priority"]))
        under = len(pri_misses) - over
        print(f"\n  {over} over-escalated (wastes officer time), "
              f"{under} under-escalated (real risk — a critical treated as routine)")

    # ---- confidence calibration ----------------------------------------
    print("\n" + "-" * 78)
    print(f"CONFIDENCE CALIBRATION  (human-review threshold = {HUMAN_REVIEW_CONFIDENCE_THRESHOLD})")
    print("-" * 78)

    right = [r for r in scored if r["exact_ok"]]
    wrong = [r for r in scored if not r["exact_ok"]]
    if right:
        print(f"  mean confidence when CORRECT    {sum(r['confidence'] for r in right) / len(right):.2f}")
    if wrong:
        print(f"  mean confidence when WRONG      {sum(r['confidence'] for r in wrong) / len(wrong):.2f}")
        print("  (want a clear gap here — if they're close, confidence isn't")
        print("   telling you anything and the review threshold can't work)")

    caught = [r for r in wrong if r["confidence"] < HUMAN_REVIEW_CONFIDENCE_THRESHOLD]
    escaped = [r for r in wrong if r["confidence"] >= HUMAN_REVIEW_CONFIDENCE_THRESHOLD]
    flagged_ok = [r for r in right if r["confidence"] < HUMAN_REVIEW_CONFIDENCE_THRESHOLD]

    print(f"\n  Routing everything below {HUMAN_REVIEW_CONFIDENCE_THRESHOLD} to a human would:")
    print(f"    catch    {len(caught):>3} of {len(wrong)} wrong answers")
    print(f"    MISS     {len(escaped):>3} wrong answers that looked confident   <-- the dangerous ones")
    print(f"    cost     {len(flagged_ok):>3} needless reviews of correct answers")

    if escaped:
        print("\n  Confidently wrong:")
        for r in escaped:
            print(f"    [{r['confidence']:.2f}] {_truncate(r['text'], 52)}")
            print(f"           want {r['expected_category']}/{r['expected_priority']}"
                  f"  got {r['got_category']}/{r['got_priority']}")

    # ---- every miss, in full -------------------------------------------
    all_misses = [r for r in scored if not r["exact_ok"]]
    if all_misses:
        print("\n" + "-" * 78)
        print("ALL MISSES IN DETAIL")
        print("-" * 78)
        for r in all_misses:
            tag = " [borderline]" if r["borderline"] else ""
            print(f"\n  {_truncate(r['text'], 70)}{tag}")
            print(f"    want: {r['expected_category']}/{r['expected_priority']}"
                  f"    got: {r['got_category']}/{r['got_priority']}"
                  f"    conf {r['confidence']:.2f}  via {r['source']}")
            if r.get("normalized_text"):
                print(f"    normalized: {_truncate(r['normalized_text'], 66)}")
            if r.get("note"):
                print(f"    label reason: {r['note']}")

    # ---- provider mix ---------------------------------------------------
    sources = Counter(r.get("source") for r in rows)
    print("\n" + "-" * 78)
    print("PROVIDER MIX")
    print("-" * 78)
    for source, count in sources.most_common():
        print(f"  {source:<10} {count}")
    if sources.get("groq"):
        print("\n  Note: some answers came from the Groq fallback, so this score")
        print("  is a blend of two models. Re-run when Gemini isn't throttled")
        print("  for a clean single-model number.")

    return {
        "scored": n,
        "total": len(rows),
        "fallbacks": len(fallbacks),
        "errors": len(errors),
        "category_accuracy": cat_hits / n,
        "priority_accuracy": pri_hits / n,
        "exact_accuracy": exact_hits / n,
        "mean_confidence_correct": (sum(r["confidence"] for r in right) / len(right)) if right else None,
        "mean_confidence_wrong": (sum(r["confidence"] for r in wrong) / len(wrong)) if wrong else None,
        "confidently_wrong": len(escaped),
        "provider_mix": dict(sources),
        "used_normalize": use_normalize,
    }


_SEVERITY = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _severity(priority: str) -> int:
    return _SEVERITY.get(priority, -1)


def main():
    parser = argparse.ArgumentParser(description="Score the classifier against the held-out set.")
    parser.add_argument("--normalize", action="store_true",
                        help="run normalize_text() first (the real production path)")
    parser.add_argument("--start", type=int, default=0, help="skip the first N cases")
    parser.add_argument("--limit", type=int, default=None, help="only run N cases")
    parser.add_argument("--delay", type=float, default=15.0,
                        help="seconds between calls (default 15 for free-tier limits)")
    args = parser.parse_args()

    cases = HOLDOUT_SET[args.start:]
    if args.limit:
        cases = cases[: args.limit]

    calls_per_case = 2 if args.normalize else 1
    est_minutes = (len(cases) * args.delay * calls_per_case) / 60
    print(f"\nRunning {len(cases)} holdout cases"
          f" ({'normalize -> classify' if args.normalize else 'classify only'}),"
          f" {args.delay:g}s between calls.")
    print(f"Estimated runtime: ~{est_minutes:.0f} minutes. Progress prints as it goes.\n")

    rows = []
    os.makedirs(RESULTS_DIR, exist_ok=True)
    partial_path = os.path.join(RESULTS_DIR, "_partial.json")

    for i, case in enumerate(cases):
        if i > 0:
            time.sleep(args.delay)
        if args.normalize and i > 0:
            time.sleep(args.delay)  # second call for this case needs its own slot

        row = run_case(case, args.normalize)
        rows.append(row)

        # Save after every case — a 25-minute run should never lose its work
        # to a crash or a Ctrl-C on case 29.
        with open(partial_path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=2)

        if row.get("source") == "error":
            mark = "!"
        elif row.get("source") == "fallback":
            mark = "-"
        elif row["exact_ok"]:
            mark = "OK"
        elif row["category_ok"]:
            mark = "~"   # right department, wrong urgency
        else:
            mark = "X"
        print(f"  [{args.start + i + 1:>2}/{args.start + len(cases)}] {mark:<2} "
              f"{_truncate(case['text'], 56)}")

    summary = report(rows, args.normalize)

    suffix = "normalized" if args.normalize else "raw"
    out_path = os.path.join(RESULTS_DIR, f"holdout_{suffix}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, ensure_ascii=False, indent=2)
    if os.path.exists(partial_path):
        os.remove(partial_path)

    print(f"\nFull results saved to: {out_path}")
    print("Commit this file. After any prompt or example change, re-run and")
    print("diff the two — that's how you tell an improvement from a regression.\n")


if __name__ == "__main__":
    main()
