#!/usr/bin/env python3
"""
vibetell_cli.py — Engine v5-beta corpus tester

Usage:
    python3 vibetell_cli.py --llm passwords.csv --csprng random.txt
    python3 vibetell_cli.py --llm llm1.csv llm2.txt --csprng csprng.txt
    python3 vibetell_cli.py --unknown mystery.txt
    python3 vibetell_cli.py --llm passwords.csv --misses
    echo 'Kx#9mP!2vL@nQ7wR' | python3 vibetell_cli.py

Input formats supported:
    .csv  — expects a 'password' column (other columns ignored)
            optional 'model' column enables per-model breakdown
    .txt  — one password per line
          — read from stdin

Flags:
    --llm      <files>    Label these as LLM-generated (computes recall)
    --csprng   <files>    Label these as CSPRNG (computes FPR)
    --unknown  <files>    No ground truth — just run detection and report
    --misses              Print only missed passwords (FN for LLM, FP for CSPRNG)
    --verbose             Print every password with its verdict
    --signals             Print signal-level detail for every password (implies --verbose)
    --no-strip            Disable automatic prefix stripping; analyze the full credential
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import vibetell

# ═══════════════════════════════════════════════════════════════
# FILE LOADING
# ═══════════════════════════════════════════════════════════════

@dataclass
class PasswordEntry:
    password: str
    model:    str = ''
    source:   str = ''


def load_file(path: str) -> list[PasswordEntry]:
    """Load passwords from .csv, .txt, or stdin ('-')."""
    entries: list[PasswordEntry] = []

    if path == '-':
        for line in sys.stdin:
            pw = line.strip()
            if pw:
                entries.append(PasswordEntry(pw, source='<stdin>'))
        return entries

    p = Path(path)

    if p.suffix.lower() == '.csv':
        with open(p, newline='', encoding='utf-8', errors='replace') as fh:
            reader = csv.DictReader(fh)
            if 'password' not in (reader.fieldnames or []):
                print(f"  ⚠ {p.name}: CSV has no 'password' column — "
                      f"found {reader.fieldnames}", file=sys.stderr)
                return []
            has_model = 'model' in (reader.fieldnames or [])
            for row in reader:
                pw = row['password'].strip()
                if pw:
                    model = row.get('model', '').strip() if has_model else ''
                    entries.append(PasswordEntry(pw, model=model, source=p.name))
    else:
        with open(p, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                pw = line.strip()
                if pw:
                    entries.append(PasswordEntry(pw, source=p.name))

    return entries


# ═══════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════

VERDICTS = ['LLM_LIKELY', 'LLM_POSSIBLE', 'INCONCLUSIVE']
SKIP_REASONS = ['TOO_SHORT', 'TOO_LONG', 'NON_ASCII']

W = 72


def hbar(char='─', width=W):
    return char * width


def section(title: str):
    print(f"\n{'═' * W}")
    print(f"  {title}")
    print(f"{'═' * W}")


def report_corpus(
    label: str,
    entries: list[PasswordEntry],
    results: list[vibetell.Result],
    misses_only: bool = False,
    verbose: bool = False,
    show_signals: bool = False,
):
    """Print stats for one corpus (LLM, CSPRNG, or UNKNOWN)."""
    section(f"{label} — {len(entries)} passwords")

    verdict_counts: Counter = Counter()
    skip_counts: Counter = Counter()
    signal_counts: Counter = Counter()
    by_model: dict[str, Counter] = defaultdict(Counter)
    prefix_count = 0

    for entry, result in zip(entries, results):
        if result.verdict is None:
            skip_counts[result.reason] += 1
        else:
            verdict_counts[result.verdict] += 1
            signal_counts[result.signals] += 1
        if result.prefix is not None:
            prefix_count += 1
        if entry.model:
            if result.verdict:
                by_model[entry.model][result.verdict] += 1
            else:
                by_model[entry.model][result.reason] += 1

    total = len(entries)
    eligible = total - sum(skip_counts.values())

    # ─── Verdict distribution ─────────────────────────────────
    print(f"\n  Verdict Distribution (n={eligible} eligible, {sum(skip_counts.values())} skipped):")
    print(f"  {hbar()}")
    for v in VERDICTS:
        ct = verdict_counts.get(v, 0)
        pct = ct / eligible * 100 if eligible else 0
        bar = '█' * int(pct / 2)
        print(f"    {v:15s}  {ct:6d}  ({pct:5.1f}%)  {bar}")
    for r in SKIP_REASONS:
        ct = skip_counts.get(r, 0)
        if ct:
            print(f"    {r:15s}  {ct:6d}  (skipped)")

    # ─── Detection path breakdown ─────────────────────────────
    if any(signal_counts.values()):
        print(f"\n  Detection Paths:")
        print(f"  {hbar()}")
        for sig, ct in signal_counts.most_common():
            pct = ct / eligible * 100 if eligible else 0
            print(f"    {sig:22s}  {ct:6d}  ({pct:5.1f}%)")

    # ─── Key metrics ──────────────────────────────────────────
    detected = verdict_counts.get('LLM_LIKELY', 0) + verdict_counts.get('LLM_POSSIBLE', 0)
    likely   = verdict_counts.get('LLM_LIKELY', 0)

    if label == 'LLM':
        recall = detected / eligible * 100 if eligible else 0
        likely_pct = likely / eligible * 100 if eligible else 0
        fn = eligible - detected
        print(f"\n  Key Metrics:")
        print(f"  {hbar()}")
        print(f"    Recall (LIKELY+POSSIBLE):  {detected:6d}/{eligible}  = {recall:.2f}%")
        print(f"    Recall (LIKELY only):      {likely:6d}/{eligible}  = {likely_pct:.2f}%")
        print(f"    False negatives:           {fn:6d}")

    elif label == 'CSPRNG':
        fp_likely = verdict_counts.get('LLM_LIKELY', 0)
        fp_any    = detected
        fpr_likely = fp_likely / eligible * 100 if eligible else 0
        fpr_any    = fp_any / eligible * 100 if eligible else 0
        print(f"\n  Key Metrics:")
        print(f"  {hbar()}")
        print(f"    FPR (LIKELY):     {fp_likely:6d}/{eligible}  = {fpr_likely:.4f}%")
        print(f"    FPR (POSSIBLE+):  {fp_any:6d}/{eligible}  = {fpr_any:.4f}%")
        print(f"    True negatives:   {eligible - fp_any:6d}")

    else:
        det_pct = detected / eligible * 100 if eligible else 0
        print(f"\n  Detection Rate:")
        print(f"  {hbar()}")
        print(f"    Flagged (LIKELY+POSSIBLE): {detected:6d}/{eligible}  = {det_pct:.2f}%")
        print(f"    LIKELY:                    {likely:6d}")
        print(f"    POSSIBLE:                  {detected - likely:6d}")
        print(f"    INCONCLUSIVE:              {verdict_counts.get('INCONCLUSIVE', 0):6d}")

    # ─── Prefix stripping count ───────────────────────────────
    if prefix_count > 0:
        pct = prefix_count / total * 100
        print(f"    Prefix-stripped: {prefix_count} ({pct:.1f}%)")

    # ─── Per-model breakdown ──────────────────────────────────
    if by_model:
        print(f"\n  Per-Model Breakdown:")
        print(f"  {hbar()}")
        print(f"    {'Model':<25s} {'Total':>6s} {'LIKELY':>7s} {'POSSIB':>7s} {'INCON':>7s} {'Skip':>6s}  {'Recall':>7s}")
        print(f"    {hbar(width=W-4)}")
        for model in sorted(by_model.keys()):
            mc = by_model[model]
            m_likely = mc.get('LLM_LIKELY', 0)
            m_poss   = mc.get('LLM_POSSIBLE', 0)
            m_inc    = mc.get('INCONCLUSIVE', 0)
            m_skip   = sum(mc.get(r, 0) for r in SKIP_REASONS)
            m_total  = m_likely + m_poss + m_inc + m_skip
            m_elig   = m_likely + m_poss + m_inc
            m_det    = m_likely + m_poss
            m_recall = f"{m_det / m_elig * 100:.1f}%" if m_elig else "n/a"
            print(f"    {model:<25s} {m_total:6d} {m_likely:7d} {m_poss:7d} {m_inc:7d} {m_skip:6d}  {m_recall:>7s}")

    # ─── Signal histograms ────────────────────────────────────
    scts = [r.features.sc for r in results if r.features]
    llrs = [r.features.llr for r in results if r.features]
    if len(scts) >= 2:
        print(f"\n  Signal Statistics:")
        print(f"  {hbar()}")
        print(f"    SCT   min={min(scts):.4f}  max={max(scts):.4f}  "
              f"mean={sum(scts)/len(scts):.4f}  "
              f"<0.024: {sum(1 for x in scts if x < 0.024)} ({sum(1 for x in scts if x < 0.024)/len(scts)*100:.1f}%)  "
              f"=0: {sum(1 for x in scts if x == 0)} ({sum(1 for x in scts if x == 0)/len(scts)*100:.1f}%)")
        print(f"    LLR   min={min(llrs):.2f}  max={max(llrs):.2f}  "
              f"mean={sum(llrs)/len(llrs):.2f}  "
              f">0: {sum(1 for x in llrs if x > 0)} ({sum(1 for x in llrs if x > 0)/len(llrs)*100:.1f}%)")

        # ─── Soft indicator stats ─────────────────────────────────
        rare_ct = sum(1 for r in results if r.features and r.features.rs)
        rpt_ct  = sum(1 for r in results if r.features and r.features.hr)
        print(f"    Rare symbols: {rare_ct} ({rare_ct/len(scts)*100:.1f}%)  "
              f"Repeats: {rpt_ct} ({rpt_ct/len(scts)*100:.1f}%)")

    # ─── Missed passwords / verbose ───────────────────────────
    if misses_only or verbose:
        if label == 'LLM':
            misses = [(e, r) for e, r in zip(entries, results)
                      if r.verdict in ('INCONCLUSIVE', None)]
            if misses:
                print(f"\n  False Negatives ({len(misses)}):")
                print(f"  {hbar()}")
                _print_password_list(misses, show_signals)
        elif label == 'CSPRNG':
            fps = [(e, r) for e, r in zip(entries, results)
                   if r.verdict in ('LLM_LIKELY', 'LLM_POSSIBLE')]
            if fps:
                print(f"\n  False Positives ({len(fps)}):")
                print(f"  {hbar()}")
                _print_password_list(fps, show_signals)
        else:
            if verbose:
                print(f"\n  All Results:")
                print(f"  {hbar()}")
                _print_password_list(list(zip(entries, results)), show_signals)


def _print_password_list(items: list[tuple[PasswordEntry, vibetell.Result]], show_signals: bool):
    for entry, result in items:
        pw = entry.password
        v = result.verdict or result.reason
        sig = result.signals

        pw_display = pw if len(pw) <= 44 else pw[:41] + '...'
        model_str = f"  [{entry.model}]" if entry.model else ""

        # Prefix stripping visibility
        prefix_str = ""
        if result.prefix is not None:
            prefix_str = f"  prefix: {result.prefix} -> analyzed: {result.analyzed_portion}"
            if len(prefix_str) > 60:
                prefix_str = f"  prefix: {result.prefix}"

        if show_signals and result.features:
            f = result.features
            cl = f.class_llr
            soft = []
            if f.rs: soft.append('rare')
            if f.hr: soft.append('rpt')
            soft_str = '+'.join(soft) if soft else '—'
            print(f"    {pw_display:<44s}  {v:15s}  {sig:22s}  "
                  f"SCT={f.sc:.4f}  LLR={f.llr:+6.2f}  "
                  f"D={cl.digit_llr:+.1f} L={cl.letter_llr:+.1f} S={cl.symbol_llr:+.1f}  "
                  f"strength={result.signal_strength:.2f}  soft={soft_str}"
                  f"{prefix_str}{model_str}")
        else:
            print(f"    {pw_display:<44s}  {v:15s}  via {sig}"
                  f"{prefix_str}{model_str}")


# ═══════════════════════════════════════════════════════════════
# COMBINED METRICS
# ═══════════════════════════════════════════════════════════════

def report_combined(
    llm_results: list[vibetell.Result],
    csprng_results: list[vibetell.Result],
):
    section("COMBINED METRICS")

    llm_elig    = [r for r in llm_results if r.verdict is not None]
    csprng_elig = [r for r in csprng_results if r.verdict is not None]

    if not llm_elig or not csprng_elig:
        print("  Insufficient data for combined metrics.")
        return

    tp_likely = sum(1 for r in llm_elig if r.verdict == 'LLM_LIKELY')
    fp_likely = sum(1 for r in csprng_elig if r.verdict == 'LLM_LIKELY')
    fn_likely = len(llm_elig) - tp_likely
    tn_likely = len(csprng_elig) - fp_likely

    tp_any = sum(1 for r in llm_elig if r.verdict in ('LLM_LIKELY', 'LLM_POSSIBLE'))
    fp_any = sum(1 for r in csprng_elig if r.verdict in ('LLM_LIKELY', 'LLM_POSSIBLE'))
    fn_any = len(llm_elig) - tp_any
    tn_any = len(csprng_elig) - fp_any

    def safe_div(a, b): return a / b if b else 0.0

    print(f"\n  {'Threshold':<22s}  {'Prec':>7s}  {'Recall':>7s}  {'F1':>7s}  {'FPR':>9s}  {'TP':>6s}  {'FP':>5s}  {'FN':>5s}  {'TN':>6s}")
    print(f"  {hbar(width=W-2)}")

    for lbl, tp, fp, fn, tn in [
        ('LIKELY only', tp_likely, fp_likely, fn_likely, tn_likely),
        ('LIKELY+POSSIBLE', tp_any, fp_any, fn_any, tn_any),
    ]:
        prec   = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1     = safe_div(2 * prec * recall, prec + recall)
        fpr    = safe_div(fp, fp + tn)
        print(f"  {lbl:<22s}  {prec:7.4f}  {recall:7.4f}  {f1:7.4f}  {fpr:9.6f}  {tp:6d}  {fp:5d}  {fn:5d}  {tn:6d}")

    # Confidence calibration
    print(f"\n  Signal Strength Distribution (detected passwords):")
    print(f"  {hbar()}")
    all_detected = [r for r in llm_elig + csprng_elig
                    if r.verdict in ('LLM_LIKELY', 'LLM_POSSIBLE')]
    if all_detected:
        confs = [r.signal_strength for r in all_detected]
        buckets = [0] * 10
        for c in confs:
            idx = min(int(c * 10), 9)
            buckets[idx] += 1
        for i, ct in enumerate(buckets):
            if ct:
                lo = i * 10
                hi = (i + 1) * 10
                bar = '█' * max(1, int(ct / max(buckets) * 30))
                print(f"    {lo:3d}–{hi:3d}:  {ct:5d}  {bar}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=f'vibetell {vibetell.__version__} corpus tester',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--llm', nargs='+', metavar='FILE',
                        help='LLM-generated password files (recall)')
    parser.add_argument('--csprng', nargs='+', metavar='FILE',
                        help='CSPRNG password files (FPR)')
    parser.add_argument('--unknown', nargs='+', metavar='FILE',
                        help='Unknown-origin password files')
    parser.add_argument('--misses', action='store_true',
                        help='Print missed passwords (FN for LLM, FP for CSPRNG)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print every password with verdict')
    parser.add_argument('--signals', action='store_true',
                        help='Print signal-level detail (implies --verbose)')
    parser.add_argument('--no-strip', dest='no_strip', action='store_true',
                        help='Disable automatic prefix stripping; analyze the full credential')

    args = parser.parse_args()

    if not args.llm and not args.csprng and not args.unknown:
        if not sys.stdin.isatty():
            # Data is being piped in — treat as unknown-origin credentials
            args.unknown = ['-']
        else:
            parser.print_help()
            sys.exit(1)

    if args.signals:
        args.verbose = True

    print(f"{'═' * W}")
    print(f"  vibetell {vibetell.__version__} · corpus test")
    print(f"{'═' * W}")

    llm_entries:     list[PasswordEntry] = []
    llm_results:     list[vibetell.Result] = []
    csprng_entries:  list[PasswordEntry] = []
    csprng_results:  list[vibetell.Result] = []
    unknown_entries: list[PasswordEntry] = []
    unknown_results: list[vibetell.Result] = []

    if args.llm:
        for path in args.llm:
            entries = load_file(path)
            src_name = '<stdin>' if path == '-' else Path(path).name
            print(f"  Loaded {len(entries):,d} passwords from {src_name} (LLM)")
            llm_entries.extend(entries)
        llm_results = [vibetell.analyze(e.password, no_strip=args.no_strip) for e in llm_entries]

    if args.csprng:
        for path in args.csprng:
            entries = load_file(path)
            src_name = '<stdin>' if path == '-' else Path(path).name
            print(f"  Loaded {len(entries):,d} passwords from {src_name} (CSPRNG)")
            csprng_entries.extend(entries)
        csprng_results = [vibetell.analyze(e.password, no_strip=args.no_strip) for e in csprng_entries]

    if args.unknown:
        for path in args.unknown:
            entries = load_file(path)
            src_name = '<stdin>' if path == '-' else Path(path).name
            print(f"  Loaded {len(entries):,d} passwords from {src_name} (unknown)")
            unknown_entries.extend(entries)
        unknown_results = [vibetell.analyze(e.password, no_strip=args.no_strip) for e in unknown_entries]

    if llm_entries:
        report_corpus('LLM', llm_entries, llm_results,
                      misses_only=args.misses, verbose=args.verbose,
                      show_signals=args.signals)

    if csprng_entries:
        report_corpus('CSPRNG', csprng_entries, csprng_results,
                      misses_only=args.misses, verbose=args.verbose,
                      show_signals=args.signals)

    if unknown_entries:
        report_corpus('UNKNOWN', unknown_entries, unknown_results,
                      misses_only=args.misses, verbose=args.verbose,
                      show_signals=args.signals)

    if llm_entries and csprng_entries:
        report_combined(llm_results, csprng_results)

    print(f"\n{'═' * W}")
    print(f"  Done.")
    print(f"{'═' * W}")


if __name__ == '__main__':
    main()
