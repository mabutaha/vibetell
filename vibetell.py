"""
vibetell — LLM password detection engine (v0.5.0-beta)

Detects LLM-generated passwords from a single sample using two signals:
  1. SCT (Same-Class Transition rate) — structural (parameter-free)
  2. LLR (Log-Likelihood Ratio) — vocabulary (corpus-fitted)

Public API:
    analyze(pw) -> Result
    analyze_batch(passwords) -> list[Result]
    extract_features(pw) -> Features
    strip_prefix(pw) -> tuple[str, str | None]
    compute_sct(pw) -> float
    expected_sct_exact(pw) -> float
    char_class(c) -> str
    class_template(pw) -> str

Usage:
    import vibetell
    result = vibetell.analyze("Kx#9mP!2vL@nQ7wR")
    print(result.verdict)           # LLM_LIKELY | LLM_POSSIBLE | INCONCLUSIVE
    print(result.signal_strength)   # 0.0 - 0.99, NOT a probability
    print(result.signals)           # detection path label
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

__version__ = "0.5.0-beta"

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

SCT_THRESHOLD  = 0.024   # global minimum — no length-adaptive threshold goes below this
LLR_THRESHOLD  = 0.0
MIN_LENGTH     = 12
MAX_LENGTH     = 256
CONFIDENCE_CAP = 0.99

# ─── Length-adaptive SCT threshold table ─────────────────
# Calibrated from 50K CSPRNG passwords per length bucket.
# Each entry: (min_length, threshold) where threshold = (k+0.5)/(n-1)
# for the maximum k same-class pairs where P(pairs <= k) <= 0.3% FPR.
# Monotonically non-decreasing. For a password of length n, use the
# entry with the largest min_length <= n.
_SCT_THRESHOLD_TABLE: list[tuple[int, float]] = [
    # length  threshold    pairs allowed   calibrated FPR
    (  12,    0.024   ),  # 0 pairs        existing behavior
    (  28,    0.0556  ),  # 1 pair         0.160%
    (  36,    0.0714  ),  # 2 pairs        0.116%
    (  40,    0.0897  ),  # 3 pairs        0.184%
    (  48,    0.0957  ),  # 4 pairs        0.122%
    (  56,    0.1182  ),  # 6 pairs        0.242%
    (  64,    0.1190  ),  # 7 pairs        0.140%
    (  80,    0.1456  ),  # 11 pairs       0.268%
    (  96,    0.1526  ),  # 14 pairs       0.170%
    ( 128,    0.1693  ),  # 21 pairs       0.204%
]

# ─── Symbolless LLR threshold table ─────────────────────
# Calibrated from 25K CSPRNG alphanumeric-only ([A-Za-z0-9]) passwords
# per length bucket. Each entry: (min_length, threshold) at the 99.7th
# percentile — 99.7% of CSPRNG alphanumeric passwords have LLR below
# this value. For symbolless passwords (counts['S']==0), LLR > this
# threshold is anomalous.
_SYMBOLLESS_LLR_TABLE: list[tuple[int, float]] = [
    # length  threshold    (99.7th percentile of alphanumeric CSPRNG)
    (  16,     2.04  ),
    (  20,     0.41  ),
    (  24,    -1.26  ),
    (  28,    -3.06  ),
    (  32,    -4.88  ),
    (  36,    -7.35  ),
    (  40,    -8.52  ),
    (  44,   -10.57  ),
    (  48,   -13.88  ),
    (  56,   -18.55  ),
    (  64,   -22.98  ),
    (  80,   -33.38  ),
    (  96,   -42.70  ),
    ( 128,   -64.62  ),
]

# ─── LLR frequency tables ────────────────────────────────────

_DIGITS = {
    '9': 0.2313, '2': 0.2195, '7': 0.1885, '4': 0.0991,
    '8': 0.0959, '3': 0.0511, '6': 0.0442, '5': 0.0387,
    '1': 0.0284, '0': 0.0035,
}

_UPPER = {
    'L': 0.1817, 'Q': 0.1454, 'K': 0.1009, 'P': 0.0992, 'R': 0.0962,
    'T': 0.0615, 'X': 0.0594, 'Z': 0.0504, 'G': 0.0400, 'N': 0.0392,
    'W': 0.0266, 'J': 0.0142, 'V': 0.0139, 'F': 0.0113, 'B': 0.0112,
    'H': 0.0112, 'Y': 0.0102, 'D': 0.0084, 'M': 0.0059, 'A': 0.0032,
    'S': 0.0027, 'C': 0.0026, 'E': 0.0023, 'U': 0.0020, 'I': 0.0002,
    'O': 0.0001,
}

_LOWER = {
    'm': 0.1639, 'v': 0.1493, 'x': 0.1126, 'n': 0.1072, 'p': 0.0812,
    'w': 0.0766, 'q': 0.0512, 'k': 0.0506, 'z': 0.0345, 't': 0.0264,
    'r': 0.0255, 'j': 0.0206, 'c': 0.0126, 's': 0.0121, 'f': 0.0117,
    'b': 0.0109, 'd': 0.0091, 'h': 0.0091, 'e': 0.0071, 'y': 0.0065,
    'u': 0.0060, 'g': 0.0046, 'a': 0.0046, 'l': 0.0039, 'o': 0.0012,
    'i': 0.0009,
}

_SYMS = {
    '#': 0.2384, '$': 0.2131, '@': 0.1984, '!': 0.1934, '&': 0.0732,
    '^': 0.0409, '*': 0.0229, '%': 0.0162, '?': 0.0013, '+': 0.0006,
    '_': 0.0006, '-': 0.0002, '=': 0.0002, ')': 0.0001, '~': 0.0001,
    '(': 0.00005, '\\': 0.00001, ']': 0.00001, '{': 0.00001, '}': 0.00001,
    '"': 0.00001, "'": 0.00001, ',': 0.00001, '.': 0.00001, '/': 0.00001,
    ':': 0.00001, ';': 0.00001, '<': 0.00001, '>': 0.00001, '[': 0.00001,
    '`': 0.00001, '|': 0.00001,
}

_UNIFORM = {'D': 1 / 10, 'U': 1 / 26, 'l': 1 / 26, 'S': 1 / 32}

LLR_TABLE: dict[str, float] = {}
for _freq, _cls in [(_DIGITS, 'D'), (_UPPER, 'U'), (_LOWER, 'l'), (_SYMS, 'S')]:
    for _ch, _p in _freq.items():
        LLR_TABLE[_ch] = math.log(max(_p, 1e-6) / _UNIFORM[_cls])

LLM_SYMBOLS = set('#$@!&^*%?+_-=)~(')

# ─── Prefix table ────────────────────────────────────────────

PREFIX_PATTERNS: list[tuple[str | re.Pattern, str]] = [
    # Env-file KEY=value patterns (checked first — longer, unambiguous)
    ("JWT_SECRET=", "jwt-secret"),
    ("SECRET_KEY=", "secret-key"),
    ("API_KEY=", "api-key"),
    ("API_SECRET=", "api-secret"),
    ("AUTH_TOKEN=", "auth-token"),
    ("ACCESS_TOKEN=", "access-token"),
    ("PRIVATE_KEY=", "private-key"),
    ("DATABASE_URL=", "database-url"),
    # Django
    ("django-insecure-", "django"),
    # Anthropic API keys
    (re.compile(r'^sk-ant-[a-z0-9]+-'), "anthropic"),
    # OpenRouter
    ("sk-or-v1-", "openrouter"),
    # OpenAI project keys
    ("sk-proj-", "openai-proj"),
    # OpenAI general (after more specific sk- prefixes)
    # NOTE: sk- is only 3 chars. Validated separately for FPR safety.
    ("sk-", "openai"),
    # GitHub
    ("ghp_", "github-pat"),
    ("gho_", "github-oauth"),
    ("ghs_", "github-server"),
    ("github_pat_", "github-finegrained"),
    # GitLab
    ("glpat-", "gitlab"),
    # AWS
    ("AKIA", "aws"),
    # Stripe
    ("whsec_", "stripe-webhook"),
    ("sk_live_", "stripe-live"),
    ("sk_test_", "stripe-test"),
    # Slack
    ("xoxb-", "slack-bot"),
    ("xoxp-", "slack-user"),
    # npm / PyPI / SendGrid
    ("npm_", "npm"),
    ("pypi-", "pypi"),
    ("SG.", "sendgrid"),
    # Supabase
    ("sbp_", "supabase"),
    # Vault
    ("hvs.", "vault"),
]


# ═══════════════════════════════════════════════════════════════
# THRESHOLD FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def sct_threshold(n: int) -> float:
    """Length-adaptive SCT threshold calibrated to ~0.3% CSPRNG FPR.

    At short lengths, this equals SCT_THRESHOLD (requires perfect alternation).
    At long lengths, it relaxes to allow several same-class pairs while
    maintaining the same false positive rate.

    Uses a step-function lookup: for a password of length n, returns the
    threshold from the entry with the largest min_length <= n.
    """
    threshold = SCT_THRESHOLD
    for min_len, thr in _SCT_THRESHOLD_TABLE:
        if n >= min_len:
            threshold = thr
        else:
            break
    return threshold


def symbolless_llr_threshold(n: int) -> float:
    """Shifted LLR threshold for passwords with zero symbols.

    For alphanumeric-only passwords (counts['S']==0), the standard LLR > 0
    threshold is unreachable because the symbol component (which contributes
    +1.8 to +2.0 per character) is absent. This function returns the 99.7th
    percentile of the CSPRNG alphanumeric LLR distribution at the given length.

    Returns a negative value — LLR above this threshold is anomalous for
    a CSPRNG alphanumeric password.
    """
    threshold = LLR_THRESHOLD
    for min_len, thr in _SYMBOLLESS_LLR_TABLE:
        if n >= min_len:
            threshold = thr
        else:
            break
    return threshold


# ═══════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class ClassLLR:
    digit_llr:  float = 0.0
    letter_llr: float = 0.0
    symbol_llr: float = 0.0
    digit_n:    int   = 0
    letter_n:   int   = 0
    symbol_n:   int   = 0

    @property
    def digit_avg(self)  -> float: return self.digit_llr  / self.digit_n  if self.digit_n  else 0.0
    @property
    def letter_avg(self) -> float: return self.letter_llr / self.letter_n if self.letter_n else 0.0
    @property
    def symbol_avg(self) -> float: return self.symbol_llr / self.symbol_n if self.symbol_n else 0.0


@dataclass
class Features:
    n:          int
    sc:         float        # SCT rate
    es:         float        # E[SCT] under independence
    zs:         float        # z-score
    sigma:      float
    llr:        float        # total LLR
    class_llr:  ClassLLR
    hr:         bool         # has_repeats
    rs:         bool         # has_rare_symbols
    mr:         int          # max same-class run
    nc:         int          # number of classes present
    counts:     dict


@dataclass
class Result:
    verdict:          str | None      # LLM_LIKELY | LLM_POSSIBLE | INCONCLUSIVE | None
    signals:          str             # detection path label
    signal_strength:  float           # Signal intensity within verdict tier.
                                      # NOT a probability of LLM origin.
                                      # Higher = more signals fired, stronger agreement.
    features:         Features | None = None
    reason:           str | None = None   # TOO_SHORT | TOO_LONG | NON_ASCII
    prefix:           str | None = None   # stripped prefix label, or None
    analyzed_portion: str = ""            # the string that was actually analyzed


# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def char_class(c: str) -> str:
    """Return 'U', 'l', 'D', or 'S' for a printable ASCII character."""
    if 'A' <= c <= 'Z': return 'U'
    if 'a' <= c <= 'z': return 'l'
    if '0' <= c <= '9': return 'D'
    return 'S'


def class_template(pw: str) -> str:
    """Return the character-class sequence for a password (e.g. 'UlDSlDU')."""
    return ''.join(char_class(c) for c in pw)


def compute_sct(pw: str) -> float:
    """Compute the Same-Class Transition rate for a password."""
    n = len(pw)
    if n < 2:
        return 0.0
    seq = [char_class(c) for c in pw]
    same = sum(1 for i in range(n - 1) if seq[i] == seq[i + 1])
    return same / (n - 1)


def expected_sct_exact(pw: str) -> float:
    """Compute E[SCT] under independence for a password's class composition."""
    n = len(pw)
    if n < 2:
        return 0.0
    seq = [char_class(c) for c in pw]
    counts = Counter(seq)
    vals = [counts.get(k, 0) for k in 'UlDS']
    A = sum(k * (k - 1) for k in vals)
    return A / (n * (n - 1))


def has_rare_symbols(pw: str) -> bool:
    for c in pw:
        code = ord(c)
        is_symbol = not (48 <= code <= 57 or 65 <= code <= 90 or 97 <= code <= 122)
        if is_symbol and c not in LLM_SYMBOLS:
            return True
    return False


def has_repeats(pw: str) -> bool:
    return len(set(pw)) < len(pw)


def _normal_cdf(z: float) -> float:
    if z > 6:  return 1.0
    if z < -6: return 0.0
    a1, a2, a3, a4, a5, p = (
        0.254829592, -0.284496736, 1.421413741,
        -1.453152027, 1.061405429, 0.3275911,
    )
    sign = 1 if z >= 0 else -1
    x = abs(z) / math.sqrt(2)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return 0.5 * (1.0 + sign * y)


def _compute_llr(pw: str) -> float:
    return sum(LLR_TABLE.get(c, 0.0) for c in pw)


def _compute_class_llr(pw: str) -> ClassLLR:
    cl = ClassLLR()
    for c in pw:
        cls = char_class(c)
        v = LLR_TABLE.get(c, 0.0)
        if cls == 'D':
            cl.digit_llr += v; cl.digit_n += 1
        elif cls in ('U', 'l'):
            cl.letter_llr += v; cl.letter_n += 1
        else:
            cl.symbol_llr += v; cl.symbol_n += 1
    return cl


def extract_features(pw: str) -> Features:
    """Extract all detection features from a password string."""
    n = len(pw)
    seq = [char_class(c) for c in pw]
    counts = Counter(seq)
    for k in 'UlDS':
        counts.setdefault(k, 0)

    same = sum(1 for i in range(n - 1) if seq[i] == seq[i + 1])
    sc = same / (n - 1)

    vals = [counts[k] for k in 'UlDS']
    A  = sum(k * (k - 1) for k in vals)
    B  = sum(k * (k - 1) * (k - 2) for k in vals)
    C  = sum(k * (k - 1) * (k - 2) * (k - 3) for k in vals)
    Dv = sum((k * (k - 1)) ** 2 for k in vals)

    es = A / (n * (n - 1))
    p_ov = B / (n * (n - 1) * (n - 2)) if n >= 3 else 0.0
    p_no = (C + A * A - Dv) / (n * (n - 1) * (n - 2) * (n - 3)) if n >= 4 else 0.0

    m = n - 1
    vs = (
        m * es * (1 - es)
        + 2 * (m - 1) * (p_ov - es * es)
        + (m - 1) * (m - 2) * (p_no - es * es)
    )
    sigma = math.sqrt(max(0, vs) / (m * m)) or 1e-9
    zs = (sc - es) / sigma

    llr = _compute_llr(pw)
    class_llr = _compute_class_llr(pw)

    hr = has_repeats(pw)
    rs = has_rare_symbols(pw)

    mr, cur = 1, 1
    for i in range(1, n):
        if seq[i] == seq[i - 1]:
            cur += 1
            mr = max(mr, cur)
        else:
            cur = 1

    nc = sum(1 for k in 'UlDS' if counts[k] > 0)

    return Features(
        n=n, sc=sc, es=es, zs=zs, sigma=sigma, llr=llr,
        class_llr=class_llr, hr=hr, rs=rs, mr=mr, nc=nc,
        counts=dict(counts),
    )


# ═══════════════════════════════════════════════════════════════
# PREFIX STRIPPING
# ═══════════════════════════════════════════════════════════════

def strip_prefix(pw: str) -> tuple[str, str | None]:
    """
    Check if pw starts with a known API key / token prefix.
    Returns (payload, prefix_label) or (pw, None) if no match.

    If stripping would leave the payload shorter than MIN_LENGTH,
    returns the original string with no prefix label.
    """
    for pattern, label in PREFIX_PATTERNS:
        if isinstance(pattern, re.Pattern):
            m = pattern.match(pw)
            if m:
                payload = pw[m.end():]
                if len(payload) >= MIN_LENGTH:
                    return (payload, label)
                else:
                    return (pw, None)
        else:
            if pw.startswith(pattern):
                payload = pw[len(pattern):]
                if len(payload) >= MIN_LENGTH:
                    return (payload, label)
                else:
                    return (pw, None)
    return (pw, None)


# ═══════════════════════════════════════════════════════════════
# ANALYSIS ENGINE (v5 beta)
# ═══════════════════════════════════════════════════════════════

def analyze(pw: str, *, no_strip: bool = False) -> Result:
    """
    Analyze a single password for signs of LLM generation.

    Returns a Result with verdict, signal_strength, detection path,
    features, and prefix information.

    Args:
        pw:       The credential to analyze.
        no_strip: If True, skip automatic prefix stripping and analyze the
                  full string as-is. Use this when the prefix is part of the
                  credential and should not be removed (e.g. when automatic
                  stripping falsely removes a meaningful character sequence).
    """
    n = len(pw)

    # Validation (before prefix stripping — validate the full input)
    for ch in pw:
        code = ord(ch)
        if code < 33 or code > 126:
            return Result(verdict=None, signals='', signal_strength=0.0,
                          reason='NON_ASCII', analyzed_portion=pw)
    if n < MIN_LENGTH:
        return Result(verdict=None, signals='', signal_strength=0.0,
                      reason='TOO_SHORT', analyzed_portion=pw)
    if n > MAX_LENGTH:
        return Result(verdict=None, signals='', signal_strength=0.0,
                      reason='TOO_LONG', analyzed_portion=pw)

    # Prefix stripping
    if no_strip:
        analyzed, prefix_label = pw, None
    else:
        analyzed, prefix_label = strip_prefix(pw)
    analyzed_n = len(analyzed)

    f = extract_features(analyzed)

    # Signal evaluation
    adaptive_sct_thr = sct_threshold(analyzed_n)
    sct_fires = f.sc < adaptive_sct_thr
    sct_zero  = f.sc == 0.0

    # SCT detection zone labeling:
    #   SCT = 0.000  → SCT₀ paths (unchanged)
    #   0 < SCT < 0.024 → SCT paths (unchanged, within global threshold)
    #   0.024 <= SCT < adaptive → SCT(len) paths (length-adaptive detection)
    sct_len_adaptive = (sct_fires and not sct_zero
                        and f.sc >= SCT_THRESHOLD)

    # LLR evaluation — shifted threshold for symbolless passwords
    if f.counts['S'] == 0 and f.nc >= 3:
        llr_thr = symbolless_llr_threshold(analyzed_n)
        llr_fires = f.llr > llr_thr
        symbolless_shift = (llr_thr != LLR_THRESHOLD)
    else:
        llr_fires = f.llr > LLR_THRESHOLD
        symbolless_shift = False

    # Soft indicators — modulate confidence within tiers, never across
    repeat_informative = analyzed_n <= 20
    soft_repeat = f.hr and repeat_informative
    soft_rare   = f.rs
    soft_count  = int(soft_repeat) + int(soft_rare)

    # ── Reduced-charset LLR bypass ──────────────────────────────
    reduced_charset = f.nc <= 2

    if reduced_charset:
        # LLR table is meaningless for <=2 class passwords.
        # SCT alone determines verdict.
        if sct_zero:
            sct_conf = 1.0 - _normal_cdf(f.zs)
            conf = min(CONFIDENCE_CAP, 0.88 + 0.10 * sct_conf)
            return Result(verdict='LLM_LIKELY',
                          signals='SCT₀ (reduced charset)',
                          signal_strength=conf, features=f,
                          prefix=prefix_label, analyzed_portion=analyzed)
        if sct_fires:
            sct_conf = 1.0 - _normal_cdf(f.zs)
            conf = min(CONFIDENCE_CAP, 0.82 + 0.15 * sct_conf)
            return Result(verdict='LLM_LIKELY',
                          signals='SCT (reduced charset)',
                          signal_strength=conf, features=f,
                          prefix=prefix_label, analyzed_portion=analyzed)
        # SCT doesn't fire — INCONCLUSIVE (no LLR to fall back on)
        return Result(verdict='INCONCLUSIVE',
                      signals='none (reduced charset)',
                      signal_strength=0.0, features=f,
                      prefix=prefix_label, analyzed_portion=analyzed)

    # ── Priority 1: SCT=0.000 at length ≥ 20 ─────────────────
    SCT0_LLR_PER_CHAR_BASE = -0.20

    if sct_zero and analyzed_n >= 20:
        llr_per_char = f.llr / analyzed_n

        if analyzed_n >= 50:
            llr_gate = None  # no gate — SCT=0 alone -> LIKELY
        elif analyzed_n >= 30:
            llr_gate = -0.40
        else:
            llr_gate = SCT0_LLR_PER_CHAR_BASE

        gate_passes = llr_gate is None or llr_per_char > llr_gate

        if gate_passes:
            sct_conf = 1.0 - _normal_cdf(f.zs)
            llr_bonus = 0.10 * min(1.0, f.llr / 20.0) if llr_fires else 0.0

            # Length-adaptive confidence for gateless path
            if llr_gate is None and not llr_fires:
                conf = min(CONFIDENCE_CAP, 0.92 + 0.05 * sct_conf)
            else:
                conf = min(CONFIDENCE_CAP, 0.85 + 0.10 * sct_conf + llr_bonus)

            sig = 'SCT₀+LLR' if llr_fires else 'SCT₀ (structural)'
            return Result(verdict='LLM_LIKELY', signals=sig,
                          signal_strength=conf, features=f,
                          prefix=prefix_label, analyzed_portion=analyzed)

        # SCT₀ but deeply negative per-char LLR (below the gate)
        sct_conf = 1.0 - _normal_cdf(f.zs)
        soft_penalty = soft_count * 0.03
        conf = min(0.80, 0.65 + 0.15 * sct_conf - soft_penalty)
        return Result(verdict='LLM_POSSIBLE', signals='SCT₀ (structural)',
                      signal_strength=conf, features=f,
                      prefix=prefix_label, analyzed_portion=analyzed)

    # ── Priority 2: Both SCT + LLR fire → LLM_LIKELY ─────────
    if sct_fires and llr_fires:
        sct_conf = 1.0 - _normal_cdf(f.zs)
        llr_norm = min(1.0, f.llr / 20.0) if not symbolless_shift else min(1.0, (f.llr - llr_thr) / 10.0)
        soft_penalty = soft_count * 0.04

        if analyzed_n >= 40:
            conf = min(CONFIDENCE_CAP, 0.78 + 0.12 * sct_conf + 0.08 * llr_norm - soft_penalty)
        else:
            conf = min(CONFIDENCE_CAP, 0.72 + 0.14 * sct_conf + 0.12 * llr_norm - soft_penalty)

        # Detection path label
        sct_part = 'SCT(len)' if sct_len_adaptive else 'SCT'
        llr_part = 'LLR'
        suffix = ' (no-sym)' if symbolless_shift else ''
        sig = f'{sct_part}+{llr_part}{suffix}'

        return Result(verdict='LLM_LIKELY', signals=sig,
                      signal_strength=conf, features=f,
                      prefix=prefix_label, analyzed_portion=analyzed)

    # ── Priority 3: SCT fires alone → LLM_POSSIBLE or LIKELY ─
    if sct_fires and not llr_fires:
        sct_conf = 1.0 - _normal_cdf(f.zs)
        soft_penalty = soft_count * 0.05

        sct_label = 'SCT(len)' if sct_len_adaptive else 'SCT'

        # Only promote when SCT is below the GLOBAL threshold (near-zero FPR
        # at 40+). Length-adaptive detections (SCT >= 0.024 but < adaptive
        # threshold) stay at POSSIBLE to avoid FPR regression.
        if analyzed_n >= 40 and not sct_len_adaptive:
            conf = min(CONFIDENCE_CAP, 0.82 + 0.12 * sct_conf - soft_penalty)
            return Result(verdict='LLM_LIKELY',
                          signals='SCT (length-promoted)',
                          signal_strength=conf, features=f,
                          prefix=prefix_label, analyzed_portion=analyzed)

        conf = min(0.80, 0.55 + 0.25 * sct_conf - soft_penalty)
        return Result(verdict='LLM_POSSIBLE', signals=f'{sct_label} only',
                      signal_strength=conf, features=f,
                      prefix=prefix_label, analyzed_portion=analyzed)

    # ── Priority 4: LLR fires alone → LLM_POSSIBLE ───────────
    if not sct_fires and llr_fires:
        if symbolless_shift:
            llr_norm = min(1.0, (f.llr - llr_thr) / 10.0)
        else:
            llr_norm = min(1.0, f.llr / 15.0)
        conf = min(0.75, 0.45 + 0.20 * llr_norm)
        suffix = ' (no-sym)' if symbolless_shift else ''
        return Result(verdict='LLM_POSSIBLE', signals=f'LLR only{suffix}',
                      signal_strength=conf, features=f,
                      prefix=prefix_label, analyzed_portion=analyzed)

    # ── Neither signal fires → INCONCLUSIVE ───────────────────
    return Result(verdict='INCONCLUSIVE', signals='none',
                  signal_strength=0.0, features=f,
                  prefix=prefix_label, analyzed_portion=analyzed)


def analyze_batch(passwords: list[str], *, no_strip: bool = False) -> list[Result]:
    """Analyze a list of passwords. Returns results in same order.

    Args:
        passwords: List of credential strings.
        no_strip:  If True, disable prefix stripping for all passwords in
                   the batch. Passed through to analyze().
    """
    return [analyze(pw, no_strip=no_strip) for pw in passwords]
