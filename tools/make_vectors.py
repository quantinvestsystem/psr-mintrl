"""Regenerate test-vectors.json with an independent reference: NumPy + SciPy.

    pip install numpy scipy
    python tools/make_vectors.py

The library itself has no dependencies; this script only produces the expected values
that the Python and JavaScript tests check against.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, norm, skew

ROOT = Path(__file__).resolve().parent.parent
MIN_TRADES = 30


def reference(r, sr_star, confidence):
    r = np.asarray(r, dtype=float)
    n = len(r)
    sr = float(np.mean(r) / np.std(r, ddof=1))
    g1 = float(skew(r, bias=False))
    k = float(kurtosis(r, fisher=False, bias=False))
    d = 1.0 - g1 * sr + (k - 1.0) / 4.0 * sr ** 2
    psr = float(norm.cdf((sr - sr_star) * math.sqrt(n - 1) / math.sqrt(d))) if n >= MIN_TRADES else None
    mintrl = float(1.0 + d * (norm.ppf(confidence) / (sr - sr_star)) ** 2) if sr > sr_star else None
    return {"n": n, "sr": sr, "skewness": g1, "kurtosis": k, "psr": psr, "min_trl": mintrl}


def report_averages(total, win_pct, avg_win, avg_loss):
    k = round(total * win_pct / 100)
    return [abs(avg_win)] * k + [-abs(avg_loss)] * (total - k)


def case(name, description, returns, verdict, source=None):
    base = reference(returns, 0.0, 0.95)
    mintrl = base["min_trl"]
    out = {
        "name": name,
        "description": description,
        "returns": returns,
        "expected": {
            "n": base["n"],
            "sr": base["sr"],
            "skewness": base["skewness"],
            "kurtosis": base["kurtosis"],
            "psr_0": base["psr"],
            "psr_0.05": reference(returns, 0.05, 0.95)["psr"],
            "min_trl_0_0.95": mintrl,
            "min_trl_0_0.90": reference(returns, 0.0, 0.90)["min_trl"],
            "more_trades_0_0.95": None if mintrl is None else max(0, math.ceil(mintrl) - base["n"]),
            "verdict_0_0.95": verdict,
        },
    }
    if source:
        out["report_averages"] = source
    return out


def main():
    v = json.loads((ROOT / "tools" / "inputs.json").read_text())
    e_src = {"total_trades": 354, "profit_trades_pct": 41.5, "avg_profit": 182.30, "avg_loss": -110.90}
    cases = [
        case("A", "60 trades, nearly symmetric, clear edge", v["A"], "distinguishable"),
        case("B", "40 trades, right-skewed: every fifth trade +0.03", v["B"], "distinguishable"),
        case("C", "50 trades, negative mean", v["C"], "no_edge"),
        case("D", "first 29 trades of A: too few trades, PSR not reported", v["A"][:29], "too_few"),
        case("E", "four numbers from an MT5 report (354 trades, 41.5% won, 182.30 / -110.90)",
             report_averages(354, 41.5, 182.30, -110.90), "inconclusive", e_src),
    ]
    doc = {
        "reference": f"numpy {np.__version__}, scipy {__import__('scipy').__version__}",
        "tolerance": {"sr": 1e-8, "skewness": 1e-8, "kurtosis": 1e-8, "psr": 1e-6, "min_trl": 1e-6},
        "cases": cases,
    }
    (ROOT / "test-vectors.json").write_text(json.dumps(doc, indent=1) + "\n")
    print("wrote test-vectors.json:", ", ".join(c["name"] for c in cases))


if __name__ == "__main__":
    main()
