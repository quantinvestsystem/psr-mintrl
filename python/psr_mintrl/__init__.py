"""Probabilistic Sharpe Ratio and Minimum Track Record Length per trade.

Bailey, D. H. and López de Prado, M. (2012). The Sharpe Ratio Efficient Frontier.
Journal of Risk, 15(2).

No dependencies beyond the standard library. Online calculator with the same formulas:
https://quantinvest.co/tools/psr-calculator/
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable, List, Optional

__all__ = ["Result", "analyze", "psr", "min_trl", "from_report_averages", "MIN_TRADES"]
__version__ = "1.0.0"

MIN_TRADES = 30
"""Below this many trades PSR is not reported: the normal approximation behind it is too rough."""

_N = NormalDist()

# Verdicts
TOO_FEW = "too_few"                    # fewer than MIN_TRADES trades
NO_EDGE = "no_edge"                    # Sharpe ratio at or below the benchmark
INCONCLUSIVE = "inconclusive"          # positive edge, but PSR below the confidence level
DISTINGUISHABLE = "distinguishable"    # PSR at or above the confidence level
CANNOT_ASSESS = "cannot_assess"        # all trades equal, or the distribution makes D <= 0


@dataclass(frozen=True)
class Result:
    n: int                        # finite trade results used
    skipped: int                  # NaN / infinite values left out
    mean: Optional[float]
    sd: Optional[float]           # sample standard deviation, ddof = 1
    sr: Optional[float]           # Sharpe ratio per trade, not annualized
    skewness: Optional[float]     # unbiased (scipy.stats.skew, bias=False)
    kurtosis: Optional[float]     # unbiased, non-excess (scipy.stats.kurtosis, fisher=False, bias=False)
    denom: Optional[float]        # D = 1 - skewness*SR + (kurtosis - 1)/4 * SR^2
    psr: Optional[float]          # None below MIN_TRADES or when it cannot be assessed
    min_trl: Optional[float]      # None when SR <= benchmark (no length is enough)
    more_trades: Optional[int]    # max(0, ceil(min_trl) - n)
    verdict: str
    reason: Optional[str] = None  # "few", "constant" or "irregular"


def analyze(returns: Iterable[float], sr_benchmark: float = 0.0, confidence: float = 0.95) -> Result:
    """Full analysis of closed-trade results (money, percent or R; one value per trade)."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    raw = [float(x) for x in returns]
    r = [x for x in raw if math.isfinite(x)]
    n, skipped = len(r), len(raw) - len(r)
    empty = dict(n=n, skipped=skipped, mean=None, sd=None, sr=None, skewness=None, kurtosis=None,
                 denom=None, psr=None, min_trl=None, more_trades=None)
    if n < 4:
        return Result(**empty, verdict=TOO_FEW, reason="few")

    m = math.fsum(r) / n
    dev = [x - m for x in r]
    ss = math.fsum(e * e for e in dev)
    m2, m3, m4 = ss / n, math.fsum(e ** 3 for e in dev) / n, math.fsum(e ** 4 for e in dev) / n
    sd = math.sqrt(ss / (n - 1))
    if not sd > 1e-12 * max(1.0, abs(m)):  # differences at rounding level: all results equal
        return Result(**{**empty, "mean": m, "sd": sd}, verdict=CANNOT_ASSESS, reason="constant")

    sr = m / sd
    g1 = m3 / m2 ** 1.5 * math.sqrt(n * (n - 1)) / (n - 2)
    g2 = m4 / m2 ** 2 - 3.0
    k = ((n + 1) * g2 + 6.0) * (n - 1) / ((n - 2) * (n - 3)) + 3.0
    d = 1.0 - g1 * sr + (k - 1.0) / 4.0 * sr ** 2
    stats = dict(n=n, skipped=skipped, mean=m, sd=sd, sr=sr, skewness=g1, kurtosis=k, denom=d)
    if not d > 0:
        return Result(**stats, psr=None, min_trl=None, more_trades=None,
                      verdict=CANNOT_ASSESS, reason="irregular")

    p = _N.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(d)) if n >= MIN_TRADES else None
    mtrl = more = None
    if sr > sr_benchmark:
        mtrl = 1.0 + d * (_N.inv_cdf(confidence) / (sr - sr_benchmark)) ** 2
        more = max(0, math.ceil(mtrl) - n)

    if n < MIN_TRADES:
        verdict = TOO_FEW
    elif sr <= sr_benchmark:
        verdict = NO_EDGE
    else:
        verdict = DISTINGUISHABLE if p >= confidence else INCONCLUSIVE
    return Result(**stats, psr=p, min_trl=mtrl, more_trades=more, verdict=verdict)


def psr(returns: Iterable[float], sr_benchmark: float = 0.0) -> Optional[float]:
    """Probability that the true Sharpe ratio per trade exceeds the benchmark."""
    return analyze(returns, sr_benchmark).psr


def min_trl(returns: Iterable[float], sr_benchmark: float = 0.0, confidence: float = 0.95) -> Optional[float]:
    """Number of trades this Sharpe ratio needs to beat the benchmark at the given confidence."""
    return analyze(returns, sr_benchmark, confidence).min_trl


def from_report_averages(total_trades: int, profit_trades_pct: float,
                         avg_profit: float, avg_loss: float) -> List[float]:
    """Build trade results from four numbers of a MetaTrader 5 Strategy Tester report.

    Every win equals the average win and every loss the average loss, so the spread of
    results is understated: the Sharpe ratio is an upper limit, PSR is optimistic and
    MinTRL a lower bound. Use the full trade list when you have it.
    Do not use the report's own "Sharpe Ratio": it is on a different scale, not per trade.
    """
    n = int(round(total_trades))
    k = int(round(n * profit_trades_pct / 100.0))
    return [abs(avg_profit)] * k + [-abs(avg_loss)] * (n - k)
