# psr-mintrl

Is a backtest result distinguishable from luck, and if not, how many more trades does it need?

This library computes the **probabilistic Sharpe ratio (PSR)** and the **minimum track record length (MinTRL)** of a list of closed trades, as defined by Bailey and López de Prado (2012). Python and JavaScript, no dependencies, same results to 1e-15.

**Try it without code:** [PSR and MinTRL calculator](https://quantinvest.co/tools/psr-calculator/). Paste your trade results or four numbers from a MetaTrader 5 report. It runs the same formulas in your browser.

## Why per trade

A backtest is a sample of trades. The question is whether the average trade is above zero (or above a benchmark) by more than the sample could produce by chance. PSR answers it with the number of trades, their skewness and their fat tails, without annualizing anything. MinTRL turns it around: how many trades this Sharpe ratio needs to reach the chosen confidence.

## Python

```bash
pip install "git+https://github.com/quantinvestsystem/psr-mintrl#subdirectory=python"
```

```python
from psr_mintrl import analyze, from_report_averages

trades = [125.40, -62.10, -58.00, 210.75, ...]   # closed-trade results: money, percent or R
res = analyze(trades, sr_benchmark=0.0, confidence=0.95)

res.sr            # Sharpe ratio per trade, not annualized
res.psr           # probability that the true SR per trade is above the benchmark (None below 30 trades)
res.min_trl       # trades needed at this SR (None if SR <= benchmark)
res.more_trades   # how many more than you have
res.verdict       # "too_few" | "no_edge" | "inconclusive" | "distinguishable" | "cannot_assess"
```

## JavaScript

```js
const { analyze, fromReportAverages } = require("./js/psr-mintrl.js");  // or <script>: window.PsrMinTrl
const res = analyze([125.40, -62.10, -58.00, 210.75 /* ... */], 0, 0.95);
// res.sr, res.psr, res.minTrl, res.moreTrades, res.verdict
```

## Formulas

For `n` trade results with mean `m` and sample standard deviation `s` (ddof = 1):

- `SR = m / s`
- `G1`: unbiased skewness, `K`: unbiased, non-excess kurtosis (as `scipy.stats.skew(bias=False)` and `scipy.stats.kurtosis(fisher=False, bias=False)`)
- `D = 1 − G1·SR + (K − 1)/4 · SR²`
- `PSR(SR*) = Φ( (SR − SR*) · √(n − 1) / √D )`
- `MinTRL(SR*, p) = 1 + D · ( Φ⁻¹(p) / (SR − SR*) )²`

| Situation | Result |
|---|---|
| fewer than 30 trades | PSR not reported (`None` / `null`); MinTRL still computed from 4 trades |
| all trades equal | `cannot_assess`, reason `constant` |
| `D ≤ 0` | `cannot_assess`, reason `irregular` |
| `SR ≤ SR*` | PSR reported, MinTRL `None`: no number of trades is enough |
| NaN or infinite values | left out and counted in `skipped` |

## Four numbers from an MT5 report

If you only have the report summary, `from_report_averages(total_trades, profit_trades_pct, avg_profit, avg_loss)` (JS: `fromReportAverages`) builds a trade list in which every win equals the average win and every loss the average loss. Real trades vary more than that, so the Sharpe ratio comes out as an upper limit: **PSR is optimistic and MinTRL a lower bound.** Use the full trade list when you can.

Do not use the report's own "Sharpe Ratio" here. The Strategy Tester computes it on a different scale, not per trade.

## What it cannot tell you

The calculation trusts the trades it is given. It assumes one strategy tested once and trades independent of each other. The best of many optimization runs looks better than it is, and several positions open at once behave like one larger trade, so the real evidence is weaker than the count suggests. Wrong costs in the backtest change every number.

## Tests

`test-vectors.json` holds five cases (A–E) with expected values from an independent reference in NumPy and SciPy (`tools/make_vectors.py`). Both implementations must match them to 1e-8 (SR, skewness, kurtosis) and 1e-6 (PSR, MinTRL).

```bash
cd python && python -m unittest -v
cd js && npm test
```

An MQL5 version for MetaTrader 5 is being prepared for the MQL5 Code Base and will be checked against the same vectors.

## Reference

Bailey, D. H. and López de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. *Journal of Risk*, 15(2), 3–44.

## License

MIT. Maintained by [Quant Invest System](https://quantinvest.co/).
