/*!
 * psr-mintrl: Probabilistic Sharpe Ratio and Minimum Track Record Length per trade.
 * Bailey, D. H. and López de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. Journal of Risk, 15(2).
 * MIT License. Online calculator with the same formulas: https://quantinvest.co/tools/psr-calculator/
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.PsrMinTrl = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  var MIN_TRADES = 30;

  // Φ(x): West (2005), "Better approximations to cumulative normal functions"; error about 1e-15.
  function normCdf(x) {
    var z = Math.abs(x), c;
    if (z > 37) c = 0;
    else {
      var e = Math.exp(-z * z / 2);
      if (z < 7.07106781186547) {
        var n = ((((((3.52624965998911e-2 * z + 0.700383064443688) * z + 6.37396220353165) * z + 33.912866078383) * z
          + 112.079291497871) * z + 221.213596169931) * z + 220.206867912376);
        var d = (((((((8.83883476483184e-2 * z + 1.75566716318264) * z + 16.064177579207) * z + 86.7807322029461) * z
          + 296.564248779674) * z + 637.333633378831) * z + 793.826512519948) * z + 440.413735824752);
        c = e * n / d;
      } else {
        var b = z + 0.65; b = z + 4 / b; b = z + 3 / b; b = z + 2 / b; b = z + 1 / b;
        c = e / b / 2.506628274631;
      }
    }
    return x > 0 ? 1 - c : c;
  }

  // Φ⁻¹(p): Acklam's rational approximation refined with one Halley step: full double precision.
  function normInv(p) {
    if (!(p > 0 && p < 1)) throw new RangeError("p must be between 0 and 1");
    var a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
    var b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
    var c = [-7.784894002430293e-3, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    var d = [7.784695709041462e-3, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
    var q, r, x;
    if (p < 0.02425) {
      q = Math.sqrt(-2 * Math.log(p));
      x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    } else if (p > 1 - 0.02425) {
      q = Math.sqrt(-2 * Math.log(1 - p));
      x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    } else {
      q = p - 0.5; r = q * q;
      x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    }
    var err = normCdf(x) - p, u = err * Math.sqrt(2 * Math.PI) * Math.exp(x * x / 2);
    return x - u / (1 + x * u / 2);
  }

  /**
   * Full analysis of closed-trade results (money, percent or R; one value per trade).
   * verdict: "too_few" | "no_edge" | "inconclusive" | "distinguishable" | "cannot_assess"
   * reason:  "few" | "constant" | "irregular" | null
   */
  function analyze(values, srBenchmark, confidence) {
    srBenchmark = srBenchmark || 0;
    confidence = confidence === undefined ? 0.95 : confidence;
    if (!(confidence > 0 && confidence < 1)) throw new RangeError("confidence must be between 0 and 1");
    var r = [], skipped = 0, i;
    for (i = 0; i < values.length; i++) { var v = Number(values[i]); if (isFinite(v)) r.push(v); else skipped++; }
    var n = r.length, out = { n: n, skipped: skipped, mean: null, sd: null, sr: null, skewness: null, kurtosis: null,
      denom: null, psr: null, minTrl: null, moreTrades: null, verdict: "too_few", reason: null };
    if (n < 4) { out.reason = "few"; return out; }
    var m = 0; for (i = 0; i < n; i++) m += r[i]; m /= n;
    var m2 = 0, m3 = 0, m4 = 0;
    for (i = 0; i < n; i++) { var e = r[i] - m, e2 = e * e; m2 += e2; m3 += e2 * e; m4 += e2 * e2; }
    var ss = m2; m2 /= n; m3 /= n; m4 /= n;
    var sd = Math.sqrt(ss / (n - 1));
    out.mean = m; out.sd = sd;
    if (!(sd > 1e-12 * Math.max(1, Math.abs(m)))) { out.verdict = "cannot_assess"; out.reason = "constant"; return out; }
    var sr = m / sd;
    var g1 = m3 / Math.pow(m2, 1.5) * Math.sqrt(n * (n - 1)) / (n - 2);
    var g2 = m4 / (m2 * m2) - 3;
    var k = ((n + 1) * g2 + 6) * (n - 1) / ((n - 2) * (n - 3)) + 3;
    var D = 1 - g1 * sr + (k - 1) / 4 * sr * sr;
    out.sr = sr; out.skewness = g1; out.kurtosis = k; out.denom = D;
    if (!(D > 0)) { out.verdict = "cannot_assess"; out.reason = "irregular"; return out; }
    if (n >= MIN_TRADES) out.psr = normCdf((sr - srBenchmark) * Math.sqrt(n - 1) / Math.sqrt(D));
    if (sr > srBenchmark) {
      out.minTrl = 1 + D * Math.pow(normInv(confidence) / (sr - srBenchmark), 2);
      out.moreTrades = Math.max(0, Math.ceil(out.minTrl) - n);
    }
    if (n < MIN_TRADES) out.verdict = "too_few";
    else if (sr <= srBenchmark) out.verdict = "no_edge";
    else out.verdict = out.psr >= confidence ? "distinguishable" : "inconclusive";
    return out;
  }

  function psr(values, srBenchmark) { return analyze(values, srBenchmark).psr; }
  function minTrl(values, srBenchmark, confidence) { return analyze(values, srBenchmark, confidence).minTrl; }

  // Four numbers from an MT5 Strategy Tester report. Every win = average win, every loss = average loss,
  // so the spread is understated: SR is an upper limit, PSR optimistic, MinTRL a lower bound.
  function fromReportAverages(totalTrades, profitTradesPct, avgProfit, avgLoss) {
    var n = Math.round(totalTrades), k = Math.round(n * profitTradesPct / 100), v = [];
    for (var i = 0; i < n; i++) v.push(i < k ? Math.abs(avgProfit) : -Math.abs(avgLoss));
    return v;
  }

  return { analyze: analyze, psr: psr, minTrl: minTrl, fromReportAverages: fromReportAverages,
    normCdf: normCdf, normInv: normInv, MIN_TRADES: MIN_TRADES, version: "1.0.0" };
});
