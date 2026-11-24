// Shared test vectors (../../test-vectors.json, reference: NumPy + SciPy).
//   node --test js/test/*.test.js
"use strict";
const test = require("node:test");
const assert = require("node:assert");
const path = require("node:path");
const P = require("../psr-mintrl.js");
const DOC = require(path.join(__dirname, "..", "..", "test-vectors.json"));
const TOL = DOC.tolerance;

function close(got, want, tol, label) {
  if (want === null) return assert.strictEqual(got, null, label);
  assert.ok(got !== null && Math.abs(got - want) <= tol, `${label}: ${got} != ${want}`);
}

for (const c of DOC.cases) {
  test(`vector ${c.name}: ${c.description}`, () => {
    let r = c.returns;
    const e = c.expected;
    if (c.report_averages) {
      const a = c.report_averages;
      r = P.fromReportAverages(a.total_trades, a.profit_trades_pct, a.avg_profit, a.avg_loss);
      assert.deepStrictEqual(r, c.returns);
    }
    const res = P.analyze(r, 0, 0.95);
    assert.strictEqual(res.n, e.n);
    close(res.sr, e.sr, TOL.sr, "sr");
    close(res.skewness, e.skewness, TOL.skewness, "skewness");
    close(res.kurtosis, e.kurtosis, TOL.kurtosis, "kurtosis");
    close(res.psr, e.psr_0, TOL.psr, "psr_0");
    close(P.psr(r, 0.05), e["psr_0.05"], TOL.psr, "psr_0.05");
    close(res.minTrl, e["min_trl_0_0.95"], TOL.min_trl, "min_trl 0.95");
    close(P.minTrl(r, 0, 0.90), e["min_trl_0_0.90"], TOL.min_trl, "min_trl 0.90");
    assert.strictEqual(res.moreTrades, e["more_trades_0_0.95"]);
    assert.strictEqual(res.verdict, e["verdict_0_0.95"]);
  });
}

test("edge cases", () => {
  assert.strictEqual(P.analyze(Array(40).fill(5)).reason, "constant");
  assert.strictEqual(P.analyze(DOC.cases[0].returns.concat([NaN, Infinity])).skipped, 2);
  assert.strictEqual(P.analyze([1, 2]).verdict, "too_few");
  assert.throws(() => P.analyze([1, -1, 2, 3], 0, 95), RangeError);
});

test("normal distribution", () => {
  close(P.normCdf(1.959963984540054), 0.975, 1e-12, "cdf");
  close(P.normInv(0.95), 1.6448536269514722, 1e-12, "ppf 0.95");
  close(P.normInv(0.001), -3.090232306167813, 1e-10, "ppf 0.001");
});
