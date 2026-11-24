"""Shared test vectors (../../test-vectors.json, reference: NumPy + SciPy).

    cd python && python -m unittest -v
"""
import json
import math
import unittest
from pathlib import Path

import psr_mintrl as P

DOC = json.loads((Path(__file__).resolve().parents[2] / "test-vectors.json").read_text())
TOL = DOC["tolerance"]


class Vectors(unittest.TestCase):
    def close(self, got, want, tol, label):
        if want is None:
            self.assertIsNone(got, label)
        else:
            self.assertIsNotNone(got, label)
            self.assertLessEqual(abs(got - want), tol, f"{label}: {got} != {want}")

    def test_cases(self):
        for c in DOC["cases"]:
            with self.subTest(case=c["name"]):
                r, e = c["returns"], c["expected"]
                if "report_averages" in c:
                    a = c["report_averages"]
                    r = P.from_report_averages(a["total_trades"], a["profit_trades_pct"], a["avg_profit"], a["avg_loss"])
                    self.assertEqual(r, c["returns"])
                res = P.analyze(r, 0.0, 0.95)
                self.assertEqual(res.n, e["n"])
                self.close(res.sr, e["sr"], TOL["sr"], "sr")
                self.close(res.skewness, e["skewness"], TOL["skewness"], "skewness")
                self.close(res.kurtosis, e["kurtosis"], TOL["kurtosis"], "kurtosis")
                self.close(res.psr, e["psr_0"], TOL["psr"], "psr_0")
                self.close(P.psr(r, 0.05), e["psr_0.05"], TOL["psr"], "psr_0.05")
                self.close(res.min_trl, e["min_trl_0_0.95"], TOL["min_trl"], "min_trl 0.95")
                self.close(P.min_trl(r, 0.0, 0.90), e["min_trl_0_0.90"], TOL["min_trl"], "min_trl 0.90")
                self.assertEqual(res.more_trades, e["more_trades_0_0.95"])
                self.assertEqual(res.verdict, e["verdict_0_0.95"])


class EdgeCases(unittest.TestCase):
    def test_constant(self):
        res = P.analyze([5.0] * 40)
        self.assertEqual((res.verdict, res.reason), (P.CANNOT_ASSESS, "constant"))

    def test_non_finite_skipped(self):
        r = DOC["cases"][0]["returns"] + [math.nan, math.inf]
        self.assertEqual(P.analyze(r).skipped, 2)

    def test_too_few(self):
        self.assertEqual(P.analyze([1.0, 2.0]).verdict, P.TOO_FEW)

    def test_bad_confidence(self):
        with self.assertRaises(ValueError):
            P.analyze([1.0, -1.0, 2.0, 3.0], confidence=95)


if __name__ == "__main__":
    unittest.main()
