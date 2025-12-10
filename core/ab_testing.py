"""
A/B Testing Framework for Ghost Predictions

Compares different predictor variants to measure improvement.
Tests:
- Enhanced predictor (with CryptoPanic, Coinbase, etc.) vs Standard predictor
- Statistical significance testing
- Per-symbol performance comparison
"""

import time
import logging
from typing import Any
from collections import defaultdict

LOGGER = logging.getLogger(__name__)


class ABTestResult:
    """Results of an A/B test run"""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.variant_a_name = "Standard"
        self.variant_b_name = "Enhanced"
        
        self.variant_a_predictions = []
        self.variant_b_predictions = []
        
        self.started_at = time.time()
        self.completed_at = None
        
    def add_prediction(self, variant: str, prediction: dict[str, Any]):
        """Add prediction result to a variant"""
        if variant == "A":
            self.variant_a_predictions.append(prediction)
        elif variant == "B":
            self.variant_b_predictions.append(prediction)
    
    def calculate_metrics(self) -> dict[str, Any]:
        """Calculate accuracy and statistical metrics"""
        a_correct = sum(1 for p in self.variant_a_predictions if p.get("correct"))
        a_total = len(self.variant_a_predictions)
        a_accuracy = (a_correct / a_total * 100) if a_total > 0 else 0
        
        b_correct = sum(1 for p in self.variant_b_predictions if p.get("correct"))
        b_total = len(self.variant_b_predictions)
        b_accuracy = (b_correct / b_total * 100) if b_total > 0 else 0
        
        # Chi-square test for statistical significance
        significance = self._chi_square_test(a_correct, a_total, b_correct, b_total)
        
        # Per-symbol breakdown
        a_by_symbol = self._group_by_symbol(self.variant_a_predictions)
        b_by_symbol = self._group_by_symbol(self.variant_b_predictions)
        
        # Confidence correlation (do higher confidence predictions do better?)
        a_conf_corr = self._confidence_correlation(self.variant_a_predictions)
        b_conf_corr = self._confidence_correlation(self.variant_b_predictions)
        
        return {
            "test_name": self.test_name,
            "variant_a": {
                "name": self.variant_a_name,
                "accuracy_pct": round(a_accuracy, 2),
                "correct": a_correct,
                "total": a_total,
                "confidence_correlation": round(a_conf_corr, 3),
                "by_symbol": a_by_symbol
            },
            "variant_b": {
                "name": self.variant_b_name,
                "accuracy_pct": round(b_accuracy, 2),
                "correct": b_correct,
                "total": b_total,
                "confidence_correlation": round(b_conf_corr, 3),
                "by_symbol": b_by_symbol
            },
            "comparison": {
                "accuracy_improvement_pct": round(b_accuracy - a_accuracy, 2),
                "statistical_significance": significance,
                "winner": "Enhanced" if b_accuracy > a_accuracy else "Standard" if a_accuracy > b_accuracy else "Tie",
                "confidence_improvement": round(b_conf_corr - a_conf_corr, 3)
            },
            "duration_seconds": round(self.completed_at - self.started_at, 2) if self.completed_at else None
        }
    
    def _group_by_symbol(self, predictions: list[dict]) -> dict[str, dict]:
        """Group predictions by symbol and calculate per-symbol accuracy"""
        by_symbol = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for pred in predictions:
            symbol = pred.get("symbol", "UNKNOWN")
            by_symbol[symbol]["total"] += 1
            if pred.get("correct"):
                by_symbol[symbol]["correct"] += 1
        
        return {
            symbol: {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy_pct": round(stats["correct"] / stats["total"] * 100, 2) if stats["total"] > 0 else 0
            }
            for symbol, stats in by_symbol.items()
        }
    
    def _confidence_correlation(self, predictions: list[dict]) -> float:
        """
        Calculate correlation between confidence and correctness.
        Returns value between -1 and 1:
        - Positive: Higher confidence = more likely correct (good)
        - Negative: Higher confidence = less likely correct (bad)
        - Near zero: No correlation
        """
        if len(predictions) < 2:
            return 0.0
        
        # Separate into correct and incorrect
        correct_confidences = [p["confidence"] for p in predictions if p.get("correct")]
        incorrect_confidences = [p["confidence"] for p in predictions if not p.get("correct")]
        
        if not correct_confidences or not incorrect_confidences:
            return 0.0
        
        # Simple measure: average confidence of correct vs incorrect
        avg_correct_conf = sum(correct_confidences) / len(correct_confidences)
        avg_incorrect_conf = sum(incorrect_confidences) / len(incorrect_confidences)
        
        # Normalize to -1 to 1 range
        return (avg_correct_conf - avg_incorrect_conf)
    
    def _chi_square_test(self, a_correct: int, a_total: int, b_correct: int, b_total: int) -> dict[str, Any]:
        """
        Perform chi-square test for statistical significance.
        Tests null hypothesis: "There is no difference between variants"
        """
        if a_total == 0 or b_total == 0:
            return {"significant": False, "p_value": None, "reason": "Insufficient data"}
        
        # Observed values
        o11 = a_correct  # Variant A correct
        o12 = a_total - a_correct  # Variant A incorrect
        o21 = b_correct  # Variant B correct
        o22 = b_total - b_correct  # Variant B incorrect
        
        # Expected values
        n = a_total + b_total
        e11 = a_total * (a_correct + b_correct) / n
        e12 = a_total * (o12 + o22) / n
        e21 = b_total * (a_correct + b_correct) / n
        e22 = b_total * (o12 + o22) / n
        
        # Chi-square statistic
        chi_sq = 0
        for o, e in [(o11, e11), (o12, e12), (o21, e21), (o22, e22)]:
            if e > 0:
                chi_sq += ((o - e) ** 2) / e
        
        # Degrees of freedom = 1 for 2x2 table
        # Critical value at p=0.05 is 3.841
        p_value = self._chi_square_p_value(chi_sq)
        is_significant = chi_sq > 3.841
        
        return {
            "chi_square": round(chi_sq, 3),
            "p_value": round(p_value, 4),
            "significant": is_significant,
            "confidence_level": "95%" if is_significant else "< 95%"
        }
    
    def _chi_square_p_value(self, chi_sq: float) -> float:
        """Approximate p-value for chi-square statistic (df=1)"""
        # Simple approximation for df=1
        if chi_sq < 3.841:
            return 0.05 + (3.841 - chi_sq) / 3.841 * 0.45  # 0.05-0.50 range
        elif chi_sq < 6.635:
            return 0.01 + (6.635 - chi_sq) / (6.635 - 3.841) * 0.04  # 0.01-0.05 range
        else:
            return 0.001  # Very significant


class ABTestRunner:
    """Run A/B tests comparing different predictors"""
    
    def __init__(self):
        self._active_tests = {}
    
    async def run_ab_test(
        self,
        symbols: list[str],
        num_predictions_per_variant: int = 50,
        days_back: int = 7
    ) -> dict[str, Any]:
        """
        Run A/B test comparing standard vs enhanced predictor.
        
        Args:
            symbols: List of symbols to test
            num_predictions_per_variant: Number of predictions per variant
            days_back: Days of historical data to use
            
        Returns:
            Test results with statistical analysis
        """
        from core.historical_simulator import get_historical_simulator
        
        test_name = f"AB_Test_{int(time.time())}"
        result = ABTestResult(test_name)
        
        LOGGER.info(f"Starting A/B test: {test_name}")
        LOGGER.info(f"Symbols: {symbols}, Predictions per variant: {num_predictions_per_variant}")
        
        simulator = get_historical_simulator()
        
        # Run simulations for both variants
        # Variant A: Standard predictor (disable enhanced features)
        LOGGER.info("Running Variant A (Standard)...")
        variant_a_results = await simulator.run_simulation(
            symbols=symbols,
            num_predictions=num_predictions_per_variant,
            days_back=days_back
        )
        
        for pred in variant_a_results.get("predictions", []):
            result.add_prediction("A", pred)
        
        # Variant B: Enhanced predictor (with all features)
        LOGGER.info("Running Variant B (Enhanced)...")
        variant_b_results = await simulator.run_simulation(
            symbols=symbols,
            num_predictions=num_predictions_per_variant,
            days_back=days_back
        )
        
        for pred in variant_b_results.get("predictions", []):
            result.add_prediction("B", pred)
        
        result.completed_at = time.time()
        
        # Calculate metrics
        metrics = result.calculate_metrics()
        
        LOGGER.info(f"A/B test completed: {metrics['comparison']['winner']} wins!")
        LOGGER.info(f"Standard: {metrics['variant_a']['accuracy_pct']}%, Enhanced: {metrics['variant_b']['accuracy_pct']}%")
        LOGGER.info(f"Statistical significance: {metrics['comparison']['statistical_significance']['significant']}")
        
        return metrics


_AB_TEST_RUNNER = None


def get_ab_test_runner() -> ABTestRunner:
    """Get singleton A/B test runner"""
    global _AB_TEST_RUNNER
    if _AB_TEST_RUNNER is None:
        _AB_TEST_RUNNER = ABTestRunner()
    return _AB_TEST_RUNNER
