"""
Ghost Feature Correlation Analyzer
===================================
Analyzes which features actually predict price movement.
Drops noise features that hurt accuracy.

Finds:
- Feature correlation with prediction correctness
- Redundant features
- Optimal feature weights
- Feature combinations that work
"""

import logging
import numpy as np
from typing import Any
from collections import defaultdict

LOGGER = logging.getLogger("ghost.feature_analyzer")


class FeatureAnalyzer:
    """Analyze feature importance and correlation"""
    
    def __init__(self):
        self.correlations = {}
        self.feature_impact = {}
        
    async def analyze_features(self) -> dict[str, Any]:
        """
        Analyze which features correlate with prediction accuracy.
        
        Returns:
            Feature analysis results with recommendations
        """
        LOGGER.info("Starting feature correlation analysis...")
        
        # Fetch reconciled predictions
        training_data = await self._fetch_training_data()
        
        if len(training_data) < 100:
            return {
                "ok": False,
                "error": f"Insufficient data: {len(training_data)} predictions (need 100+)",
                "predictions_found": len(training_data)
            }
        
        LOGGER.info(f"Analyzing {len(training_data)} predictions...")
        
        # Extract features from all predictions
        all_features = self._extract_all_features(training_data)
        
        if not all_features:
            return {
                "ok": False,
                "error": "No features found in predictions",
                "predictions_found": len(training_data)
            }
        
        # Calculate correlation for each feature
        feature_scores = {}
        for feature_name in all_features.keys():
            correlation = self._calculate_feature_correlation(
                training_data, feature_name
            )
            feature_scores[feature_name] = correlation
        
        # Sort by absolute correlation (highest impact first)
        sorted_features = sorted(
            feature_scores.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Categorize features
        strong_features = [(f, score) for f, score in sorted_features if abs(score) > 0.15]
        weak_features = [(f, score) for f, score in sorted_features if abs(score) < 0.05]
        
        LOGGER.info(f"✅ Analysis complete: {len(strong_features)} strong, {len(weak_features)} weak features")
        
        return {
            "ok": True,
            "total_predictions": len(training_data),
            "total_features": len(all_features),
            "strong_features": strong_features[:10],  # Top 10
            "weak_features": weak_features,
            "all_feature_scores": dict(sorted_features),
            "recommendations": self._generate_recommendations(sorted_features)
        }
    
    async def _fetch_training_data(self) -> list[dict[str, Any]]:
        """Fetch predictions with outcomes from PostgreSQL"""
        import os
        from core.prediction_store import get_prediction_store
        from sqlalchemy import text
        
        store = get_prediction_store()
        
        # Check if PostgreSQL
        is_postgres = os.getenv("DATABASE_URL", "").startswith("postgresql")
        if not is_postgres or not hasattr(store, 'engine'):
            LOGGER.warning("Not using PostgreSQL")
            return []
        
        query = text("""
            SELECT 
                symbol,
                features_json,
                was_correct,
                actual_price_change_pct
            FROM ghost_predictions
            WHERE actual_direction IS NOT NULL
              AND was_correct IS NOT NULL
              AND features_json IS NOT NULL
            ORDER BY run_at DESC
            LIMIT 5000
        """)
        
        with store.engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()
            
            training_data = []
            for row in rows:
                import json
                features = json.loads(row.features_json) if row.features_json else {}
                
                training_data.append({
                    "symbol": row.symbol,
                    "features": features,
                    "was_correct": row.was_correct,
                    "actual_price_change_pct": row.actual_price_change_pct
                })
            
            return training_data
    
    def _extract_all_features(self, data: list[dict]) -> dict[str, list]:
        """
        Extract all unique features across predictions.
        
        Returns:
            {feature_name: [values]}
        """
        all_features = defaultdict(list)
        
        for record in data:
            features = record.get("features", {})
            for feature_name, value in features.items():
                if isinstance(value, (int, float)):
                    all_features[feature_name].append(value)
        
        return dict(all_features)
    
    def _calculate_feature_correlation(
        self, data: list[dict], feature_name: str
    ) -> float:
        """
        Calculate Pearson correlation between feature value and prediction correctness.
        
        Returns:
            Correlation coefficient (-1 to 1)
            Positive = feature helps accuracy
            Negative = feature hurts accuracy
            Near zero = no correlation
        """
        feature_values = []
        correctness = []
        
        for record in data:
            features = record.get("features", {})
            if feature_name not in features:
                continue
            
            value = features[feature_name]
            if not isinstance(value, (int, float)):
                continue
            
            feature_values.append(value)
            correctness.append(1 if record["was_correct"] else 0)
        
        if len(feature_values) < 10:
            return 0.0
        
        # Calculate Pearson correlation
        feature_array = np.array(feature_values)
        correctness_array = np.array(correctness)
        
        # Normalize
        feature_mean = feature_array.mean()
        feature_std = feature_array.std()
        correctness_mean = correctness_array.mean()
        correctness_std = correctness_array.std()
        
        if feature_std == 0 or correctness_std == 0:
            return 0.0
        
        feature_norm = (feature_array - feature_mean) / feature_std
        correctness_norm = (correctness_array - correctness_mean) / correctness_std
        
        correlation = (feature_norm * correctness_norm).mean()
        
        return float(correlation)
    
    def _generate_recommendations(self, sorted_features: list[tuple]) -> dict[str, Any]:
        """Generate actionable recommendations based on analysis"""
        
        strong_positive = [f for f, score in sorted_features if score > 0.15]
        strong_negative = [f for f, score in sorted_features if score < -0.15]
        weak = [f for f, score in sorted_features if abs(score) < 0.05]
        
        recommendations = {
            "keep_features": strong_positive[:15],  # Top 15 helpful features
            "drop_features": weak + strong_negative,  # Noise + harmful features
            "feature_weights": {
                f: round(score, 3) 
                for f, score in sorted_features[:20]  # Top 20
            },
            "summary": f"Keep {len(strong_positive)} strong features, drop {len(weak)} weak features"
        }
        
        # Specific recommendations
        if "sentiment_score" in strong_positive:
            recommendations["note_sentiment"] = "✅ Sentiment is helping - keep CryptoPanic"
        elif "sentiment_score" in weak or "sentiment_score" in strong_negative:
            recommendations["note_sentiment"] = "❌ Sentiment not helping - consider removing CryptoPanic"
        
        if "rsi" in strong_positive:
            recommendations["note_rsi"] = "✅ RSI is helpful - technical indicators working"
        elif "rsi" in weak:
            recommendations["note_rsi"] = "⚠️ RSI not correlated - may need different timeframes"
        
        return recommendations


def get_feature_analyzer() -> FeatureAnalyzer:
    """Get feature analyzer instance"""
    return FeatureAnalyzer()
