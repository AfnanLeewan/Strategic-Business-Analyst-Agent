"""
Predictive Analytics Service (Optional Module).
Provides sales forecasting using linear regression.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SalesPredictor:
    """Sales forecasting using linear regression."""
    
    def __init__(self):
        """Initialize the predictor."""
        self.model = LinearRegression()
        self.scaler = StandardScaler()
    
    def detect_sales_data(self, df: pd.DataFrame) -> bool:
        """
        Check if DataFrame contains sales data suitable for forecasting.
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if suitable data is found
        """
        # Look for date and sales columns (support various column names)
        date_keywords = ['date', 'datetime', 'time', 'orderdate', 'order_date']
        sales_keywords = ['sales', 'revenue', 'amount', 'total']
        
        has_date = any(any(kw in col.lower() for kw in date_keywords) for col in df.columns)
        has_sales = any(any(kw in col.lower() for kw in sales_keywords) for col in df.columns)
        
        return has_date and has_sales
    
    def prepare_data(self, df: pd.DataFrame) -> Optional[tuple]:
        """
        Prepare data for forecasting.
        
        Args:
            df: Input DataFrame with Date and Sales columns
            
        Returns:
            Tuple of (X, y, date_column, sales_column) or None
        """
        try:
            # Find date column (support various names)
            date_keywords = ['date', 'datetime', 'time', 'orderdate', 'order_date']
            date_col = None
            for col in df.columns:
                if any(kw in col.lower() for kw in date_keywords):
                    date_col = col
                    logger.info(f"Detected date column: '{col}'")
                    break
            
            # Find sales column (support various names)
            sales_keywords = ['sales', 'revenue', 'amount', 'total']
            sales_col = None
            for col in df.columns:
                if any(kw in col.lower() for kw in sales_keywords):
                    sales_col = col
                    logger.info(f"Detected sales column: '{col}'")
                    break
            
            if not date_col or not sales_col:
                logger.warning("Could not find date or sales columns")
                return None
            
            # Convert date column to datetime
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Remove rows with missing values
            df_clean = df[[date_col, sales_col]].dropna()
            
            if len(df_clean) < 3:
                logger.warning("Insufficient data for forecasting (need at least 3 data points)")
                return None
            
            # Sort by date
            df_clean = df_clean.sort_values(date_col)
            
            # Create numerical features from dates (days since first date)
            first_date = df_clean[date_col].min()
            df_clean['days_since_start'] = (df_clean[date_col] - first_date).dt.days
            
            X = df_clean[['days_since_start']].values
            y = df_clean[sales_col].values
            
            logger.info(f"Prepared {len(df_clean)} data points for forecasting")
            return X, y, date_col, sales_col, df_clean
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return None
    
    def forecast(self, df: pd.DataFrame, months_ahead: int = 3) -> Optional[Dict[str, Any]]:
        """
        Forecast sales for the next N months.
        
        Args:
            df: Input DataFrame
            months_ahead: Number of months to forecast
            
        Returns:
            Dictionary with forecast results or None
        """
        try:
            if not self.detect_sales_data(df):
                logger.info("No suitable sales data detected for forecasting")
                return None
            
            # Prepare data
            prep_result = self.prepare_data(df)
            if prep_result is None:
                return None
            
            X, y, date_col, sales_col, df_clean = prep_result
            
            # Train model
            self.model.fit(X, y)
            score = self.model.score(X, y)
            
            logger.info(f"Model trained with R² score: {score:.4f}")
            
            # Generate predictions for next N months
            last_date = df_clean[date_col].max()
            last_days = X[-1][0]
            
            predictions = []
            for month in range(1, months_ahead + 1):
                # Approximate 30 days per month
                future_days = last_days + (month * 30)
                pred_sales = self.model.predict([[future_days]])[0]
                pred_date = last_date + timedelta(days=month * 30)
                
                predictions.append({
                    "month": month,
                    "date": pred_date.strftime("%Y-%m-%d"),
                    "predicted_sales": float(pred_sales)
                })
            
            # Calculate statistics
            historical_mean = float(np.mean(y))
            historical_std = float(np.std(y))
            predicted_mean = float(np.mean([p["predicted_sales"] for p in predictions]))
            
            # Create summary text for LLM
            summary = self._create_forecast_summary(
                predictions, historical_mean, predicted_mean, score
            )
            
            return {
                "success": True,
                "predictions": predictions,
                "model_score": float(score),
                "historical_stats": {
                    "mean": historical_mean,
                    "std": historical_std,
                    "min": float(np.min(y)),
                    "max": float(np.max(y))
                },
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Error in forecasting: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_forecast_summary(
        self,
        predictions: list,
        historical_mean: float,
        predicted_mean: float,
        score: float
    ) -> str:
        """Create a text summary of the forecast for LLM integration."""
        summary_parts = [
            "=== SALES FORECAST ===",
            f"\nModel Performance: R² Score = {score:.4f}",
            f"Historical Average Sales: ${historical_mean:,.2f}",
            f"Predicted Average Sales (next {len(predictions)} months): ${predicted_mean:,.2f}",
            f"\nForecast Details:"
        ]
        
        for pred in predictions:
            summary_parts.append(
                f"- {pred['date']}: ${pred['predicted_sales']:,.2f}"
            )
        
        # Add trend analysis
        if predicted_mean > historical_mean:
            change_pct = ((predicted_mean - historical_mean) / historical_mean) * 100
            summary_parts.append(
                f"\nTrend: Upward trend expected (+{change_pct:.1f}% vs historical average)"
            )
        elif predicted_mean < historical_mean:
            change_pct = ((historical_mean - predicted_mean) / historical_mean) * 100
            summary_parts.append(
                f"\nTrend: Downward trend expected (-{change_pct:.1f}% vs historical average)"
            )
        else:
            summary_parts.append("\nTrend: Stable sales expected")
        
        return "\n".join(summary_parts)


# Global predictor instance
sales_predictor = SalesPredictor()
