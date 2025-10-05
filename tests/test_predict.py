import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predict import predict

def test_predict():
    sample_input = {
        'AMC': 'aditya birla sun life mutual fund',
        'Morning star rating': 4,
        'Value Research rating': 5,
        '1 month return': 2.5,
        'NAV': 50.0,
        '1 Year return': 15.0,
        'Minimum investment': 5000.0,
        'AUM': 10000.0,
        'Category': 'Equity',
        'Risk': 'High'
    }
    try:
        result = predict(sample_input)
        assert isinstance(result, float)
        assert result > -100  # Reasonable for returns
    except Exception as e:
        pytest.fail(f"Prediction failed: {str(e)}")