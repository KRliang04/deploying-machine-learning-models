import math

import numpy as np

from regression_model.predict import make_prediction

# sample_input_data is defined in conftest.py
def test_make_prediction(sample_input_data):
    # Given
    expected_first_prediction_value = 113422
    expected_no_predictions = 1449

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], np.float64)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    # Set up tolerance for first prediction value in the test.csv
    assert math.isclose(predictions[0], expected_first_prediction_value, abs_tol=100)
