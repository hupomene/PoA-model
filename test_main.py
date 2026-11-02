import unittest
from unittest.mock import patch
from main import app


class InputTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_invalid_number(self):
        response = self.client.get('/predict?CRED_SCORE_NUM=abc')
        self.assertEqual(response.status_code, 400)

    def test_infinite_number(self):
        response = self.client.get('/predict?CRED_SCORE_NUM=inf')
        self.assertEqual(response.status_code, 400)

    def test_prediction_response_with_stubbed_model(self):
        with patch.object(__import__('main').model, 'predict_proba', return_value=[[0.25, 0.75]]):
            response = self.client.get('/predict?CRED_SCORE_NUM=700&BANKRUPTCY=N')
        self.assertEqual(response.status_code, 200)
        self.assertAlmostEqual(response.get_json()['PoA'], 0.75)

    def test_invalid_category(self):
        response = self.client.get('/predict?BANKRUPTCY=invalid')
        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main()
