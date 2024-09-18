import unittest
from perceptron import inner_product, perceptron_train, error
from preprocessing import create_vocabulary, create_feature_vectors

class TestPerceptron(unittest.TestCase):
    
    def test_inner_product(self):
        self.assertEqual(inner_product([1, 2], [2, 3]), 8)
        self.assertEqual(inner_product([0, 0], [0, 0]), 0)

    def test_create_vocabulary(self):
        vocab = create_vocabulary('sample_train.txt', thresh=2)
        self.assertIsInstance(vocab, list)

    def test_error(self):
        w = [1, 2]
        x = [[2, 3], [3, 4]]
        y = [1, -1]
        self.assertGreaterEqual(error(w, x, y), 0)

if __name__ == '__main__':
    unittest.main()
