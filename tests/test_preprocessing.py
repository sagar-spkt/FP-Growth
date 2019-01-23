# Sagar Sapkota 2014-2018
# FP-Growth Data Mining Frequent Pattern Algorithm
# Author: Sagar Sapkota <spkt.sagar@gmail.com>
#
# License: MIT License
import unittest

import numpy as np

from fpgrowth.preprocessing import TransactionEncoder


class TransactionEncoderTests(unittest.TestCase):
    """
    Tests for the Transactional Encoder
    """

    def setUp(self):
        """
        Build the Transactional Encoder and test its feature
        """
        self.dataset = [['Apple', 'Beer', 'Rice', 'Chicken'],
                        ['Apple', 'Beer', 'Rice'],
                        ['Apple', 'Beer'],
                        ['Apple', 'Bananas'],
                        ['Milk', 'Beer', 'Rice', 'Chicken'],
                        ['Milk', 'Beer', 'Rice'],
                        ['Milk', 'Beer'],
                        ['Apple', 'Bananas']]
        self.unique_item = sorted({'Bananas', 'Apple', 'Beer', 'Chicken', 'Milk', 'Rice'})
        self.expected_array = np.array([[1, 0, 1, 1, 0, 1],
                                        [1, 0, 1, 0, 0, 1],
                                        [1, 0, 1, 0, 0, 0],
                                        [1, 1, 0, 0, 0, 0],
                                        [0, 0, 1, 1, 1, 1],
                                        [0, 0, 1, 0, 1, 1],
                                        [0, 0, 1, 0, 1, 0],
                                        [1, 1, 0, 0, 0, 0]])

    def test_fit(self):
        """
        Test if unique column names from transaction DataFrame
        are learned
        """
        te = TransactionEncoder()
        te = te.fit(self.dataset)
        self.assertEqual(len(te.columns), len(self.unique_item),
                         msg="All items were not learned")
        self.assertListEqual(te.columns, self.unique_item,
                             msg="Learned items mismatched")

    def test_transform(self):
        """
        Test transactions encoded into a one-hot NumPy array
        """
        te = TransactionEncoder()
        tarray = te.fit(self.dataset).transform(self.dataset)
        np.testing.assert_array_equal(tarray, self.expected_array,
                                      err_msg="Encoding not matched")

    def test_inverse_transform(self):
        """
        Test if inverse transforms work or not
        """
        te = TransactionEncoder().fit(self.dataset)
        sorted_dataset = [sorted(arr) for arr in self.dataset]
        np.testing.assert_array_equal(np.array(sorted_dataset), np.array(te.inverse_transform(self.expected_array)),
                                      err_msg="Decoding not matched")

    def test_fit_transform(self):
        """
        Test if fit transform works or not
        """
        te = TransactionEncoder()
        np.testing.assert_array_equal(np.array(self.expected_array), np.array(te.fit_transform(self.dataset)))
