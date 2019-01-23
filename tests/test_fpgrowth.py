# Sagar Sapkota 2014-2018
# FP-Growth Data Mining Frequent Pattern Algorithm
# Author: Sagar Sapkota <spkt.sagar@gmail.com>
#
# License: MIT License
import unittest

from fpgrowth.fpgrowth import FPNode


class FPNodeTests(unittest.TestCase):
    def setUp(self):
        self.node = FPNode('Apple', 2, None)
        self.node.add_child('Banana')
        self.node.add_child('Mango')

    def test_has_child(self):
        self.assertEqual(self.node.has_child('Banana'), True)
        self.assertEqual(self.node.has_child('Beer'), False)

    def test_get_child(self):
        self.assertEqual(self.node.get_child('Beer'), None)
        existed_child = self.node.get_child('Mango')
        self.assertIsInstance(existed_child, FPNode)
        self.assertEqual(existed_child.value, 'Mango')

    def test_add_child(self):
        self.assertEqual(self.node.get_child('Beer'), None)
        self.node.add_child('Beer')
        self.assertNotEqual(self.node.get_child('Beer'), None)

        self.assertEqual(self.node.get_child('Banana').count, 1)
        self.node.add_child('Banana')
        self.assertEqual(self.node.get_child('Banana').count, 2)
