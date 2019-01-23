# Sagar Sapkota 2014-2018
# FP-Growth Data Mining Frequent Pattern Algorithm
# Author: Sagar Sapkota <spkt.sagar@gmail.com>
#
# License: MIT License


class FPNode:
    """
    A Node in FP-Tree
    """

    def __init__(self, value, count, parent):
        self.value = value
        self.count = count
        self.parent = parent
        self.link = None
        self.children = []

    def has_child(self, value):
        children_value = [child.value for child in self.children]
        return value in children_value

    def get_child(self, value):
        for child_node in self.children:
            if child_node.value == value:
                return child_node
        return None

    def add_child(self, value):
        if self.has_child(value):
            existing_child = self.get_child(value)
            existing_child.count += 1
            return existing_child
        else:
            child = FPNode(value, 1, self)
            self.children.append(child)
            return child
