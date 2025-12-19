"""
@file: frame_transformer.py
@author: Wu Maojia
@update: 2025.12.19
"""
from typing import Union

class ChildTree:
    def __init__(self):
        self.tree = {}

    def __getitem__(self, point: tuple) -> Union[set, None]:
        return self.tree.get(point)

    def add(self, point: tuple, child_point: tuple) -> None:
        if self.tree.get(point) is None:
            self.tree[point] = set()
        self.tree[point].add(child_point)

    def remove(self, point: tuple, child_point: tuple) -> None:
        if self.tree.get(point) is not None:
            self.tree[point].discard(child_point)