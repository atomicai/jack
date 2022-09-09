import os
import pathlib
import unittest
from typing import List

from icecream import ic
from jack.recoiling.module import recoil


def is_increasing(s: List[float]):
    return all(x < y for x, y in zip(s, s[1:]))


dataset = [("Здравствуйте у меня такая проблема. Я бы ну очень хотела бы ну убрать все общие слова в общем ну вот.", "")]


class StopperTest(unittest.TestCase):
    def setUp(self):
        self.m = recoil.Stopper(pathlib.Path(os.getcwd()) / "jack" / "configuring" / "stopwords.txt")

    def test_dummy(self):
        for i, ex in enumerate(dataset):
            x, y = ex
            x = x.strip().lower()
            res = self.m(x.split(" "))
            ic(res)


if __name__ == "__main__":
    unittest.main()
