import pathlib
import unittest
from typing import List

from icecream import ic
from jack.tooling import io


def is_increasing(s: List[float]):
    return all(x < y for x, y in zip(s, s[1:]))


class IOTest(unittest.TestCase):
    def setUp(self):
        self.data_dir = pathlib.Path.home() / "Dataset"
        self.filename = "rc_set"
        self.sheet_name = 1
        self.text_col = "stt_op_text"
        self.label_col = "КП"

    def test_args(self):
        ic("hello")
        df = next(
            io.load(
                self.data_dir,
                self.filename,
                ext=".xlsx",
                rename_columns={self.text_col: "text", self.label_col: "label"},
                sheet_name=self.sheet_name,
            )
        )
        ic(df.head())
        self.assertTrue(df.shape == (5583, 2))


if __name__ == "__main__":
    unittest.main()
