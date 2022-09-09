from functools import lru_cache


class IState(object):
    @lru_cache
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        pass
