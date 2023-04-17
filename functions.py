import pandas as pd

from typing import Callable


class Function:
    _string: str = ""
    _func: Callable[[float], float] = lambda x: 0

    def __init__(self, s, f):
        self._string = s
        self._func = f

    def at(self, x: float) -> float:
        return self._func(x)

    def __str__(self):
        return "function: (" + self._string + ")"


class TableFunction:
    _table: pd.DataFrame = None     # | x | y |

    def __init__(self, t: pd.DataFrame):
        # assert t.columns[0] == 'x' and t.columns.values[1] == 'y', "Must contains only (x,y) cols"
        # assert all(t.notnull().values), "Must contains only non null values"
        # TODO: assertions
        self._table = t

    def table(self) -> pd.DataFrame:
        return self._table

    def x_values(self) -> pd.Series:
        return self._table['x']

    def y_values(self) -> pd.Series:
        return self._table['y']
