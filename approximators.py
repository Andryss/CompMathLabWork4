import math

from functions import *
from modified_gauss import *


class VarianceCalculator:

    @staticmethod   # мера отклонения
    def squared_deviation(table_function: TableFunction, approximated_function: Function) -> float:
        squared_deviation = 0
        t = table_function.table().values
        for i in range(len(t)):
            squared_deviation += (approximated_function.at(t[i, 0]) - t[i, 1]) ** 2
        return squared_deviation

    @staticmethod   # среднеквадратическое отклонение
    def standard_deviation(table_function: TableFunction, approximated_function: Function) -> float:
        s = VarianceCalculator.squared_deviation(table_function, approximated_function)
        return VarianceCalculator.standard_deviation_from_squared_deviation(s, len(table_function.table()))

    @staticmethod
    def standard_deviation_from_squared_deviation(s: float, n: int) -> float:
        try:
            val = s / n
            return math.sqrt(val)
        except Exception as e:
            raise Exception(f"Can't calculate standard_deviation: {e.__str__()}")

    @staticmethod
    def squared_deviation_and_standard_deviation(table_function: TableFunction,
                                                 approximated_function: Function) -> [float, float]:
        s = VarianceCalculator.squared_deviation(table_function, approximated_function)
        omega = VarianceCalculator.standard_deviation_from_squared_deviation(s, len(table_function.table()))
        return [s, omega]


class Approximator:
    name: str = None

    def approximate(self, func: TableFunction) -> Function:
        raise Exception("method isn't overriden")


class _PolynomialSystemGenerator:

    @staticmethod
    def generate(table_function: TableFunction, n: int) -> np.ndarray:
        system = [[] for _ in range(n)]
        x_vals = table_function.x_values().copy()
        x = pd.Series([1] * len(x_vals))
        for bottom_row in range(-n + 1, n):
            for row in range(max(0, bottom_row), min(n, bottom_row + n)):   # window iteration
                system[row].append(x.sum())
            x *= x_vals
        y = table_function.y_values().copy()
        for row in range(n):
            system[row].append(y.sum())
            y *= x_vals
        return np.array(system)


class LinearApproximatorCoefficientResolver:

    @staticmethod
    def resolve(func: TableFunction) -> [float, float]:
        system = _PolynomialSystemGenerator.generate(func, 2)
        result = calculate_gauss_from_parameters(system)
        return result.answer[::-1]


class LinearApproximator(Approximator):
    name = "linear approximator"

    def approximate(self, func: TableFunction) -> Function:
        a, b = LinearApproximatorCoefficientResolver.resolve(func)
        return Function(f'{a} * x + {b}',
                        lambda x: a * x + b)


class SquareApproximatorCoefficientResolver:

    @staticmethod
    def resolve(func: TableFunction) -> [float, float]:
        system = _PolynomialSystemGenerator.generate(func, 3)
        result = calculate_gauss_from_parameters(system)
        return result.answer[::-1]


class SquarePolynomialApproximator(Approximator):
    name = "square polynomial approximator"

    def approximate(self, func: TableFunction) -> Function:
        a, b, c = SquareApproximatorCoefficientResolver.resolve(func)
        return Function(f'{a} * x^2 + {b} * x + {c}',
                        lambda x: a * x ** 2 + b * x + c)


class CubeApproximatorCoefficientResolver:

    @staticmethod
    def resolve(func: TableFunction) -> [float, float]:
        system = _PolynomialSystemGenerator.generate(func, 4)
        result = calculate_gauss_from_parameters(system)
        return result.answer[::-1]


class CubePolynomialApproximator(Approximator):
    name = "cube polynomial approximator"

    def approximate(self, func: TableFunction) -> Function:
        a, b, c, d = CubeApproximatorCoefficientResolver.resolve(func)
        return Function(f'{a} * x^3 + {b} * x^2 + {c} * x + {d}',
                        lambda x: a * x ** 3 + b * x ** 2 + c * x + d)


class PowerApproximatorCoefficientResolver:

    @staticmethod
    def resolve(func: TableFunction) -> [float, float]:
        assert (func.table() > 0).all().all(), "Can't resolve coefficients (both x and y must be positive)"
        new_table = pd.DataFrame()
        new_table['x'] = func.table()['x'].apply(lambda x: math.log(x))
        new_table['y'] = func.table()['y'].apply(lambda y: math.log(y))
        _b, _a = LinearApproximatorCoefficientResolver.resolve(TableFunction(new_table))
        return [math.exp(_a), _b]


class PowerApproximator(Approximator):
    name = "power approximator"

    def approximate(self, func: TableFunction) -> Function:
        a, b = PowerApproximatorCoefficientResolver.resolve(func)
        return Function(f'{a} * x^{b}',
                        lambda x: a * x ** b)


class ExponentialApproximatorCoefficientResolver:

    @staticmethod
    def resolve(func: TableFunction) -> [float, float]:
        assert (func.table()['y'] > 0).all(), "Can't resolve coefficients (y must be positive)"
        new_table = pd.DataFrame()
        new_table['x'] = func.table()['x']
        new_table['y'] = func.table()['y'].apply(lambda y: math.log(y))
        _b, _a = LinearApproximatorCoefficientResolver.resolve(TableFunction(new_table))
        return [math.exp(_a), _b]


class ExponentialApproximator(Approximator):
    name = "exponential approximator"

    def approximate(self, func: TableFunction) -> Function:
        a, b = ExponentialApproximatorCoefficientResolver.resolve(func)
        return Function(f'{a} * e^({b} * x)',
                        lambda x: a * math.exp(b * x))


class LogarithmicApproximatorCoefficientResolver:

    @staticmethod
    def resolve(func: TableFunction) -> [float, float]:
        assert (func.table()['x'] > 0).all(), "Can't resolve coefficients (x must be positive)"
        new_table = pd.DataFrame()
        new_table['x'] = func.table()['x'].apply(lambda x: math.log(x))
        new_table['y'] = func.table()['y']
        _a, _b = LinearApproximatorCoefficientResolver.resolve(TableFunction(new_table))
        return [_a, _b]


class LogarithmicApproximator(Approximator):
    name = "logarithmic approximator"

    def approximate(self, func: TableFunction) -> Function:
        a, b = LogarithmicApproximatorCoefficientResolver.resolve(func)
        return Function(f'{a} * ln(x) + {b}',
                        lambda x: a * math.log(x) + b)


def get_all_approximators() -> list[Approximator]:
    return [
        LinearApproximator(),
        SquarePolynomialApproximator(),
        CubePolynomialApproximator(),
        PowerApproximator(),
        ExponentialApproximator(),
        LogarithmicApproximator()
    ]


class ApproximationMetrics:
    squared_deviation: float
    standard_deviation: float

    def __init__(self, sq_d, st_d):
        self.squared_deviation = sq_d
        self.standard_deviation = st_d

    @staticmethod
    def from_approximated(table_function: TableFunction, approximated_function: Function):
        return ApproximationMetrics(
            *VarianceCalculator.squared_deviation_and_standard_deviation(table_function, approximated_function)
        )


class ApproximationResultEntity:
    approximator: Approximator
    approximated_function: Function
    metrics: ApproximationMetrics

    def __init__(self, app, app_f, met):
        self.approximator = app
        self.approximated_function = app_f
        self.metrics = met


class ApproximationResultEntityError(ApproximationResultEntity):
    error: Exception

    def __init__(self, app, err):
        super().__init__(app, None, None)
        self.approximator = app
        self.error = err


class ApproximationResult:
    source_function: TableFunction
    approximations: list[ApproximationResultEntity]

    def __init__(self, src, app):
        self.source_function = src
        self.approximations = app


def approximate(table_function: TableFunction) -> ApproximationResult:
    approximation_results = []
    for approximator in get_all_approximators():
        try:
            approximated_function = approximator.approximate(table_function)
            metrics = ApproximationMetrics.from_approximated(table_function, approximated_function)
            approximation_results.append(ApproximationResultEntity(approximator, approximated_function, metrics))
        except Exception as e:
            approximation_results.append(ApproximationResultEntityError(approximator, e))
    return ApproximationResult(table_function, approximation_results)
