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
        val = s / n
        assert val > 0, "Can't calculate standard deviation"
        return math.sqrt(val)

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


class LinearApproximatorCoefficientResolver:

    # noinspection DuplicatedCode
    @staticmethod
    def resolve(func: TableFunction) -> [float, float]:
        x, y = func.x_values(), func.y_values()
        n, sx1, sy1 = len(func.table()), x.sum(), y.sum()
        x, y = x * func.x_values(), y * func.x_values()
        sx2, sx1y1 = x.sum(), y.sum()
        system = np.array([
            [n,    sx1,  sy1],
            [sx1,  sx2,  sx1y1]
        ])
        result = calculate_gauss_from_parameters(system)
        return result.answer[::-1]


class LinearApproximator(Approximator):
    name = "linear approximator"

    def approximate(self, func: TableFunction) -> Function:
        a, b = LinearApproximatorCoefficientResolver.resolve(func)
        return Function(f'{a} * x + {b}',
                        lambda x: a * x + b)


class SquareApproximatorCoefficientResolver:

    # noinspection DuplicatedCode
    @staticmethod
    def resolve(func: TableFunction) -> [float, float]:
        x, y = func.x_values(), func.y_values()
        n, sx1, sy1 = len(func.table()), x.sum(), y.sum()
        x, y = x * func.x_values(), y * func.x_values()
        sx2, sx1y1 = x.sum(), y.sum()
        x, y = x * func.x_values(), y * func.x_values()
        sx3, sx2y1 = x.sum(), y.sum()
        x = x * func.x_values()
        sx4 = x.sum()
        system = np.array([
            [n,    sx1,  sx2,  sy1],
            [sx1,  sx2,  sx3,  sx1y1],
            [sx2,  sx3,  sx4,  sx2y1]
        ])
        result = calculate_gauss_from_parameters(system)
        return result.answer[::-1]


class SquarePolynomialApproximator(Approximator):
    name = "square polynomial approximator"

    def approximate(self, func: TableFunction) -> Function:
        a, b, c = SquareApproximatorCoefficientResolver.resolve(func)
        return Function(f'{a} * x^2 + {b} * x + {c}',
                        lambda x: a * x ** 2 + b * x + c)


class CubeApproximatorCoefficientResolver:

    # noinspection DuplicatedCode
    @staticmethod
    def resolve(func: TableFunction) -> [float, float]:
        x, y = func.x_values(), func.y_values()
        n, sx1, sy1 = len(func.table()), x.sum(), y.sum()
        x, y = x * func.x_values(), y * func.x_values()
        sx2, sx1y1 = x.sum(), y.sum()
        x, y = x * func.x_values(), y * func.x_values()
        sx3, sx2y1 = x.sum(), y.sum()
        x, y = x * func.x_values(), y * func.x_values()
        sx4, sx3y1 = x.sum(), y.sum()
        x, y = x * func.x_values(), y * func.x_values()
        sx5, sx4y1 = x.sum(), y.sum()
        x = x * func.x_values()
        sx6 = x.sum()
        system = np.array([
            [n,    sx1,  sx2,  sx3,  sy1],
            [sx1,  sx2,  sx3,  sx4,  sx1y1],
            [sx2,  sx3,  sx4,  sx5,  sx2y1],
            [sx3,  sx4,  sx5,  sx6,  sx3y1]
        ])
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
        assert all(func.table() > 0), "Can't resolve coefficients (both x and y must be positive)"
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
        assert all(func.table()['y'] > 0), "Can't resolve coefficients (y must be positive)"
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
        assert all(func.table()['x'] > 0), "Can't resolve coefficients (x must be positive)"
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
