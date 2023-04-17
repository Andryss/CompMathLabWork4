import math

from functions import *
from modified_gauss import *


class VarianceCalculator:

    @staticmethod   # мера отклонения
    def deviation_measure(table_function: TableFunction, approximated_function: Function) -> float:
        deviation_measure = 0
        t = table_function.table().values
        for i in range(len(t)):
            deviation_measure += (approximated_function.at(t[i, 0]) - t[i, 1]) ** 2
        return deviation_measure

    @staticmethod   # среднеквадратическое отклонение
    def standard_deviation(table_function: TableFunction, approximated_function: Function) -> float:
        s = VarianceCalculator.deviation_measure(table_function, approximated_function)
        return VarianceCalculator.standard_deviation_from_deviation_measure(s, len(table_function.table()))

    @staticmethod
    def standard_deviation_from_deviation_measure(s: float, n: int) -> float:
        return math.sqrt(s / n)

    @staticmethod
    def deviation_measure_and_standard_deviation(table_function: TableFunction,
                                                 approximated_function: Function) -> [float, float]:
        s = VarianceCalculator.deviation_measure(table_function, approximated_function)
        omega = VarianceCalculator.standard_deviation_from_deviation_measure(s, len(table_function.table()))
        return [s, omega]


class Approximator:
    name: str = None

    def approximate(self, func: TableFunction) -> Function:
        raise Exception("method isn't overriden")


class LinearApproximatorCoefficientResolver:

    @staticmethod
    def resolve(func: TableFunction) -> [float, float]:
        x, y = func.x_values(), func.y_values()
        n, sx, sy = len(func.table()), x.sum(), y.sum()
        x, y = x * func.x_values(), y * func.x_values()
        sxx, sxy = x.sum(), y.sum()
        system = np.array([
            [n,   sx,   sy],
            [sx,  sxx,  sxy]
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

    @staticmethod
    def resolve(func: TableFunction) -> [float, float]:
        x, y = func.x_values(), func.y_values()
        n, sx, sy = len(func.table()), x.sum(), y.sum()
        x, y = x * func.x_values(), y * func.x_values()
        sxx, sxy = x.sum(), y.sum()
        x, y = x * func.x_values(), y * func.x_values()
        sxxx, sxxy = x.sum(), y.sum()
        x = x * func.x_values()
        sxxxx = x.sum()
        system = np.array([
            [n,    sx,    sxx,    sy],
            [sx,   sxx,   sxxx,   sxy],
            [sxx,  sxxx,  sxxxx,  sxxy]
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

    @staticmethod
    def resolve(func: TableFunction) -> [float, float]:
        x, y = func.x_values(), func.y_values()
        n, sx, sy = len(func.table()), x.sum(), y.sum()
        x, y = x * func.x_values(), y * func.x_values()
        sxx, sxy = x.sum(), y.sum()
        x, y = x * func.x_values(), y * func.x_values()
        sxxx, sxxy = x.sum(), y.sum()
        x, y = x * func.x_values(), y * func.x_values()
        sxxxx, sxxxy = x.sum(), y.sum()
        x, y = x * func.x_values(), y * func.x_values()
        sxxxxx, sxxxxy = x.sum(), y.sum()
        x = x * func.x_values()
        sxxxxxx = x.sum()
        system = np.array([
            [n,     sx,     sxx,     sxxx,     sy],
            [sx,    sxx,    sxxx,    sxxxx,    sxy],
            [sxx,   sxxx,   sxxxx,   sxxxxx,   sxxy],
            [sxxx,  sxxxx,  sxxxxx,  sxxxxxx,  sxxxy]
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
        new_table = pd.DataFrame()
        new_table['x'] = func.table()['x']
        new_table['y'] = func.table()['y'].apply(lambda y: math.log(y))
        _b, _a = LinearApproximatorCoefficientResolver.resolve(TableFunction(new_table))
        return [math.exp(_a), _b]


class ExponentialApproximator(Approximator):
    name = "exponential approximator"

    def approximate(self, func: TableFunction) -> Function:
        a, b = ExponentialApproximatorCoefficientResolver.resolve(func)
        return Function(f'{a} * e^(b * x)',
                        lambda x: a * math.exp(b * x))


class LogarithmicApproximatorCoefficientResolver:

    @staticmethod
    def resolve(func: TableFunction) -> [float, float]:
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


if __name__ == "__main__":
    table_func = TableFunction(pd.DataFrame({'x': [1, 2, 3, 4], 'y': [1, 4, 9, 16]}))
    # TODO: add approximator assertions
    approximator = ExponentialApproximator()
    print(approximator.approximate(table_func))
