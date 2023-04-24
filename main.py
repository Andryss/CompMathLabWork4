import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt

from approximators import *
from functions import *


def read_int_from_console(number_name: str) -> int:
    print(f"\nEnter {number_name}:")
    line = input().strip()
    try:
        return int(line)
    except Exception as e:
        raise Exception(f"Can't read int value: {e.__str__()}")


def read_point_from_console() -> [float, float]:
    try:
        args = input().strip().split()
        assert len(args) == 2, "point must have 2 float coordinates"
        return [float(args[0]), float(args[1])]
    except Exception as e:
        raise Exception(f"Can't read point from console: {e.__str__()}")


def read_table_function_from_console() -> TableFunction:
    points_count = read_int_from_console("points count (at least 5)")
    assert points_count >= 5, "Must be at least 5 points"
    print("\nEnter points coordinates sequentially (in each row like \"<x> <y>\"):")
    points = []
    for i in range(points_count):
        points.append(read_point_from_console())
    try:
        return TableFunction(pd.DataFrame(data=np.array(points), columns=['x', 'y']))
    except Exception as e:
        raise Exception(f"Can't read table function from console: {e.__str__()}")


def read_table_function_from_file() -> TableFunction:
    print("\nEnter the filename you want to read from:")
    filename = input().strip()
    try:
        values = pd.read_csv(filename, header=None).values
        table = pd.DataFrame(data=values, columns=['x', 'y'])
        assert len(table) >= 5, "Must be at least 5 points"
        return TableFunction(table)
    except Exception as e:
        raise Exception("file \"" + filename + "\" can't be opened: " + e.__str__())


def read_table_function() -> TableFunction:
    print("\nChoose the method you want to read the table:")
    print("0\tread from console")
    print("1\tread from file")
    line = input("(enter the number) ").strip()
    if line == "0":
        return read_table_function_from_console()
    elif line == "1":
        return read_table_function_from_file()
    else:
        raise Exception("No such option :(")


def read_bool(text: str) -> bool:
    line = input(f"{text} [Y/y for YES, NO otherwise]: ").strip().lower()
    if line == 'y':
        return True
    else:
        return False


def read_show_all() -> bool:
    return read_bool("\nDo you want to see all the plots?")


def print_source_function(result: ApproximationResult):
    print(f"\nSource function is: \n{result.source_function.table().T}")


def print_best_approximation(result: ApproximationResult):
    if result.best_approximation is not None:
        print("\nBest approximation is:")
        print_result_entity(result.best_approximation)
    else:
        print("\nNo best approximation found")


def show_approximation_plot(table_function: TableFunction, approximation_result: ApproximationResultEntity,
                            best_approximation_result: ApproximationResultEntity = None, number_of_points=10_000):
    if approximation_result.approximated_function is not None:
        warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

        plt.scatter(table_function.x_values(), table_function.y_values(), c='red', label='source points')

        x_func = np.linspace(table_function.x_values().min(), table_function.x_values().max(), number_of_points)

        if best_approximation_result is not None:
            y_func_best = []
            for x_val in x_func:
                y_func_best.append(best_approximation_result.approximated_function.at(x_val))
            plt.plot(x_func, y_func_best, c='lightblue', label=best_approximation_result.approximator.name)

        y_func = []
        for x_val in x_func:
            y_func.append(approximation_result.approximated_function.at(x_val))
        plt.plot(x_func, y_func, c='blue', label=approximation_result.approximator.name)

        plt.grid()
        plt.legend()
        plt.show()


def print_result_entity(result_entity: ApproximationResultEntity):
    print(f"\n{result_entity.__str__()}")


def print_result(result: ApproximationResult, verbose: bool = False):
    if result.source_function.table().shape[0] < 30:
        pd.set_option('display.expand_frame_repr', False)

    print("\nHere is approximation result:")
    print_source_function(result)
    print_best_approximation(result)
    show_approximation_plot(result.source_function, result.best_approximation)
    print("\nAll approximations:")
    for result_entity in result.approximations:
        print_result_entity(result_entity)

        if verbose:
            show_approximation_plot(result.source_function, result_entity, best_approximation_result=result.best_approximation)


def show_result(result: ApproximationResult, verbose: bool = False):
    print_result(result, verbose)


def run():
    try:
        table_function = read_table_function()
        show_all = read_show_all()
        approximation_result = approximate(table_function)
        show_result(approximation_result, verbose=show_all)
    except Exception as e:
        print(e, file=sys.stderr)


if __name__ == '__main__':
    run()
