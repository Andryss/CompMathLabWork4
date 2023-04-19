import sys

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


def print_result_entity(result_entity: ApproximationResultEntity):
    print(f"\n{result_entity.approximator.name}:")
    if isinstance(result_entity, ApproximationResultEntityError):
        print(f"error: {result_entity.error.__str__()}")
    else:
        print(f"function: {result_entity.approximated_function.__str__()}")
        print(f"deviation: {result_entity.metrics.standard_deviation}")


def print_result(result: ApproximationResult):
    # pd.set_option('display.expand_frame_repr', False)

    print("\nHere is approximation result:")
    print(f"Source function was: \n{result.source_function.table().T}")
    for result_entity in result.approximations:
        print_result_entity(result_entity)


def show_result(result: ApproximationResult):
    print_result(result)


def run():
    try:
        table_function = read_table_function()
        # table_function = TableFunction(pd.DataFrame(data=pd.read_csv("table_1.csv", header=None).values, columns=['x', 'y']))
        print(table_function.table())
        approximation_result = approximate(table_function)
        show_result(approximation_result)
    except Exception as e:
        print(e, file=sys.stderr)


if __name__ == '__main__':
    run()
