import sys

import numpy as np

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
    points_count = read_int_from_console("points count")
    print("\nEnter points coordinates sequentially (in each row like \"<x> <y>\"):")
    points = []
    for i in range(points_count):
        points.append(read_point_from_console())
    try:
        return TableFunction(pd.DataFrame(data=np.array(points), columns=['x','y']))
    except Exception as e:
        raise Exception(f"Can't read table function from console: {e.__str__()}")


def read_table_function_from_file() -> TableFunction:
    print("\nEnter the filename you want to read from:")
    filename = input().strip()
    try:
        values = pd.read_csv(filename, header=None).values
        table = pd.DataFrame(data=values, columns=['x', 'y'])
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


def run():
    try:
        table_function = read_table_function()
        print(table_function.table())
    except Exception as e:
        print(e, file=sys.stderr)


if __name__ == '__main__':
    run()
