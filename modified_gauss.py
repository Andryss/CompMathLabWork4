import numpy as np
from enum import Enum


class SolutionCount(Enum):
    ZERO = 1
    ONE = 2
    INFINITY = 3

    def __str__(self):
        return self.name


# Gaussian is already taken :(
class ResultContainer:
    original_matrix: np.ndarray = None
    solution_count: SolutionCount = None
    triangle_matrix: np.ndarray = None
    triangles: np.ndarray = None
    determinant: float = None
    answer: np.ndarray = None
    answer_residual: np.ndarray = None  # English(невязка) == residual


def calculate_gauss_from_parameters(array: np.ndarray) -> ResultContainer:
    return _calculate(array)


def _calculate(array: np.ndarray) -> ResultContainer:
    result = ResultContainer()
    result.original_matrix = array
    result.solution_count = _kronecker_capelli(array)
    if result.solution_count != SolutionCount.ONE:
        return result
    result.triangle_matrix, result.triangles, switches = _forward_path(array.astype(np.float64))
    result.determinant = _calculate_determinant(result.triangle_matrix, switches)
    result.answer = _backward_path(result.triangle_matrix)
    result.answer_residual = _calculate_residual(result.original_matrix, result.answer)
    return result


def _kronecker_capelli(array: np.ndarray) -> SolutionCount:
    a_matrix = array[:, :-1]
    a_extended = array
    rank_a = np.linalg.matrix_rank(a_matrix)
    rank_a_ext = np.linalg.matrix_rank(a_extended)
    # rank(A) != rank(A|B)
    if rank_a != rank_a_ext:
        return SolutionCount.ZERO
    n = max(len(array), len(array[0]) - 1)
    # rank(A) == rank(A|B) < n
    if rank_a < n:
        return SolutionCount.INFINITY
    # rank(A) == rank(A|B) == n
    else:
        return SolutionCount.ONE


def _forward_path(array: np.ndarray) -> [np.ndarray, np.ndarray, int]:
    n = len(array)
    triangles = []
    switches = 0
    for i in range(n-1):
        max_val = abs(array[i][i])
        max_row = i
        for j in range(i + 1, n):
            if abs(array[j][i]) > max_val:
                max_val = abs(array[j][i])
                max_row = j
        if i != max_row:
            array[i], array[max_row] = list(array[max_row]), list(array[i])
            switches += 1

        a_ii = array[i][i]
        for j in range(i + 1, n):
            a_ji = array[j][i]
            r = - a_ji / a_ii
            for k in range(i, n + 1):
                array[j][k] += array[i][k] * r

        triangles.append(array.copy())

    return [array, triangles, switches]


def _calculate_determinant(triangle_matrix: np.ndarray, switches: int) -> float:
    n = len(triangle_matrix)
    determinant = (-1.0) ** switches
    for i in range(n):
        determinant *= triangle_matrix[i][i]
    return determinant


def _backward_path(array: np.ndarray) -> np.ndarray:
    n = len(array)
    x = list(0.0 for _ in range(n))
    for i in range(n - 1, -1, -1):
        a_ii = array[i][i]
        b = array[i][n]
        for j in range(i + 1, n):
            b -= array[i][j] * x[j]
        x[i] = b / a_ii
    return np.array(x)


def _calculate_residual(matrix: np.ndarray, answer: np.ndarray) -> np.ndarray:
    residuals = list()
    n = len(matrix)
    for i in range(n):
        residual = -matrix[i][n]
        for j in range(n):
            residual += matrix[i][j] * answer[j]
        residuals.append(residual)
    return np.array(residuals)
