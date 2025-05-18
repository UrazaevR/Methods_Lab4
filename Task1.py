from typing import Callable
import numpy as np


def tipical_function_maker(a: float, b: float, c: float) -> Callable:
    def f(x: float, y: float) -> float:
        return a*x**2 + b*x*y + c*y**2
    return f


def derivative_maker(a: float, b: float, c: float) -> Callable:
    def f(x: float, y: float) -> tuple[float, float]:
        return 2*a*x + b*y, b*x + 2*c*y
    return f


def Hessian(a: float, b: float, c: float) -> list[list[float]]:
    matrix = [[2*a, b],
              [b, 2*c]]
    if 2*a*2*c - b*b == 0:
        raise Exception("Определитель Гессиана равен 0!")
    return matrix


def invert_Hessian(a: float, b: float, c: float) -> list[list[float]]:
    Hess = Hessian(a, b, c)
    det = 2*a*2*c - b*b
    answer = [[Hess[1][1] / det, -Hess[0][1] / det],
              [-Hess[1][0] / det, Hess[0][0] / det]]
    return answer


def deter(matrix: list[list[float]]) -> float:
    return matrix[0][0]*matrix[1][1] - matrix[1][0]*matrix[0][1]


def norma(x: list | tuple) -> float:
    ans = 0
    for elem in x:
        ans += elem**2
    return ans ** 0.5


def input_parameters():
    a, b, c = map(float, input("Введите коэффициенты a, b и c через пробел: ").split())
    x = tuple(map(float, input('Введите начальный вектор x (Числа через пробел): ').split()))
    e1 = float(input('Введите e1: '))
    e2 = float(input('Введите e2: '))
    M = int(input('Введите максимальное число итераций: '))
    return a, b, c, x, e1, e2, M


def half_division(f: callable, a: float, b: float, e: float, minimum: bool=True) -> tuple:
    k = 0

    xk = (a + b) / 2
    L = abs(b - a)

    while L > e:
        yk = a + L/4
        zk = b - L/4
        if minimum:
            if f(yk) < f(xk):
                b = xk
                xk = yk
            else:
                if f(zk) < f(xk):
                    a = xk
                    xk = zk
                else:
                    a = yk
                    b = zk
            L = abs(b - a)
            k += 1

        else:
            if f(yk) > f(xk):
                b = xk
                xk = yk
            else:
                if f(zk) > f(xk):
                    a = xk
                    xk = zk
                else:
                    a = yk
                    b = zk
            L = abs(b - a)
            k += 1
    return xk, f(xk), k


def Newton(a: float, b: float, c: float, x: tuple[float, float], e1: float, e2: float, M: int) -> tuple[tuple[float, float], float, int]:
    f = tipical_function_maker(a, b, c)
    df = derivative_maker(a, b, c)
    k = 0
    fl = False
    while k < M:
        if norma(df(*x)) <= e1:
            break
        inv = invert_Hessian(a, b, c)
        if deter(inv) > 0:
            min_H = [[-elem for elem in row] for row in inv]
            d = [min_H[0][0] * df(*x)[0] + min_H[0][1] * df(*x)[0], min_H[1][0] * df(*x)[1] + min_H[1][1] * df(*x)[1]]
            new_x = x[0] + d[0], x[1] + d[1]
        else:
            d = [-df(*x)[0], -df(*x)[1]]

            def fi(t: float) -> float:
                x1 = x[0] - t*df(*x)[0]
                x2 = x[1] - t*df(*x)[1]
                return f(x1, x2)

            tk = half_division(fi, -1, 1, 0.001)[0]
            new_x = x[0] - tk*d[0], x[1] - tk*d[1]
        if norma((new_x[0] - x[0], new_x[1] - x[1])) < e2 and abs(f(*new_x) - f(*x)) < e2:
            if fl:
                x = new_x
                break
            else:
                fl = True
        else:
            fl = False
        x = new_x
        k += 1
    return x, f(*x), k + 1


if __name__ == '__main__':
    x, f, k = Newton(*input_parameters())
    print(f"x = {x}")
    print(f"f(x) = {f}")
    print(f"Число итераций: {k}")
