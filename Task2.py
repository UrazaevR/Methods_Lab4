from typing import Callable
import numpy as np


def norma(x: list | tuple) -> float:
    ans = 0
    for elem in x:
        ans += elem**2
    return ans ** 0.5


def half_division(f: callable, a: float, b: float, e: float, minimum: bool = True) -> tuple:
    k = 0
    xk = (a + b) / 2
    L = abs(b - a)
    while L > e:
        yk = a + L / 4
        zk = b - L / 4
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


class Myfunction:
    def __init__(self, a: float, b: float, c: float):
        self.a = a
        self.b = b
        self.c = c
        self.pow = pow

    def derivative(self) -> Callable:
        return lambda x, y: (2 * self.a * x, 2 * self.b * y)

    def __call__(self, x: float, y: float) -> float:
        return self.a * x**2 + self.b * y**2 + self.c

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.a}, {self.b}, {self.c})'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.a}, {self.b}, {self.c})'


class MyIneq(Myfunction):
    def derivative(self) -> Callable:
        return lambda x, y: (self.a, self.b)

    def __call__(self, x: float, y: float) -> float:
        return self.a * x + self.b * y + self.c


class SumOfMyfunction:
    def __init__(self, mainf, *functions: list | tuple):
        self.mainf = mainf
        self.functions = functions

    def derivative(self):
        def dx(x: float, y: float) -> float:
            ans = self.mainf.derivative()(x, y)[0]
            for koef, func in self.functions:
                try:
                    ans += koef * (func.derivative()(x, y)[0] / -func(x, y))
                except ZeroDivisionError:
                    return float('inf')
            return ans

        def dy(x: float, y: float) -> float:
            ans = self.mainf.derivative()(x, y)[1]
            for koef, func in self.functions:
                try:
                    ans += koef * (func.derivative()(x, y)[1] / -func(x, y))
                except ZeroDivisionError:
                    return float('inf')
            return ans

        return lambda x, y: (dx(x, y), dy(x, y))

    def __call__(self, x: float, y: float):
        ans = self.mainf(x, y)
        for koef, func in self.functions:
            try:
                ans += koef * np.log(-func(x, y))
            except (ValueError, ZeroDivisionError):
                return float('inf')
        return ans


def fastest_gradient_method(f: Myfunction, x: tuple[float, float], e1: float, e2: float, M: int) -> tuple[tuple[float, float], float, int]:
    df = f.derivative()
    k = 0
    fl = False
    while k < M:
        if norma(df(*x)) < e1:
            break

        def fi(t: float) -> float:
            x1 = x[0] - t * df(*x)[0]
            x2 = x[1] - t * df(*x)[1]
            return f(x1, x2)

        tk = half_division(fi, -1, 1, 0.001)[0]
        new_x = x[0] - tk * df(*x)[0], x[1] - tk * df(*x)[1]
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


def gradient_method(f: Callable, x: tuple[float, float], e1: float, e2: float, M: int, t: float, minimum: bool = True) -> tuple[tuple[float, float], float, int]:
    df = f.derivative()
    k = 0
    fl = False
    while True:
        if norma(df(*x)) < e1:
            break
        if k >= M:
            break
        new_x = (x[0] - t * df(*x)[0], x[1] - t * df(*x)[1]) if minimum else (x[0] + t * df(*x)[0], x[1] + t * df(*x)[1])
        if not np.isfinite(f(*new_x)):
            t /= 2
            continue

        while (f(*new_x) - f(*x) >= 0 and minimum) or (f(*new_x) - f(*x) <= 0 and not minimum):
            t /= 2
            new_x = (x[0] - t * df(*x)[0], x[1] - t * df(*x)[1]) if minimum else (x[0] + t * df(*x)[0], x[1] + t * df(*x)[1])
            if t < 1e-10 or not np.isfinite(f(*new_x)):
                print("Warning: Step size became too small or infeasible. May not have converged.")
                return x, f(*x), k + 1
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


def barrier_method(f: Callable, ineq: list[Callable], x: list[float], r: float, C: float, e: float):
    k = 0
    while True:
        F = SumOfMyfunction(f, *((-r, g) for g in ineq))
        x, _, _ = gradient_method(F, x, e, e, 100, 0.001)
        k += 1
        if abs(sum(-r * np.log(-g(*x)) for g in ineq)) < e:
            return x, f(*x), k
        r /= C


if __name__ == '__main__':
    f = Myfunction(*map(float, input("Введите коэффициенты для основной функции через пробел (a b c): ").split()))
    print('Далее вводите коэффициенты для ограничений типа <=, также через пробел (a b c) (чтобы прекратить введите пустую строку)')
    ineq = []
    n = 1
    while (s := input(f'Коэффициенты для ограничения номер {n}: ').strip()) != '':
        ineq.append(MyIneq(*map(float, s.split())))
        n += 1
    x = list(map(float, input('Введите начальное значение x (через пробел): ').split()))

    r = float(input("Введите начальное значение r: "))
    C = float(input("Введите значение C (C > 1): "))
    e = float(input('Введите e: '))

    x, fx, k = barrier_method(f, ineq, x, r, C, e)
    print(f"x = {x}")
    print(f"f(x) = {fx}")
    print("Число итераций:", k)