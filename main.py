from typing import Callable
import numpy as np
from PyQt6.QtWidgets import QWidget, QApplication, QInputDialog, QListWidgetItem
from PyQt6 import QtCore
from window import Ui_Window
import sys


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
        return lambda x, y: (2 * self.a * x, 2 * self.b * y) # Corrected derivative

    def __call__(self, x: float, y: float) -> float:
        return self.a * x**2 + self.b * y**2 + self.c

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.a}, {self.b}, {self.c})'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.a}, {self.b}, {self.c})'


class MyEqual(Myfunction):
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
                ans += koef * func(x, y) * func.derivative()(x, y)[0]  # Corrected derivative calculation
            return ans

        def dy(x: float, y: float) -> float:
            ans = self.mainf.derivative()(x, y)[1]
            for koef, func in self.functions:
                ans += koef * func(x, y) * func.derivative()(x, y)[1]  # Corrected derivative calculation
            return ans

        return lambda x, y: (dx(x, y), dy(x, y))

    def __call__(self, x: float, y: float):
        return self.mainf(x, y) + sum(koef * func(x, y)**2 for koef, func in self.functions)


class SumOfLogMyfunction:
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
        # new_x = x[0] - t * df(*x)[0], x[1] - t * df(*x)[1]
        if minimum:
            new_x = (x[0] - t * df(*x)[0], x[1] - t * df(*x)[1])
        else:
            new_x = (x[0] + t * df(*x)[0], x[1] + t * df(*x)[1])
        while (f(*new_x) - f(*x) >= 0 and minimum) or (f(*new_x) - f(*x) <= 0 and not minimum):
            t /= 2
            # new_x = x[0] - t * df(*x)[0], x[1] - t * df(*x)[1]
            if minimum:
                new_x = (x[0] - t * df(*x)[0], x[1] - t * df(*x)[1])
            else:
                new_x = (x[0] + t * df(*x)[0], x[1] + t * df(*x)[1])
            if t < 1e-10:
                return x, f(*x), k+1
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


def fine_method(f: Callable, equal: list[Callable], x: list[float], r: float, C: float, e: float):
    k = 0

    def P(x: list[float], rk: float) -> float:
        ans = (rk / 2) * (sum((val(*x))**2 for val in equal))
        return ans

    while True:
        F = SumOfMyfunction(f, *((r / 2, g) for g in equal))
        x, _, _ = gradient_method(F, x, 0.015, 0.02, 100, 0.001)
        k += 1
        if P(x, r) <= e:
            return x, f(*x), k
        else:
            r *= C


def barrier_method(f: Callable, ineq: list[Callable], x: list[float], r: float, C: float, e: float):
    k = 0
    while True:
        F = SumOfLogMyfunction(f, *((-r, g) for g in ineq))
        x, _, _ = gradient_method(F, x, e, e, 100, 0.001)
        k += 1
        if abs(sum(-r * np.log(-g(*x)) for g in ineq)) < e:
            return x, f(*x), k
        r /= C


class Win(QWidget, Ui_Window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.result_button.clicked.connect(self.get_result)
        self.add_button.clicked.connect(self.new_equal)
        self.delete_button.clicked.connect(self.delete_item)

    def new_equal(self):
        koef, ok = QInputDialog.getText(self, 'Новое ограничение', 'Введите коэффициенты для ограничения через пробел')
        if ok:
            try:
                a, b, c = map(float, koef.split())
                form = f'{a if a != int(a) else int(a)}x {'+ ' if b > 0 else ''}{b if b != int(b) else int(b)}y {'+ ' if c > 0 else ''}{c if c != int(c) else int(c)}'
                new_item = QListWidgetItem(self.equals)
                new_item.setText(form)
                new_item.a = a
                new_item.b = b
                new_item.c = c
            except Exception as ex:
                self.output.setText(str(ex))

    def delete_item(self):
        if self.equals.currentItem() is not None:
            self.equals.takeItem(self.equals.currentRow())

    def get_result(self):
        try:
            if not self.koef_str.text().strip():
                raise Exception('Коэффициент c не должен быть пустым')
            if not self.x_str.text().strip():
                raise Exception('Необходимо задать начальный вектор Х')
            if not self.r_str.text().strip():
                raise Exception('Необходимо задать r')
            if not self.c_str.text().strip():
                raise Exception('Необходимо задать C')
            if not self.e_str.text().strip():
                raise Exception('Необходимо задать допустимую погрешность e')
            e = float(self.e_str.text())
            if e == 0:
                raise Exception('Погрешность не может быть равна нулю')
            r = float(self.r_str.text())
            C = float(self.c_str.text())
            x = list(map(float, self.x_str.text().split()))
            f = Myfunction(*map(float, self.koef_str.text().split()))
            sp = []
            for i in range(self.equals.count()):
                sp.append(MyEqual(self.equals.item(i).a, self.equals.item(i).b, self.equals.item(i).c))
            if self.method_box.currentText() == 'Метод штрафов':
                x, fx, k = fine_method(f, sp, x, r, C, e)
            else:
                x, fx, k = barrier_method(f, sp, x, r, C, e)
            self.output.setText(f'x* = {x}\nf(x*) = {fx}\nЧисло итераций: {k}')
        except Exception as ex:
            self.output.setText(str(ex))


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = Win()
    form.show()
    sys.excepthook = except_hook
    sys.exit(app.exec())