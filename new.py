import sympy as sp

def calculate_derivative(function_str, var):
    """
    Вычисляет производную заданной функции по указанной переменной.

    Аргументы:
    function_str: Строка, представляющая функцию (например, "x**2 + y*sin(x)").
    var: Строка, представляющая переменную, по которой нужно вычислить производную (например, "x").

    Возвращает:
    Функцию, которая вычисляет производную в заданных точках (x, y),
    или None, если произошла ошибка при разборе функции.
    """
    try:
        x, y = sp.symbols('x y')
        func = sp.sympify(function_str)  # Преобразуем строку в sympy-выражение
        derivative = sp.diff(func, var)  # Вычисляем производную

        # Преобразуем символьное выражение производной в функцию, которую можно вызывать.
        # Мы явно задаем lambda функцию для x и y, чтобы функция работала, даже если
        # в производной не осталось переменной, по которой брали производную
        derivative_func = sp.lambdify((x, y), derivative, 'numpy')

        return derivative_func

    except (SyntaxError, TypeError, ValueError) as e:
        print(f"Ошибка: Не удалось вычислить производную. Проверьте формат функции и переменной.\n{e}")
        return None

# Пример использования
if __name__ == '__main__':
    function_str = "x**2 + y*sin(x)"
    variable = "x"

    derivative_function = calculate_derivative(function_str, variable)

    if derivative_function:
        # Вычисляем значение производной в точке x=2, y=3
        x_value = 2
        y_value = 3
        derivative_value = derivative_function(x_value, y_value)
        print(f"Значение производной в точке x={x_value}, y={y_value}: {derivative_value}")

    function_str_2 = "y**3 + 5*x"
    variable_2 = "y"

    derivative_function_2 = calculate_derivative(function_str_2, variable_2)

    if derivative_function_2:
        # Вычисляем значение производной в точке x=1, y=4
        x_value_2 = 1
        y_value_2 = 4
        derivative_value_2 = derivative_function_2(x_value_2, y_value_2)
        print(f"Значение производной в точке x={x_value_2}, y={y_value_2}: {derivative_value_2}")

    # Пример ошибки
    function_str_err = "x**2 + y*sin(x"  # Отсутствует закрывающая скобка
    variable_err = "x"
    derivative_function_err = calculate_derivative(function_str_err, variable_err)
    print(f"Результат при ошибке: {derivative_function_err}")