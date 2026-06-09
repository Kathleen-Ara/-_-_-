import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_COUNT = 1000
MIN_VALUE = -10000
MAX_VALUE = 10000
ROUND_STEP = 100


def generate_initial_series(count, min_value, max_value, seed=None):
    """
    Формирует исходный набор данных Series.

    Данные генерируются с помощью генератора случайных чисел.
    Значения являются целыми числами из заданного диапазона.
    """
    generator = np.random.default_rng(seed)

    # Верхняя граница в numpy не включается, поэтому добавляется 1.
    values = generator.integers(min_value, max_value + 1, size=count)

    return pd.Series(values, name="Исходные значения")


def round_to_hundreds(value):
    """
    Округляет число до сотен по математическому правилу.

    Для положительных и отрицательных чисел используется округление
    до ближайшей сотни. Значения с остатком 50 округляются от нуля.
    """
    if value >= 0:
        return int(math.floor(value / ROUND_STEP + 0.5) * ROUND_STEP)

    return int(math.ceil(value / ROUND_STEP - 0.5) * ROUND_STEP)


def round_series_to_hundreds(series):
    """
    Формирует новый Series со значениями, округленными до сотен.
    """
    rounded_values = []

    for value in series:
        rounded_values.append(round_to_hundreds(value))

    return pd.Series(rounded_values, name="Округленные значения")


def calculate_series_characteristics(series):
    """
    Рассчитывает стандартные числовые характеристики Series.

    В функции не используются встроенные статистические методы Series
    и NumPy. Основные вычисления выполняются с помощью циклов.
    """
    values = list(series)

    min_value = values[0]
    max_value = values[0]
    total_sum = 0
    frequencies = {}

    for value in values:
        if value < min_value:
            min_value = value

        if value > max_value:
            max_value = value

        total_sum += value

        if value in frequencies:
            frequencies[value] += 1
        else:
            frequencies[value] = 1

    repeated_unique_count = 0
    repeated_extra_count = 0

    for value in frequencies:
        if frequencies[value] > 1:
            repeated_unique_count += 1
            repeated_extra_count += frequencies[value] - 1

    average = total_sum / len(values)
    square_deviation_sum = 0

    for value in values:
        square_deviation_sum += (value - average) ** 2

    standard_deviation = math.sqrt(square_deviation_sum / len(values))

    return {
        "Минимальное значение": min_value,
        "Количество различных повторяющихся значений": repeated_unique_count,
        "Количество повторов сверх первого появления": repeated_extra_count,
        "Максимальное значение": max_value,
        "Сумма чисел": total_sum,
        "Среднеквадратическое отклонение": standard_deviation,
    }


def print_first_series_rows(series, row_count=10):
    """
    Выводит в консоль первые строки набора Series.
    """
    print("Первые 10 строк набора Series:")
    print(series.head(row_count).to_string())
    print()


def print_characteristics(characteristics):
    """
    Выводит в консоль рассчитанные числовые характеристики.
    """
    print("Стандартные числовые характеристики набора Series:")

    for name, value in characteristics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    print()


def show_series_plots(series, rounded_series):
    """
    Строит линейный график и гистограмму по округленным значениям.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(series.index, series.values)
    plt.title("Линейный график исходного набора Series")
    plt.xlabel("Индекс элемента")
    plt.ylabel("Значение")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.hist(rounded_series.values, bins=30, edgecolor="black")
    plt.title("Гистограмма значений, округленных до сотен")
    plt.xlabel("Округленное значение")
    plt.ylabel("Частота")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def radix_sort_non_negative(values):
    """
    Выполняет поразрядную сортировку неотрицательных целых чисел.
    """
    if len(values) == 0:
        return []

    result = values[:]
    max_value = result[0]

    for value in result:
        if value > max_value:
            max_value = value

    digit_place = 1

    while max_value // digit_place > 0:
        buckets = []

        for _ in range(10):
            buckets.append([])

        for value in result:
            digit = value // digit_place % 10
            buckets[digit].append(value)

        result = []

        for bucket in buckets:
            for value in bucket:
                result.append(value)

        digit_place *= 10

    return result


def reverse_list(values):
    """
    Возвращает список в обратном порядке без использования sort.
    """
    result = []

    for index in range(len(values) - 1, -1, -1):
        result.append(values[index])

    return result


def radix_sort(values, descending=False):
    """
    Сортирует целые числа по возрастанию или убыванию.

    Для сортировки используется поразрядный алгоритм. Отрицательные
    числа обрабатываются отдельно через сортировку их модулей.
    """
    negative_abs_values = []
    positive_values = []

    for value in values:
        if value < 0:
            negative_abs_values.append(abs(value))
        else:
            positive_values.append(value)

    sorted_negative_abs = radix_sort_non_negative(negative_abs_values)
    sorted_positive = radix_sort_non_negative(positive_values)

    sorted_values = []

    # Для отрицательных чисел больший модуль означает меньшее значение.
    for value in reverse_list(sorted_negative_abs):
        sorted_values.append(-value)

    for value in sorted_positive:
        sorted_values.append(value)

    if descending:
        return reverse_list(sorted_values)

    return sorted_values


def create_analysis_dataframe(series):
    """
    Формирует DataFrame на основе исходного Series.

    В DataFrame добавляются столбцы с отсортированными значениями
    исходного набора по возрастанию и по убыванию.
    """
    source_values = list(series)

    ascending_values = radix_sort(source_values)
    descending_values = radix_sort(source_values, descending=True)

    dataframe = pd.DataFrame({
        "Исходные значения": source_values,
        "Сортировка по возрастанию": ascending_values,
        "Сортировка по убыванию": descending_values,
    })

    return dataframe


def print_first_dataframe_rows(dataframe, row_count=10):
    """
    Выводит в консоль первые строки сформированного DataFrame.
    """
    print("Первые 10 строк набора DataFrame:")
    print(dataframe.head(row_count).to_string(index=False))
    print()


def show_dataframe_plots(dataframe):
    """
    Строит два линейных графика для отсортированных значений.

    На одном полотне отображаются значения, отсортированные
    по возрастанию и по убыванию.
    """
    plt.figure(figsize=(12, 5))

    plt.plot(
        dataframe.index,
        dataframe["Сортировка по возрастанию"],
        label="По возрастанию",
    )

    plt.plot(
        dataframe.index,
        dataframe["Сортировка по убыванию"],
        label="По убыванию",
    )

    plt.title("Сравнение отсортированных значений")
    plt.xlabel("Индекс элемента")
    plt.ylabel("Значение")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    """
    Основная функция программы.

    Последовательно выполняет все пункты индивидуального задания:
    генерацию данных, расчет характеристик, визуализацию,
    формирование DataFrame и построение итоговых графиков.
    """
    # Для разных случайных наборов можно заменить seed=42 на seed=None.
    series = generate_initial_series(
        DATA_COUNT,
        MIN_VALUE,
        MAX_VALUE,
        seed=42,
    )

    print_first_series_rows(series)

    characteristics = calculate_series_characteristics(series)
    print_characteristics(characteristics)

    rounded_series = round_series_to_hundreds(series)
    show_series_plots(series, rounded_series)

    dataframe = create_analysis_dataframe(series)
    print_first_dataframe_rows(dataframe)

    show_dataframe_plots(dataframe)


if __name__ == "__main__":
    main()