import pandas as pd
import numpy as np
import scipy.stats as stats
import os

"""
ЗАДАЧА S: CUPED
ЦЕЛЬ: Повысить мощность теста путем снижения дисперсии метрики gmv_exp 
с использованием предпериодных данных gmv_hist.
ДАННЫЕ: synthetic_gmv_data_1.3.csv
"""

# Настройка путей
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, '..', 'data', 'synthetic_gmv_data_1.3.csv')

try:
    # Загрузка данных
    df = pd.read_csv(file_path)

    # Определяем переменные: X - исторические данные, Y - экспериментальные
    X = df['gmv_hist']
    Y = df['gmv_exp']
    group = df['group_name']

    # Расчет коэффициента theta
    # Используем все данные (тест + контроль) для оценки ковариации и дисперсии
    theta = np.cov(X, Y, ddof=1)[0, 1] / np.var(X, ddof=1)

    # CUPED-преобразование метрики
    # Снижаем зависимость Y от его исторического значения X
    Y_cuped = Y - theta * X

    # 4. Проверка гипотез (t-test Уэлча)
    # Группы для обычного теста
    control_exp = df[df['group_name'] == 'control']['gmv_exp']
    test_exp = df[df['group_name'] == 'test']['gmv_exp']

    # Группы для CUPED-теста
    control_cuped = Y_cuped[group == 'control']
    test_cuped = Y_cuped[group == 'test']

    # Расчет P-value (equal_var=False)
    _, pvalue_tt = stats.ttest_ind(test_exp, control_exp, equal_var=False)
    _, pvalue_cuped = stats.ttest_ind(test_cuped, control_cuped, equal_var=False)

    # Вывод результатов в вашем стиле
    print(f"P-value без CUPED: {pvalue_tt:.3f}")
    print(f"P-value с CUPED:  {pvalue_cuped:.3f}")

except FileNotFoundError:
    print(f"Ошибка: Файл не найден по пути {file_path}")
except Exception as e:
    print(f"Произошла ошибка: {e}")