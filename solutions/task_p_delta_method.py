import pandas as pd
import numpy as np
from scipy.stats import t
import os

"""
ЗАДАЧА P: Дельта-метод.
ЦЕЛЬ: Проверить гипотезу равенства среднего чека (AOV), используя дельта-метод 
для оценки дисперсии метрики-отношения (Ratio-metric).
ДАННЫЕ: synthetic_gmv_data_1.2.csv
"""

# Настройка относительных путей
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, '..', 'data', 'synthetic_gmv_data_1.2.csv')

def calculate_delta_method_variance(x, y):
    """
    Вычисление дисперсии среднего отношения X/Y с помощью дельта-метода.
    Формула учитывает вариацию числителя, знаменателя и их ковариацию.
    """
    n = len(x)
    mu_x, mu_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    cov_xy = np.cov(x, y, ddof=1)[0, 1]

    # Реализация формулы дельта-метода для дисперсии отношения
    res_var = (var_x / mu_y**2 - 2 * mu_x * cov_xy / mu_y**3 + (mu_x**2) * var_y / mu_y**4) / n
    return res_var

try:
    # Загрузка данных
    df = pd.read_csv(file_path)

    # Агрегация данных до пользователя
    # gmv_sum (X) — числитель, trip_count (Y) — знаменатель для среднего чека (AOV)
    user_stats = df.groupby(['user_id', 'group_name']).agg(
        gmv_sum=('gmv', 'sum'),
        trip_count=('gmv', 'count')
    ).reset_index()

    # Разделение на группы
    control = user_stats[user_stats['group_name'] == 'control']
    test = user_stats[user_stats['group_name'] == 'test']

    # Расчет средних чеков (Ratio-metrics)
    # AOV = (Средний GMV на юзера) / (Среднее кол-во поездок на юзера)
    mean_check_ctrl = control['gmv_sum'].mean() / control['trip_count'].mean()
    mean_check_test = test['gmv_sum'].mean() / test['trip_count'].mean()

    # Оценка дисперсий через дельта-метод
    var_ctrl = calculate_delta_method_variance(control['gmv_sum'], control['trip_count'])
    var_test = calculate_delta_method_variance(test['gmv_sum'], test['trip_count'])

    # Расчет T-статистики
    t_stat = (mean_check_test - mean_check_ctrl) / np.sqrt(var_test + var_ctrl)

    # Расчет P-value с использованием распределения Стьюдента
    n_ctrl, n_test = len(control), len(test)
    dfree = n_ctrl + n_test - 2
    p_value = 2 * (1 - t.cdf(abs(t_stat), df=dfree))

    print(f"Результаты анализа:")
    print(f"T-статистика: {t_stat:.3f}")
    print(f"P-value: {p_value:.3f}")

except FileNotFoundError:
    print(f"Ошибка: Файл не найден по пути {file_path}")