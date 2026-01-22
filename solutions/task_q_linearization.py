import numpy as np
import scipy.stats as stats
import pandas as pd
import os

"""
ЗАДАЧА Q: Линеаризация второго типа.
ЦЕЛЬ: Проверить гипотезу равенства среднего чека (AOV), используя линеаризованную метрику 
для повышения чувствительности и применения классического t-test.
ДАННЫЕ: synthetic_gmv_data_1.2.csv
"""

# Настройка путей
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, '..', 'data', 'synthetic_gmv_data_1.2.csv')

def perform_linearization_test(test_num, test_denom, ctrl_num, ctrl_denom):
    """
    Применяет линеаризацию второго типа и проводит t-test.
    """
    # Расчет средних для каждой группы
    mu_x_num, mu_x_denom = np.mean(test_num), np.mean(test_denom)
    mu_y_num, mu_y_denom = np.mean(ctrl_num), np.mean(ctrl_denom)

    # Оценки отношения (AOV)
    r_test = mu_x_num / mu_x_denom
    r_ctrl = mu_y_num / mu_y_denom

    # Линеаризация метрики для каждого пользователя
    # L(u) = r + (1/mu_denom) * (X(u) - r*Y(u))
    x_linear = r_test + (1 / mu_x_denom) * (np.array(test_num) - r_test * np.array(test_denom))
    y_linear = r_ctrl + (1 / mu_y_denom) * (np.array(ctrl_num) - r_ctrl * np.array(ctrl_denom))

    # Проведение классического t-теста Уэлча
    t_stat, p_value = stats.ttest_ind(x_linear, y_linear, equal_var=False)

    return t_stat, p_value

try:
    # Загрузка данных
    df = pd.read_csv(file_path)

    # Агрегация данных до пользователя
    # num — сумма gmv (числитель), denom — количество поездок (знаменатель)
    agg = df.groupby(['user_id', 'group_name']).agg(
        num=('gmv', 'sum'),
        denom=('gmv', 'count')
    ).reset_index()

    # Разделение на группы
    control = agg[agg['group_name'] == 'control']
    test = agg[agg['group_name'] == 'test']

    # Вызов функции линеаризации
    t_stat, p_val = perform_linearization_test(
        test_num=test['num'],
        test_denom=test['denom'],
        ctrl_num=control['num'],
        ctrl_denom=control['denom']
    )

    print(f"Результат линеаризации:")
    print(f"T-статистика: {t_stat:.3f}")
    print(f"P-value: {p_val:.3f}")

    if p_val < 0.05:
        print("Гипотеза о равенстве AOV отвергнута (различия значимы).")
    else:
        print("Не удалось отвергнуть гипотезу о равенстве AOV.")

except FileNotFoundError:
    print(f"Ошибка: Файл не найден по пути {file_path}")
except Exception as e:
    print(f"Произошла непредвиденная ошибка: {e}")