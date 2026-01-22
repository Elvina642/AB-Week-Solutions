import pandas as pd
import scipy.stats as stats
import os

"""
ЗАДАЧА O: Классический t-test.
ЦЕЛЬ: Сравнить средние поюзерные gmv в тестовой и контрольной группах.
ДАННЫЕ: synthetic_gmv_data_1.2.csv
"""

# Настройка путей
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, '..', 'data', 'synthetic_gmv_data_1.2.csv')

try:
    # Загрузка данных
    df = pd.read_csv(file_path)

    # Агрегация данных до пользователя (сумма gmv)
    df_user = df.groupby(['user_id', 'group_name'])['gmv'].sum().reset_index()

    # Формирование выборок для теста
    control_gmv = df_user[df_user['group_name'] == 'control']['gmv']
    test_gmv = df_user[df_user['group_name'] == 'test']['gmv']

    # Проведение t-теста Уэлча
    # Используем поправку Уэлча, так как она не предполагает равенства дисперсий
    t_stat, p_value = stats.ttest_ind(test_gmv, control_gmv, equal_var=False)

    print(f"Результаты анализа:")
    print(f"T-статистика: {t_stat:.3f}")
    print(f"P-value: {p_value:.3f}")

    alpha = 0.05
    if p_value < alpha:
        print(f"Результат: Статистически значим на уровне {alpha} (отклоняем H0).")
    else:
        print(f"Результат: Статистически не значим (не удалось отклонить H0).")

except FileNotFoundError:
    print(f"Ошибка: Файл не найден. Убедитесь, что данные лежат в папке data/")
except Exception as e:
    print(f"Произошла непредвиденная ошибка: {e}")