import pandas as pd
import os

"""
ЗАДАЧА M: Оценки.
ЦЕЛЬ: Сагрегировать детализированные (пособытийные) данные до поюзерных и рассчитать несмещенные оценки математического ожидания и дисперсии. 
МЕТОД: Поюзерная агрегация (сумма gmv по user_id), расчет выборочного среднего и выборочной дисперсии (ddof=1). 
ДАННЫЕ: synthetic_gmv_data_1.1.csv 
"""

def calculate_user_metrics(file_path: str):
    try:
        # Загрузка данных
        df = pd.read_csv(file_path)

        # Поюзерная агрегация (сумма gmv на каждого пользователя)
        user_aggregated = df.groupby('user_id')['gmv'].sum().reset_index()

        # Расчет несмещенных оценок
        mean_gmv = user_aggregated['gmv'].mean()
        # Используем ddof=1 для получения именно несмещенной оценки дисперсии
        var_gmv = user_aggregated['gmv'].var(ddof=1)

        # Возвращаем значения для дальнейшего использования
        return mean_gmv, var_gmv

    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути {file_path}")
        return None, None


if __name__ == "__main__":
    # Настройка путей
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, '..', 'data', 'synthetic_gmv_data_1.1.csv')

    mean_val, var_val = calculate_user_metrics(data_path)

    if mean_val is not None:
        # Формат вывода
        print(f"{mean_val:.3f} {var_val:.3f}")