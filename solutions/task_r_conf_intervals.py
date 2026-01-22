import pandas as pd
import numpy as np
from scipy.stats import t
import os

"""
ЗАДАЧА R: Доверительные интервалы.
ЦЕЛЬ: Построить ДИ для разницы средних gmv, uplift gmv, разницы AOV и uplift AOV.
МЕТОД: Распределение Стьюдента (df = n + m - 2), Дельта-метод для Ratio-метрик.
ДАННЫЕ: synthetic_gmv_data_1.2.csv
"""

# Настройка путей
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, '..', 'data', 'synthetic_gmv_data_1.2.csv')


def delta_value_ci(x_num, y_num, alpha=0.05):
    """ДИ для разницы средних GMV"""
    n, m = len(x_num), len(y_num)
    mean_x, mean_y = np.mean(x_num), np.mean(y_num)
    delta = mean_x - mean_y
    var_x, var_y = np.var(x_num, ddof=1), np.var(y_num, ddof=1)
    standard_error = np.sqrt(var_x / n + var_y / m)
    df = n + m - 2  # Упрощенная формула степеней свободы
    t_crit = t.ppf(1 - alpha / 2, df)
    margin = t_crit * standard_error
    return [round(float(delta - margin), 3), round(float(delta + margin), 3)]


def delta_gmv_percent_ci(x_num, y_num, alpha=0.05):
    """ДИ для процентного изменения средних GMV"""
    n, m = len(x_num), len(y_num)
    mean_x, mean_y = np.mean(x_num), np.mean(y_num)
    delta_per = 100 * (mean_x - mean_y) / mean_y
    var_x, var_y = np.var(x_num, ddof=1), np.var(y_num, ddof=1)
    sigma_2 = (1 / mean_y ** 2) * ((var_x / n) + (mean_x ** 2 / mean_y ** 2) * (var_y / m))
    standard_error = np.sqrt(sigma_2)
    df = n + m - 2
    t_crit = t.ppf(1 - alpha / 2, df)
    margin = 100 * t_crit * standard_error
    return [round(float(delta_per - margin), 3), round(float(delta_per + margin), 3)]


def delta_aov_ci(x_num, x_denom, y_num, y_denom, alpha=0.05):
    """ДИ для разницы средних чеков (Дельта-метод)"""
    n, m = len(x_num), len(y_num)
    mu_xt, mu_yt = np.mean(x_num), np.mean(x_denom)
    mu_xc, mu_yc = np.mean(y_num), np.mean(y_denom)
    cov_t = np.cov(x_num, x_denom, ddof=1)[0, 1]
    cov_c = np.cov(y_num, y_denom, ddof=1)[0, 1]
    var_xt, var_yt = np.var(x_num, ddof=1), np.var(x_denom, ddof=1)
    var_xc, var_yc = np.var(y_num, ddof=1), np.var(y_denom, ddof=1)
    var_rt = (1 / n) * (1 / mu_yt ** 2) * (var_xt - 2 * (mu_xt / mu_yt) * cov_t + (mu_xt ** 2 / mu_yt ** 2) * var_yt)
    var_rc = (1 / m) * (1 / mu_yc ** 2) * (var_xc - 2 * (mu_xc / mu_yc) * cov_c + (mu_xc ** 2 / mu_yc ** 2) * var_yc)
    delta = (mu_xt / mu_yt) - (mu_xc / mu_yc)
    t_crit = t.ppf(1 - alpha / 2, n + m - 2)
    margin = t_crit * np.sqrt(var_rt + var_rc)
    return [round(float(delta - margin), 3), round(float(delta + margin), 3)]


def delta_aov_percent_ci(x_num, x_denom, y_num, y_denom, alpha=0.05):
    """ДИ для процентного изменения средних чеков"""
    n, m = len(x_num), len(y_num)
    xt_bar, yt_bar = np.mean(x_num), np.mean(x_denom)
    xc_bar, yc_bar = np.mean(y_num), np.mean(y_denom)
    rt, rc = xt_bar / yt_bar, xc_bar / yc_bar
    cov_t, cov_c = np.cov(x_num, x_denom, ddof=1)[0, 1], np.cov(y_num, y_denom, ddof=1)[0, 1]
    var_xt, var_yt = np.var(x_num, ddof=1), np.var(x_denom, ddof=1)
    var_xc, var_yc = np.var(y_num, ddof=1), np.var(y_denom, ddof=1)
    var_rt = (1 / n) * (1 / yt_bar ** 2) * (var_xt - 2 * rt * cov_t + rt ** 2 * var_yt)
    var_rc = (1 / m) * (1 / yc_bar ** 2) * (var_xc - 2 * rc * cov_c + rc ** 2 * var_yc)
    sigma_hat = np.sqrt((1 / rc ** 2) * var_rt + (rt ** 2 / rc ** 4) * var_rc)
    percent_delta = 100 * ((rt - rc) / rc)
    t_crit = t.ppf(1 - alpha / 2, n + m - 2)
    margin = 100 * t_crit * sigma_hat
    return [round(float(percent_delta - margin), 3), round(float(percent_delta + margin), 3)]


try:
    df = pd.read_csv(file_path)

    # Агрегация для средних GMV
    df_user = df.groupby(['user_id', 'group_name'])['gmv'].sum().reset_index()
    test_group_val = df_user[df_user['group_name'] == 'test']['gmv'].values
    control_group_val = df_user[df_user['group_name'] == 'control']['gmv'].values

    # Агрегация для AOV (Ratio-метрики)
    df_check = df.groupby(['user_id', 'group_name']).agg(num=('gmv', 'sum'), denom=('gmv', 'count')).reset_index()
    t_rat, c_rat = df_check[df_check['group_name'] == 'test'], df_check[df_check['group_name'] == 'control']
    num_t, denom_t = t_rat['num'].values, t_rat['denom'].values
    num_c, denom_c = c_rat['num'].values, c_rat['denom'].values

    # Вычисление
    ci_val = delta_value_ci(test_group_val, control_group_val)
    ci_val_per = delta_gmv_percent_ci(test_group_val, control_group_val)
    ci_rat = delta_aov_ci(num_t, denom_t, num_c, denom_c)
    ci_rat_per = delta_aov_percent_ci(num_t, denom_t, num_c, denom_c)

    # Вывод
    print(f"Δ GMV: {ci_val}")
    print(f"%Δ GMV: {ci_val_per}")
    print(f"Δ AOV: {ci_rat}")
    print(f"%Δ AOV: {ci_rat_per}")

except FileNotFoundError:
    print(f"Ошибка: Файл не найден по пути {file_path}")