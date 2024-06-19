import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler



def calculate_rqa(prices, embedding_dimension, time_delay, radius):
    time_series = TimeSeries(prices, embedding_dimension=embedding_dimension, time_delay=time_delay)
    settings = Settings(time_series,
                        analysis_type=Classic,
                        neighbourhood=FixedRadius(radius),
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=1)
    computation = RQAComputation.create(settings, verbose=False)
    result = computation.run()
    return result


# Список с названиями инструментов для каждого сектора
it_instruments = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'META']
oil_gas_instruments = ['XOM', 'CVX', 'BP', 'GAZP.ME', 'LUKOY']
auto_instruments = ['TSLA', 'F', 'GM', 'TM', 'HMC']
metals_instruments = ['GLD', 'SLV', 'PALL', 'PLAT', 'FCX']
etf_instruments = ['SPY', 'IVV', 'VOO', 'QQQ', 'DIA']

# Объединяем все инструменты в один список
instruments = it_instruments + oil_gas_instruments + auto_instruments + metals_instruments + etf_instruments

# Словарь для хранения результатов RQA
results = {}
results_211 = []

# Выбор параметров для RQA
for embedding_dimension in [2, 3, 4]:
    for time_delay in [1, 2, 3]:
        for radius in [0.8, 0.9, 1.0]:
            for instrument in instruments:
                # Загружаем данные из yfinance
                data = yf.download(instrument, start='2022-01-01', end='2023-01-01')
                prices = data['Adj Close'].values
                # Выполняем RQA анализ с текущими параметрами
                rqa_results = calculate_rqa(prices, embedding_dimension, time_delay, radius)

                # Дополнительные показатели
                daily_returns = data['Adj Close'].pct_change().dropna()
                sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

                # Сохраняем результаты для инструмента и параметров
                result_dict = {
                    'RQA_Factors': {
                        'RR': round(rqa_results.recurrence_rate, 4),
                        'DET': round(rqa_results.determinism, 4),
                        'LAM': round(rqa_results.laminarity, 4),
                        'TT': round(rqa_results.trapping_time, 4),
                        'DIV': round(rqa_results.divergence, 4),
                        'diag_entr': round(rqa_results.entropy_diagonal_lines, 4),
                        'vert_entr': round(rqa_results.entropy_vertical_lines, 4),
                        'white_vert_entr': round(rqa_results.entropy_white_vertical_lines, 4),
                        'MINLINE': rqa_results.min_diagonal_line_length,
                    },
                    'Доходность': round(data['Adj Close'].pct_change().sum() * 100, 4),
                    'Волатильность': round(daily_returns.std(), 4),
                    'Sharpe Ratio': round(sharpe_ratio, 4)
                }
                results[(instrument, embedding_dimension, time_delay, radius)] = result_dict

                # Если настройки 2, 1, 1, сохраняем результат в отдельный список
                if embedding_dimension == 2 and time_delay == 1 and radius == 1:
                    row_211 = [instrument, embedding_dimension, time_delay, radius,
                               result_dict['RQA_Factors']['RR'], result_dict['RQA_Factors']['DET'],
                               result_dict['RQA_Factors']['LAM'], result_dict['RQA_Factors']['TT'],
                               result_dict['RQA_Factors']['DIV'], result_dict['RQA_Factors']['diag_entr'],
                               result_dict['RQA_Factors']['vert_entr'], result_dict['RQA_Factors']['white_vert_entr'],
                               result_dict['RQA_Factors']['MINLINE'], result_dict['Доходность'],
                               result_dict['Волатильность'], result_dict['Sharpe Ratio']]
                    results_211.append(row_211)

# Строим таблицу результатов для всех настроек
table_data = []
for key, result in results.items():
    instrument, embedding_dimension, time_delay, radius = key
    row = [instrument, embedding_dimension, time_delay, radius,
           result['RQA_Factors']['RR'], result['RQA_Factors']['DET'],
           result['RQA_Factors']['LAM'], result['RQA_Factors']['TT'],
           result['RQA_Factors']['DIV'], result['RQA_Factors']['diag_entr'],
           result['RQA_Factors']['vert_entr'], result['RQA_Factors']['white_vert_entr'],
           result['RQA_Factors']['MINLINE'], result['Доходность'], result['Волатильность'],
           result['Sharpe Ratio']]
    table_data.append(row)

df = pd.DataFrame(table_data,
                  columns=['Инструмент', 'Embedding Dimension', 'Time Delay', 'Radius',
                           'RR', 'DET', 'LAM', 'TT', 'DIV', 'diag_entr', 'vert_entr',
                           'white_vert_entr', 'MINLINE', 'Доходность', 'Волатильность',
                           'Sharpe Ratio'])

# Строим таблицу результатов для настроек 2, 1, 1
df_211 = pd.DataFrame(results_211,
                      columns=['Инструмент', 'Embedding Dimension', 'Time Delay', 'Radius',
                               'RR', 'DET', 'LAM', 'TT', 'DIV', 'diag_entr', 'vert_entr',
                               'white_vert_entr', 'MINLINE', 'Доходность', 'Волатильность',
                               'Sharpe Ratio'])

# Сокращаем все числовые значения до 4 знаков после запятой
df = df.round(4)
df_211 = df_211.round(4)
# Создание копии df_211
df_copy = df_211.copy()

# Заменяем точки на запятые в числовых значениях
df = df.applymap(lambda x: str(x).replace('.', ',') if isinstance(x, float) else x)
df_211 = df_211.applymap(lambda x: str(x).replace('.', ',') if isinstance(x, float) else x)

# Сохраняем таблицу в CSV-файл с BOM и точкой с запятой в качестве разделителя
with open('rqa_results3.csv', 'w', newline='', encoding='utf-8-sig') as f:
    for idx in range(0, len(df), 5):
        df.iloc[idx:idx + 5].to_csv(f, sep=';', index=False)
        f.write('\n')



# Сохранение таблицы с настройками 2, 1, 1 в отдельный CSV-файл
df_211.to_csv('rqa_results2113.csv', sep=';', index=False, encoding='utf-8-sig')

numeric_columns = ['RR', 'DET', 'LAM', 'TT', 'DIV', 'diag_entr', 'vert_entr', 'white_vert_entr', 'MINLINE',
                   'Доходность', 'Волатильность', 'Sharpe Ratio']
df_211[numeric_columns] = df_211[numeric_columns].applymap(
    lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)

# Нормализация показателей
scaler = MinMaxScaler()
df_211[['Доходность', 'Волатильность', 'Sharpe Ratio']] = scaler.fit_transform(
    df_211[['Доходность', 'Волатильность', 'Sharpe Ratio']])

# Взвешивание показателей (задайте веса по вашему усмотрению, здесь веса равны)
weights = {'Доходность': 1, 'Волатильность': 1, 'Sharpe Ratio': 1}

# Рассчет взвешенного балла для каждой акции
df_211['Score'] = (
        df_211['Доходность'] * weights['Доходность'] -
        df_211['Волатильность'] * weights['Волатильность'] +
        df_211['Sharpe Ratio'] * weights['Sharpe Ratio']
)

# Ранжирование акций по их общему баллу
df_211 = df_211.sort_values(by='Score', ascending=False).reset_index(drop=True)

# Определение категорий (лучшие, средние, худшие)
num_best = len(df_211) // 3
num_worst = len(df_211) // 3
num_mean = len(df_211) - num_best - num_worst

df_211.loc[:num_best - 1, 'Category'] = 'Best'
df_211.loc[num_best:num_best + num_mean - 1, 'Category'] = 'Mean'
df_211.loc[num_best + num_mean:, 'Category'] = 'Worst'

# Удаляем ненужные столбцы
df_doh_ranked = df_211[['Инструмент', 'RR', 'DET', 'LAM', 'TT', 'DIV', 'diag_entr', 'vert_entr', 'white_vert_entr', 'MINLINE', 'Доходность', 'Category']].sort_values(by='Доходность',
                                                                                              ascending=False)
df_vol_ranked = df_211[['Инструмент', 'RR', 'DET', 'LAM', 'TT', 'DIV', 'diag_entr', 'vert_entr', 'white_vert_entr', 'MINLINE', 'Волатильность', 'Category']].sort_values(by='Волатильность',
                                                                                                 ascending=True)
df_sharp_ranked = df_211[['Инструмент', 'RR', 'DET', 'LAM', 'TT', 'DIV', 'diag_entr', 'vert_entr', 'white_vert_entr', 'MINLINE', 'Sharpe Ratio', 'Category']].sort_values(by='Sharpe Ratio',
                                                                                                  ascending=False)

# Добавляем рейтинг
df_vol_ranked['Rating'] = df_vol_ranked['Волатильность'].rank(ascending=True, method='dense').astype(int).max() + 1 - \
                          df_vol_ranked['Волатильность'].rank(ascending=True, method='dense').astype(int)


df_doh_ranked = df_doh_ranked.applymap(lambda x: str(x).replace('.', ',') if isinstance(x, float) else x)
df_vol_ranked = df_vol_ranked.applymap(lambda x: str(x).replace('.', ',') if isinstance(x, float) else x)
df_sharp_ranked = df_sharp_ranked.applymap(lambda x: str(x).replace('.', ',') if isinstance(x, float) else x)

# Сохраняем таблицы в CSV
df_doh_ranked.to_csv('rqa_doh_ranked.csv', sep=';', index=False, encoding='utf-8-sig')
df_vol_ranked.to_csv('rqa_vol_ranked.csv', sep=';', index=False, encoding='utf-8-sig')
df_sharp_ranked.to_csv('rqa_sharp_ranked.csv', sep=';', index=False, encoding='utf-8-sig')

print("Доходность:")
print(df_doh_ranked)
print("\nВолатильность:")
print(df_vol_ranked)
print("\nSharpe Ratio:")
print(df_sharp_ranked)

# Список инструментов
instruments = ['AAPL', 'GLD', 'XOM']

fig, axs = plt.subplots(len(instruments), 3, figsize=(20, 1 * len(instruments)))

for i, instrument in enumerate(instruments):
    # Загружаем данные из yfinance
    data = yf.download(instrument, start='2022-01-01', end='2023-01-01')
    prices = data['Adj Close'].values

    for idx, embedding_dimension in enumerate([2, 3, 4]):
        data = []
        for time_delay in [1, 2, 3]:
            row = []
            for radius in [0.8, 0.9, 1.0]:
                rqa_results = calculate_rqa(prices, embedding_dimension, time_delay, radius)
                row.append(rqa_results.entropy_diagonal_lines)
            data.append(row)

        df = pd.DataFrame(data, columns=['R 0.8', 'R 0.9', 'R 1.0'], index=['TD 1', 'TD 2', 'TD 3'])

        # Отображение таблицы
        axs[i, idx].axis('tight')
        axs[i, idx].axis('off')
        the_table = axs[i, idx].table(cellText=df.values,
                                      colLabels=df.columns,
                                      rowLabels=df.index,
                                      cellLoc='center',
                                      loc='center')
        axs[i, idx].set_title(f'{instrument}: Embedding Dimension {embedding_dimension}')

plt.tight_layout()
plt.show()

# Создание DataFrame для 3D-графика
df_3d = pd.DataFrame(results_211, columns=['Инструмент', 'Embedding Dimension', 'Time Delay', 'Radius',
                                       'RR', 'DET', 'LAM', 'TT', 'DIV', 'diag_entr', 'vert_entr',
                                       'white_vert_entr', 'MINLINE', 'Доходность', 'Волатильность',
                                       'Sharpe Ratio'])

# Преобразование столбцов с числовыми данными в float
numeric_columns = ['Доходность', 'Волатильность', 'Sharpe Ratio']
df_3d[numeric_columns] = df_3d[numeric_columns].applymap(
    lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)

# Построение 3D-графика
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Построение точек с подписями
for i, row in df_3d.iterrows():
    ax.scatter(row['Доходность'], row['Волатильность'], row['Sharpe Ratio'], s=50)
    ax.text(row['Доходность'], row['Волатильность'], row['Sharpe Ratio'], row['Инструмент'], size=8)

ax.set_xlabel('Доходность')
ax.set_ylabel('Волатильность')
ax.set_zlabel('Коэффициент Шарпа')

plt.title('Взаимосвязь доходности, волатильности и коэффициента Шарпа')
plt.show()

# Удаление столбцов показателей RQA из df_copy
df_copy = df_copy[['Инструмент', 'Доходность', 'Волатильность', 'Sharpe Ratio']]

# Создание отдельных DataFrame для доходности, волатильности и коэффициента Шарпа
df_doh = df_copy[['Инструмент', 'Доходность']].sort_values(by="Доходность", ascending=False)
df_doh = df_doh.applymap(lambda x: str(x).replace('.', ',') if isinstance(x, float) else x)
df_doh.to_csv('rqa_doh_ranked2.csv', sep=';', index=False, encoding='utf-8-sig')
print("Доходность:")
print(df_doh)

df_vol = df_copy[['Инструмент', 'Волатильность']].sort_values(by="Волатильность", ascending=True)
df_vol = df_vol.applymap(lambda x: str(x).replace('.', ',') if isinstance(x, float) else x)
df_vol.to_csv('rqa_vol_ranked2.csv', sep=';', index=False, encoding='utf-8-sig')
print("\nВолатильность:")
print(df_vol)

df_sharp = df_copy[['Инструмент', 'Sharpe Ratio']].sort_values(by='Sharpe Ratio', ascending=False)
df_sharp = df_sharp.applymap(lambda x: str(x).replace('.', ',') if isinstance(x, float) else x)
df_sharp.to_csv('rqa_sharp_ranked2.csv', sep=';', index=False, encoding='utf-8-sig')
print("\nSharpe Ratio:")
print(df_sharp)

# Кластеризация с показателями RQA
rqa_features = ['RR', 'DET', 'LAM', 'TT', 'DIV', 'diag_entr', 'vert_entr', 'white_vert_entr', 'MINLINE']
df_rqa = df_211[rqa_features].copy()
df_rqa.fillna(0, inplace=True)  # Заполняем пропуски нулями

# Вычисляем оптимальное количество кластеров (например, методом "локтя")
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df_rqa)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Метод "локтя"')
plt.xlabel('Количество кластеров')
plt.ylabel('WCSS')
plt.show()

# Выполняем кластеризацию с оптимальным количеством кластеров
n_clusters = 3  # Выбираем число кластеров по графику "локтя"
kmeans_rqa = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_rqa.fit(df_rqa)

# Добавляем кластеры в df_211
df_211['RQA_Cluster'] = kmeans_rqa.labels_

# Названия для кластеров (основываясь на полученных результатах)
cluster_names = {
    0: 'Стабильные',
    1: 'Хаотичные',
    2: 'Смешанные'
}# Визуализация кластеров RQA в 3D с подписями для выбранных инструментов
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

features = ['Доходность', 'Волатильность', 'Sharpe Ratio']
for cluster in range(n_clusters):
    cluster_data = df_211[df_211['RQA_Cluster'] == cluster]
    ax.scatter(cluster_data['Доходность'], cluster_data['Волатильность'], cluster_data['Sharpe Ratio'],
               label=f'{cluster_names[cluster]}', s=50)

    # Подписываем только некоторые точки (например, инструменты с наибольшей доходностью)
    for i, row in cluster_data.iterrows():
        ax.text(row['Доходность'], row['Волатильность'], row['Sharpe Ratio'],
                row['Инструмент'], size=8)

ax.set_xlabel('Доходность')
ax.set_ylabel('Волатильность')
ax.set_zlabel('Коэффициент Шарпа')
ax.legend()
plt.title('Кластеризация с RQA признаками')
plt.show()


# Кластеризация без показателей RQA
df_basic = df_211[['Доходность', 'Волатильность', 'Sharpe Ratio']].copy()

# Вычисляем оптимальное количество кластеров (метод "локтя")
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df_basic)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Метод "локтя"')
plt.xlabel('Количество кластеров')
plt.ylabel('WCSS')
plt.show()
# Выполняем кластеризацию с оптимальным количеством кластеров
n_clusters = 3  # Выбираем число кластеров по графику "локтя"
kmeans_basic = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_basic.fit(df_basic)

# Добавляем кластеры в df_211
df_211['Basic_Cluster'] = kmeans_basic.labels_

# Визуализация кластеров без RQA в 3D с подписями
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

features = ['Доходность', 'Волатильность', 'Sharpe Ratio']
for cluster in range(n_clusters):
    cluster_data = df_211[df_211['Basic_Cluster'] == cluster]
    ax.scatter(cluster_data['Доходность'], cluster_data['Волатильность'], cluster_data['Sharpe Ratio'],
               label=f'{cluster_names[cluster]}', s=50)

    # Подписываем только некоторые точки (например, инструменты с наибольшей доходностью)
    for i, row in cluster_data.iterrows():
        ax.text(row['Доходность'], row['Волатильность'], row['Sharpe Ratio'],
                row['Инструмент'], size=8)

ax.set_xlabel('Доходность')
ax.set_ylabel('Волатильность')
ax.set_zlabel('Коэффициент Шарпа')
ax.legend()
plt.title('Кластеризация без RQA признаков')
plt.show()

# Заменяем номера кластеров на названия
df_211['RQA_Cluster_Name'] = df_211['RQA_Cluster'].map(cluster_names)
df_211['Basic_Cluster_Name'] = df_211['Basic_Cluster'].map(cluster_names)  # Добавляем эту строку!

df_211['Basic_Cluster'] = kmeans_basic.labels_  # Возвращаем эту строку
# Интерпретация результатов кластеризации
print("Интерпретация 3D кластеров с RQA:")
print(df_211[['Инструмент', 'RQA_Cluster_Name', 'DET', 'LAM', 'diag_entr', 'Доходность', 'Волатильность', 'Sharpe Ratio']])

# Сохраняем данные в CSV-файл с ненормированными значениями
df_211[['Инструмент', 'RQA_Cluster_Name', 'DET', 'LAM', 'diag_entr', 'Доходность', 'Волатильность', 'Sharpe Ratio']].to_csv('rqa_cluster_results.csv', sep=';', index=False, encoding='utf-8-sig')

print("Интерпретация 3D кластеров без RQA:")
print(df_211[['Инструмент', 'Basic_Cluster_Name', 'DET', 'LAM', 'diag_entr', 'Доходность', 'Волатильность', 'Sharpe Ratio']])
# Сохраняем данные в CSV-файл
df_211[['Инструмент', 'Basic_Cluster_Name', 'DET', 'LAM', 'diag_entr', 'Доходность', 'Волатильность', 'Sharpe Ratio']].to_csv('basic_cluster_results.csv', sep=';', index=False, encoding='utf-8-sig')


# Добавляем столбец "Категория"
df_211['Категория'] = df_211['Category'].apply(lambda x: 'Best' if x == 1 else ('Mean' if x == 2 else 'Worst'))

# Заменяем точки на запятые в числовых значениях
df_211 = df_211.applymap(lambda x: str(x).replace('.', ',') if isinstance(x, float) else x)

# Сохраняем данные в CSV-файл
df_211[['Инструмент', 'RQA_Cluster_Name', 'DET', 'LAM', 'diag_entr', 'Доходность', 'Волатильность', 'Sharpe Ratio', 'Категория']].to_csv('rqa_cluster_results.csv', sep=';', index=False, encoding='utf-8-sig')

df_211[['Инструмент', 'Basic_Cluster_Name', 'DET', 'LAM', 'diag_entr', 'Доходность', 'Волатильность', 'Sharpe Ratio', 'Категория']].to_csv('basic_cluster_results.csv', sep=';', index=False, encoding='utf-8-sig')
# Объяснение диагональной энтропии, равной 0
print("Объяснение нулевой диагональной энтропии:")
print("Диагональная энтропия (diag_entr) равна 0, когда в системе наблюдается высокая степень детерминированности. "
      "Это означает, что траектория системы в фазовом пространстве сильно повторяется,  "
      "и практически отсутствуют случайные изменения. "
      "В случае SPY (SPDR S&P 500 ETF Trust), высокая детерминированность, вероятно, связана с тем, что "
      "этот ETF отслеживает индекс S&P 500, который включает в себя широкий спектр акций. "
      "Высокая диверсификация ETF S&P 500 снижает волатильность и делает его движение более предсказуемым. "
      "В таких случаях диагональная энтропия может быть близка к 0.")