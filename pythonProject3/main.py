import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Функция для обучения и оценки модели
def train_and_evaluate(X, y, model_name, models):
    """
    Обучает и оценивает классификатор на заданных данных.

    Args:
        X: Матрица признаков.
        y: Вектор целевых переменных.
        model_name: Название модели.
        models: Словарь моделей.

    Returns:
        Словарь с результатами модели.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = models[model_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return {'Точность': accuracy, 'Отчет': report, 'Предсказанные категории': y_pred}  # Добавляем предсказанные категории

# Загружаем таблицы
df_doh = pd.read_csv('rqa_doh_ranked.csv', sep=';', decimal=',')
df_vol = pd.read_csv('rqa_vol_ranked.csv', sep=';', decimal=',')
df_sharp = pd.read_csv('rqa_sharp_ranked.csv', sep=';', decimal=',')

# Выбираем признаки RQA
rqa_features = ['RR', 'DET', 'LAM', 'TT', 'DIV', 'diag_entr', 'vert_entr', 'white_vert_entr', 'MINLINE']

# Обучение моделей
models = {
    'Наивный байесовский классификатор': GaussianNB(),
    'Деревья решений': DecisionTreeClassifier(random_state=42),
    'Логистическая регрессия': LogisticRegression(random_state=42),
    'Случайный лес': RandomForestClassifier(random_state=42),
    'Метод ближайших соседей': KNeighborsClassifier(),
    'Нейросети': MLPClassifier(random_state=42, hidden_layer_sizes=(100, 50), max_iter=2500)
}

# Словарь для хранения результатов
results = {
    'Доходность': {},
    'Волатильность': {},
    'Коэффициент Шарпа': {}
}

# Обработка каждой таблицы
for df, target_name in zip([df_doh, df_vol, df_sharp], ['Доходность', 'Волатильность', 'Коэффициент Шарпа']):
    # Преобразуем "Category" в числовые метки
    df['Category_Encoded'] = df['Category'].map({'Best': 0, 'Mean': 1, 'Worst': 2})

    # Выбираем признаки RQA (X) и целевые переменные (Y)
    X = df[rqa_features]
    y = df['Category_Encoded']

    # Нормализация данных
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Обучение и оценка моделей
    for model_name in models:
        results[target_name][model_name] = train_and_evaluate(X, y, model_name, models)

# Создание DataFrame с результатами
results_df = pd.DataFrame.from_dict(results, orient='index')
results_df = results_df.apply(pd.Series.explode).reset_index()

# Переименование столбцов
results_df.columns = results_df.columns.str.replace("'", "")

# Сохранение результатов в CSV
results_df.to_csv('classification_results.csv', sep=';', index=False, encoding='utf-8-sig')

# Вывод результатов в консоль
print("Результаты классификации:")
for target_name in results:
    print(f"\n{target_name}:")
    print(pd.DataFrame(results[target_name]).transpose())

# Оценка моделей
evaluation_table = pd.DataFrame(index=models.keys(), columns=['Доходность', 'Волатильность', 'Коэффициент Шарпа'])
for model_name in models.keys():
    for target_name in ['Доходность', 'Волатильность', 'Коэффициент Шарпа']:
        report = results[target_name][model_name]['Отчет']
        if report['accuracy'] > 0.5:
            evaluation_table.loc[model_name, target_name] = 'Подходит для взаимосвязи'
        else:
            evaluation_table.loc[model_name, target_name] = 'Не подходит для взаимосвязи'

print("\nОценка моделей:")
print(evaluation_table)

# Сохранение оценки моделей в CSV
evaluation_table.to_csv('model_evaluation.csv', sep=';', index=True, encoding='utf-8-sig')

# Вывод метрик precision, recall, f1-score в консоль
print("\nМетрики precision, recall, f1-score:")
for target_name in results:
    print(f"\n{target_name}:")
    for model_name in models.keys():
        report = results[target_name][model_name]['Отчет']
        print(f"\n{model_name}:")
        print(f"Precision: {report['macro avg']['precision']:.2f}")
        print(f"Recall:    {report['macro avg']['recall']:.2f}")
        print(f"F1-score:  {report['macro avg']['f1-score']:.2f}")
        print(f"Support:   {report['macro avg']['support']:.2f}") # Поддержка (Support)
        print(f"Accuracy:  {report['accuracy']:.2f}")  # Точность (Accuracy)
        print(f"Macro avg:  {report['macro avg']['f1-score']:.2f}") # Среднее значение F1-оценки
        print(f"Weighted avg:  {report['weighted avg']['f1-score']:.2f}") # Взвешенное среднее значение F1-оценки

# Вывод таблицы с предсказанными категориями
print("\nТаблица с предсказанными категориями:")
for target_name in results:
    print(f"\n{target_name}:")
    for model_name in models.keys():
        predicted_categories = results[target_name][model_name]['Предсказанные категории']
        # Используем  df.index  для  получения  индексов  из  исходного  DataFrame
        df_predicted = pd.DataFrame({'Инструмент': df['Инструмент'],
                                   'Предсказанная категория': pd.Series(predicted_categories).map({0: 'Best', 1: 'Mean', 2: 'Worst'}),
                                   'Настоящая категория': df['Category']})
        print(f"\n{model_name}:")
        print(df_predicted)

        # Считаем количество верных предсказаний
        correct_predictions = (df_predicted['Предсказанная категория'] == df_predicted['Настоящая категория']).sum()
        print(f"Количество верных предсказаний: {correct_predictions}")

