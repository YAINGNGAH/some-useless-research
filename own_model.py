import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Считываем данные о цене акции из файла
filename = 'Some_your.csv'
stock_data = pd.read_csv(filename)

# Создаем новую целевую переменную, представляющую, будет ли следующий день растущим или нет
stock_data['Next_Day_Growth'] = np.where(stock_data['Close'].shift(-1) > stock_data['Open'].shift(-1), 1, 0)

sequences = []
for i in range(len(stock_data)-61):
    # Используем данные о последних 60 днях
    sequence = stock_data.loc[i:i+60, 'Next_Day_Growth'].values
    sequences.append(sequence)

matrix = np.vstack(sequences)

# Классификация последовательности дней (будет ли следующий день после нее растущим?)
X = matrix[:, :-1]
# Сами последовательности дней
y = matrix[:, -1]

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

classifier1 = LogisticRegression()
classifier2 = KNeighborsClassifier(n_neighbors= 7)
classifier3 = DecisionTreeClassifier()

# Обучение модели Логистической регрессии
classifier1.fit(X_train, y_train)
# Обучение модели методом K ближайших соседей
classifier2.fit(X_train, y_train)
# Обучение модели методом дерева решений
classifier3.fit(X_train, y_train)

# Предсказание классов
y_pred1 = classifier1.predict(X_test)
y_pred2 = classifier2.predict(X_test)
y_pred3 = classifier3.predict(X_test)

print(classification_report(y_test, y_pred1))
print(classification_report(y_test, y_pred2))
print(classification_report(y_test, y_pred3))
#Каждая из трех использованных нами моделей не не смогла качественно классифицировать(precision = 0.5) текущий день по предыдущим.







