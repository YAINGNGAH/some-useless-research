import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
#Считываем данные о цене акции из файла
filename = 'Some_your_stocks.csv'
stock_data = pd.read_csv(filename)
sequences = []
#Здесь мы попробуем найти ответ на вопрос: Насколько понятие "тренд" на рынке имеет смысл. В данном случае классифицировать будем с помощью дерева решений
for i in range(len(stock_data)-61):
 #Для этого используем данные о последних 60 днях, день растущий если close_price > open_price
 close_prices = stock_data.loc[i:i+61, 'Close']
 open_prices = stock_data.loc[i:i+61, 'Open']
 sequence = np.where(close_prices < open_prices, 0, 1)
 sequences.append(sequence)

matrix = np.vstack(sequences)
#Классификация последовательности дней(будет ли следующий день после нее хорошим?)
X = matrix[:, -1].reshape(-1, 1)
#Сами последовательности дней
Y = matrix[:, :-1].reshape(-1, matrix.shape[1]-1)
#Пробуем предсказывать то, будет ли день растущим, на основании того, какими были 60 дней до него
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
#precision примерно равен 0.5, такой результат мы бы получили при попытке предсказать, каким будет следующий бросок монетки на основании предыдущих







