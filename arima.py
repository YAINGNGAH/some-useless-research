from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

data = pd.read_csv('AAPL (2).csv', names=['Date','Open','High','Low','Close', 'Adj Close', 'Volume'], header=0)

#calibrate for META
#Augmented Dickey-Fuller Statistic: -1.235884
#p-value: 0.657972 -> не статический d!=0

#Ищем d

#df = data.Close
#fig, axes = plt.subplots(2, 2, sharex=True)
#axes[0, 0].plot(df.diff().diff());
#axes[0, 0].set_title('Order of Differencing: Second')
#plot_acf(df.diff().diff().dropna(), ax=axes[0, 1])
#plt.show()

#График статический -> d = 2


#Ищем p

#fig, axes = plt.subplots(1, 2, sharex=True)
#df = data.Close
#axes[0].plot(df.diff());
#axes[0].set_title('Order of Differencing: First')
#axes[1].set(ylim=(0, 5))
#plot_pacf(df.diff().dropna(), ax=axes[1])
#plt.show()

#Значения автокорелляции ниже уровня значимости -> p = 1


#fig, axes = plt.subplots(1, 2, sharex=True)
#df = data.Close
#axes[0].plot(df.diff());
#axes[0].set_title('Order of Differencing: First')
#axes[1].set(ylim=(0, 1.2))
#plot_acf(df.diff().dropna(), ax=axes[1])
#plt.show()

#Значения автокорелляции ниже уровня значимости -> q = 1

mymodel = ARIMA(data.Close, order =(2, 1, 2))
modelfit = mymodel.fit()
prediction = modelfit.predict()
print(data['Close'])
print(prediction)

#Коэффициенты разброса меньше уровня значимости - подходит
plt.figure(figsize=(12, 6))
df = data.Close
modelfit.predict()
plt.plot( data['Close'], label='True Price')
plt.plot(prediction, label='Predicted Price', linestyle='--')
plt.title('Stock Price Prediction using ARIMA')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

#Попробуем оценить эффективность модели
#Изначально прибыль равна 0
profit = 0
#Будем использовать такую стратегию: Если ожидаемая цена > текущей, мы покупаем в лонг, иначе в шорт.
long = (prediction[1000] > data.Close[999])
for i in range(1000, len(prediction)-1):
    if long:
        #Прибыль в случае покупки в лонг
        profit += data.Close[i] - data.Close[i-1]
    else:
        #Прибыль в случае покупки в шорт
        profit += data.Close[i-1] - data.Close[i]
    #Вычитаем комиссию брокера: Для российских брокеров она составляет примерно 0.1%
    profit -= data.Close[i - 1] * 0.0001
    #Меняем логику покупки
    long = (prediction[i+1] > data.Close[i])
print(profit)
#profit = 550.7058699979996 META
#profit = 37 AAPL
