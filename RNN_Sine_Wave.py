import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

# Create sine wave data
x = np.linspace(0, 50, 501)
y = np.sin(x)

plt.plot(x,y)
plt.show()

df =pd.DataFrame(data=y, index=x, columns=['Sine'])

test_percent = 0.1
test_point = np.round(len(df)*test_percent)
test_ind = int(len(df) - test_point)

train = df.iloc[:test_ind]
test = df.iloc[test_ind:]


    