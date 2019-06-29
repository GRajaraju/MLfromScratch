import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# fake data
x = np.random.randn(1000, 1)
y = np.array([1]*500 + [0]*500).reshape(1000, 1)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

x_train, x_test, y_train, y_test = train_test_split(x, y,
                       test_size=.20, random_state=42)

w = np.random.randn(1, 1)

epochs = 100
weights = []
total_loss = []
print('initial weight: ',w) 

for epoch in range(epochs):
    y_pred = sigmoid(np.dot(x, w))
    loss = -np.sum((y * np.log(y_pred)) + ((1 - y) * np.log(1 - y_pred)))/x.shape[0]
    if epoch % 10 == 0:
        print('loss: ', loss)
    w = w - 0.001*np.dot(x.T, (y_pred - y))
    total_loss.append(loss)
    weights.append(w)

plt.plot(np.array(weights).squeeze(), total_loss)
plt.show()



