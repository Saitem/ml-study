import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
# import mglearn
from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
plt.rc('font', family = 'Vernada')


iris_dataset = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)

iris_dataframe = pd.DataFrame(x_train, columns = iris_dataset['feature_names'])

grr = pd.plotting.scatter_matrix(iris_dataframe, c = y_train, figsize = (15, 15), marker = 'o', hist_kwds = {'bins': 20}, s = 60, alpha = .8)

knn = KNeighborsClassifier(n_neighbors = 1)

knn.fit(x_train, y_train)

x_new = np.array([[5, 2.9, 1, 0.2]])

# prediction = knn.predict(x_new)

y_pred = knn.predict(x_test)

print("Predicts for testing set:\n {}".format(y_pred))

print('{:.2f}'.format(np.mean(y_pred == y_test)))
# print('Predict: {}'.format(prediction))
# print('Predicted mark: {}'.format(iris_dataset.target_names[prediction]))

# plt.show()
