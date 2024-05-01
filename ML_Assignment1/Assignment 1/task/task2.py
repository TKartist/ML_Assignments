from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import warnings

warnings.filterwarnings("ignore")

data = np.load("../Data/data.npz")

X = data["x"]
y = data["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlp_regressor = MLPRegressor(
    hidden_layer_sizes=(100, 150, 300, 150, 100),
    activation="relu",
    solver="adam",
    random_state=42,
)

mlp_regressor.fit(X_train, y_train)
y_pred = mlp_regressor.predict(X_test)

mse = mean_squared_error(y_pred=y_pred, y_true=y_test)
print(mse)
