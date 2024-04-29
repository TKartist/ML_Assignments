from sklearn.linear_model import Lasso
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = np.load("../Data/data.npz")

X = data["x"]
y = data["y"]


def equation(x):
    design_matrix = np.column_stack(
        (x[:, 0], x[:, 1], np.cos(x[:, 0]), x[:, 1] ** 2, np.tanh(x[:, 0]))
    )
    return design_matrix


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


lasso = Lasso(alpha=0.001)
lasso.fit(X=equation(X_train), y=y_train)


y_pred = lasso.predict(equation(X_test))

print(lasso.coef_)
print(mean_squared_error(y_true=y_test, y_pred=y_pred))


# Lasso penalizes the values that is too out-bound,
# and because there aren't many values that are out-bound
# hence, Lasso Regression performs very similarly.
