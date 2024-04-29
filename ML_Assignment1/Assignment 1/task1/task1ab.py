import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


data = np.load("../Data/data.npz")


x = data["x"]
y = data["y"]


def equation(x):
    design_matrix = np.column_stack(
        (x[:, 0], x[:, 1], np.cos(x[:, 0]), x[:, 1] ** 2, np.tanh(x[:, 0]))
    )
    return design_matrix


model = LinearRegression()
model.fit(
    equation(x),
    y=y,
)

theta_z = np.array([model.intercept_])
theta = np.concatenate((theta_z, model.coef_), axis=0)


def model_function(theta, x):
    t0, t1, t2, t3, t4, t5 = theta
    return (
        t0
        + t1 * x[:, 0]
        + t2 * x[:, 1]
        + t3 * np.cos(x[:, 0])
        + t4 * x[:, 1] ** 2
        + t5 * np.tanh(x[:, 0])
    )


y_pred = model_function(theta, x)
print(mean_squared_error(y_true=y, y_pred=y_pred))
