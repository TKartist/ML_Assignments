import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle


data = np.load("../Data/data.npz")


X = data["x"]
y = data["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def equation(x):
    design_matrix = np.column_stack(
        (x[:, 0], x[:, 1], np.cos(x[:, 0]), x[:, 1] ** 2, np.tanh(x[:, 0]))
    )
    return design_matrix


model = LinearRegression()
model.fit(equation(X_train), y=y_train)

with open("LinearRegression.pkl", "wb") as f:
    pickle.dump(model, f)
f.close()

theta_z = np.array([model.intercept_])
theta = np.concatenate((theta_z, model.coef_), axis=0)

print(theta)

y_pred = model.predict(X=equation(X_test))
print(mean_squared_error(y_true=y_test, y_pred=y_pred))
