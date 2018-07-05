from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def plot_data(pl, X, y):
    # plot class where y==0
    pl.plot(X[y == 0, 0], X[y == 0, 1], "ob", alpha=0.5)
    # plot class where y==1
    pl.plot(X[y == 1, 0], X[y == 1, 1], "xr", alpha=0.5)
    pl.legend(["0", "1"])
    return pl


def plot_decision_boundary(model, X, y):
    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)

    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]

    # make predicitons with the model and reshape the output so contourf can plot it
    c = model.predict(ab)
    Z = c.reshape(aa.shape)

    plt.figure(figsize=(12, 8))
    # plot the contour
    plt.contourf(aa, bb, Z, cmap="bwr", alpha=0.2)
    # plot the moons of data
    plot_data(plt, X, y)

    return plt


X, y = make_blobs(n_samples=1000, centers=2, random_state=42)
pl = plot_data(plt, X, y)
pl.show()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

model = Sequential()
model.add(Dense(1, input_shape=(2,), activation="sigmoid"))

model.compile(Adam(lr=0.05), "binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=50, verbose=0)

eval_result = model.evaluate(X_test, y_test)
print("\n\nTest loss:", eval_result[0], "Test accuracy:", eval_result[1])
plot_decision_boundary(model, X, y).show()