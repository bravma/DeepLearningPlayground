import random
from collections import Counter

import gym
import numpy as np
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.engine.saving import load_model
from keras.layers import Dense, Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

LR = 1e-3
env = gym.make("CartPole-v0")
env.reset()

goal_steps = 500
score_requirement = 50
initial_games = 10000


def random_games():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break


def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0, 2)
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]

                training_data.append([data[0], output])
        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save("saved.npy", training_data_save)

    print("Average accepted score:", np.mean(accepted_scores))
    print("Median accepted score:", np.median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data


def neural_network_model(input_size):
    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=(input_size,)))
    model.add(Dropout(0.2))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation="softmax"))
    print(model.summary())
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=LR), metrics=["accuracy"])
    return model


def train_model(training_data):
    X = np.array([i[0] for i in training_data])
    y = np.array([i[1] for i in training_data])

    model = neural_network_model(input_size=len(X[0]))

    tensorboard_log = TensorBoard(log_dir="log")
    model.fit(X, y, epochs=5, callbacks=[tensorboard_log], verbose=1)
    return model


# training_data = initial_population()
training_data = np.load("saved.npy")
# model = train_model(training_data)
# model.save("cartpole.hdf5")
model = load_model("cartpole.hdf5")

scores = []
choices = []

for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0, 2)
        else:
            x = prev_obs.reshape(-1, len(prev_obs))
            prediction = model.predict(x)
            action = np.argmax(prediction[0])
        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation])
        score += reward
        if done:
            break
    scores.append(score)

print("Average Score", sum(scores) / len(scores))
print("Choice 0: {}, Choice 1: {}"
      .format(choices.count(0) / len(choices),
              choices.count(1) / len(choices)))
