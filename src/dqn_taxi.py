import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Embedding, Reshape
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

ENV_NAME = 'Taxi-v3'


# Get the environment and extract the number of actions and observations.
env = gym.make(ENV_NAME)
np.random.seed(123)
nb_actions = env.action_space.n
state_size = env.observation_space.n

# Create model
model = Sequential()
model.add(Embedding(state_size, 10))
model.add(Reshape((10,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))
print(model.summary())

# Create callbacks to log info and save model checkpoints
checkpoint_weights_filename = 'dqn_' + ENV_NAME + '_weights_{step}.h5f'
log_filename = 'dqn_{}_log.json'.format(ENV_NAME)
callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
callbacks += [FileLogger(log_filename, interval=100)]

# Configure and compile agent
memory = SequentialMemory(limit=2000, window_length=1)
policy = EpsGreedyQPolicy()
dqn_only_embedding = DQNAgent(model=model, nb_actions=nb_actions, 
                              memory=memory, nb_steps_warmup=500, target_model_update=4, policy=policy,
                              batch_size=32, gamma=.95)
dqn_only_embedding.compile(Adam(lr=1e-4), metrics=['mae'])

# Train algorithm
dqn_only_embedding.fit(env, nb_steps=1000000, verbose=1, 
                       nb_max_episode_steps=99, log_interval=100000, callbacks=callbacks)

# Save final weights
dqn_only_embedding.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Evaluate algorithm for 5 eps
dqn_only_embedding.test(env, nb_episodes=5, visualize=True, 
                        nb_max_episode_steps=99)