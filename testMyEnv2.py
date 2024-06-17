from MyEnv2 import TwoDBOXGame
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure

env = TwoDBOXGame()
# Set up logging
log_dir = "./dqn_MyEnv2_tensorboard/"
new_logger = configure(log_dir, ["stdout", "tensorboard"])

# Instantiate the agent
model = DQN("MlpPolicy", env, verbose=1)
model.set_logger(new_logger)

# Define the evaluation and checkpoint callbacks
eval_callback = EvalCallback(env, best_model_save_path=log_dir,
                             log_path=log_dir, eval_freq=500,
                             deterministic=True, render=False)
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir,
                                         name_prefix='dqn_model')

# Create the callback list
callback = CallbackList([eval_callback, checkpoint_callback])

# Train the agent
model.learn(total_timesteps=int(2e2), progress_bar=True, callback=callback)

# Save the model
model.save("dqn_MyEnv2")

# del model  # delete trained model to demonstrate loading
#
# model = DQN.load("dqn_MyEnv2", env=env)
#
# # Evaluate policy
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
#
# #test部分
# obs = env.reset()
# for _ in range(10):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
