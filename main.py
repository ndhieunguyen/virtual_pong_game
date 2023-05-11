import gymnasium as gym
import time

env = gym.make("ALE/Pong-v5", render_mode="human")

# episodes = 5
# for episode in range(1, episodes + 1):
#     state = env.reset()
#     done = False
#     score = 0

#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         print(action)
#         new_state, reward, truncated, info, done = env.step(action)
#         score += reward
#         time.sleep(1)
#     print("Episode:{} Score:{}".format(episode, score))
# env.close()

obs = env.reset()
print("observation space:", env.observation_space)
print("action space:", env.action_space)
