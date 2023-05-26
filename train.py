from src.model import DeepQNet
import torch
from torch import nn
import os
import random
import pickle
import numpy as np
import argparse
import gymnasium as gym
import cv2


def get_args():
    parser = argparse.ArgumentParser("Use RL to play dinorun")
    parser.add_argument("--batch_size", type=int, default=64, help="The number of frame per batch")
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_decay_iters", type=float, default=2000000)
    parser.add_argument("--num_iters", type=int, default=2000000)
    parser.add_argument("--replay_memory_size", type=int, default=50000, help="Number of epoches between testing phases")
    parser.add_argument("--saved_folder", type=str, default="models")

    args = parser.parse_args()
    return args


def train(opt):
    model = DeepQNet(input_shape=(4, 210, 160), n_actions=6)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(42)
        model.cuda()
    else:
        device = torch.device("cpu")
        torch.manual_seed(42)

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)

    if not os.path.isdir(opt.saved_folder):
        os.makedirs(opt.saved_folder)

    checkpoint_path = os.path.join(opt.saved_folder, "dino.pth")
    memory_path = os.path.join(opt.saved_folder, "replay_memory.pkl")

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        iter = checkpoint["iter"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Load trained model from iteration {}".format(iter))
    else:
        iter = 0

    if os.path.isfile(memory_path):
        with open(memory_path, "rb") as f:
            replay_memory = pickle.load(f)
        print("Load replay memory")
    else:
        replay_memory = []

    criterion = nn.MSELoss()
    env = gym.make("ALE/Pong-v5", render_mode="human")
    env.reset()
    state, reward, truncated, info, done = env.step(0)
    state = torch.tensor(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)[None, :, :], dtype=torch.float32)
    state = torch.cat(tuple(state for _ in range(4)))[None, :, :, :].to(device)

    while iter < opt.num_iters:
        prediction = model(state)
        epsilon = opt.final_epsilon + (
            max(opt.num_decay_iters - iter, 0) * (opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_iters
        )
        u = random.random()
        random_action = u <= epsilon
        if random_action:
            action = random.randint(0, 6)  # Exploration
        else:
            action = torch.argmax(prediction, axis=1)[0]  # Exploitation

        next_state, reward, truncated, info, done = env.step(action)
        next_state = torch.tensor(cv2.cvtColor(next_state, cv2.COLOR_RGB2GRAY)[None, :, :], dtype=torch.float32)
        next_state = torch.cat((state[0, 1:, :, :], next_state))[None, :, :, :].to(device)
        replay_memory.append([state, action, reward, next_state, done])
        if len(replay_memory) > opt.replay_memory_size:
            del replay_memory[0]

        batch = random.sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.from_numpy(
            np.array(
                [
                    [1, 0, 0, 0, 0, 0]
                    if action == 0
                    else [0, 1, 0, 0, 0, 0]
                    if action == 1
                    else [0, 0, 1, 0, 0, 0]
                    if action == 2
                    else [0, 0, 0, 1, 0, 0]
                    if action == 3
                    else [0, 0, 0, 0, 1, 0]
                    if action == 4
                    else [0, 0, 0, 0, 0, 1]
                    for action in action_batch
                ],
                dtype=np.float32,
            )
        )
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        current_prediction_batch = model(state_batch)
        next_prediction_batch = model(next_state_batch)
        exit()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
