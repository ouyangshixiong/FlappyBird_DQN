from ple import PLE
from ple.games.flappybird import FlappyBird
from pygame.constants import K_w
import numpy as np
import parl
from parl.utils import logger
import os.path

from model import Model
from agent import Agent

# from parl.utils import ReplayMemory
from replaymemory import ReplayMemory


LEARN_FREQ = 5  # update parameters every 5 steps
MEMORY_SIZE = 20000  # replay memory size
MEMORY_WARMUP_SIZE = 200  # store some experiences in the replay memory in advance
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
GAMMA = 0.99  # discount factor of reward
MAX_STEP = 5000
actions = {"up": K_w}

def run_episode(agent, env, rpm):
    total_reward = 0
    env.init()
    step = 0
    while True:
        if step == 0:
            reward = env.act(None)
            done = False
        else:
            obs = list(env.getGameState().values())
            action = agent.sample(obs)
            if action == 1:
                act = actions["up"]
            else:
                act = None
            reward = env.act(act)
            isOver = env.game_over()
            next_obs = list(env.getGameState().values())

            # rpm.append(obs, action, reward, next_obs, isOver)
            rpm.append((obs, action, reward, next_obs, isOver))

            # train model
            # if (rpm.size() > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
                (batch_obs, batch_action, batch_reward, batch_next_obs,
                # batch_isOver) = rpm.sample_batch(BATCH_SIZE)
                batch_isOver) = rpm.sample(BATCH_SIZE)
                train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                        batch_next_obs, batch_isOver)

            total_reward += reward

            if isOver :
                env.reset_game() # 重置游戏
                break
        step += 1
    return total_reward


def evaluate(agent, env):
    # test part, run 5 episodes and average
    eval_reward = []
    for i in range(5):
        env.init()
        episode_reward = 0
        isOver = False
        step = 0
        while not isOver:
            if step == 0:
                reward = env.act(None)
                done = False
            else:
                obs = list(env.getGameState().values())
                action = agent.predict(obs)
                if action == 1:
                    act = actions["up"]
                else:
                    act = None
                reward = env.act(act)
                isOver = env.game_over()
                episode_reward += reward
            step += 1
            eval_reward.append(episode_reward)
            if step > MAX_STEP:
                break;
        env.reset_game()
    return np.mean(eval_reward)

def main():
    game = FlappyBird()
    env = PLE(game, fps=30, display_screen=False)
    env_evaluate = PLE(game, fps=30, display_screen=False)
    obs_dim = len(env.getGameState())
    action_dim = 2 # 只能是up键，还有一个其它，所以是2


    # rpm = ReplayMemory(MEMORY_SIZE, obs_dim, action_dim)
    rpm = ReplayMemory(MEMORY_SIZE)

    model = Model(act_dim=action_dim)
    algorithm = parl.algorithms.DQN(
        model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm,
        obs_dim=obs_dim,
        act_dim=action_dim,
        e_greed=0.2,  # explore
        e_greed_decrement=1e-6
    )  # probability of exploring is decreasing during training

    if os.path.exists('./model_dir'):
        agent.restore('./model_dir')

    # while rpm.size() < MEMORY_WARMUP_SIZE:  # warm up replay memory
    while len(rpm) < MEMORY_WARMUP_SIZE:  # warm up replay memory
        run_episode(agent, env, rpm)

    max_episode = 5000

    # start train
    episode = 0
    while episode < max_episode:
        # train part
        for i in range(0, 50):
            total_reward = run_episode(agent, env, rpm)
            episode += 1

        eval_reward = evaluate(agent, env_evaluate)
        logger.info('episode:{}    test_reward:{}'.format(
            episode, eval_reward))

    agent.save('./model_dir')

def test():
    game = FlappyBird()
    env = PLE(game, fps=30, display_screen=True)
    obs_dim = len(env.getGameState())
    action_dim = 2 # 只能是up键，还有一个其它，所以是2
    model = Model(act_dim=action_dim)
    algorithm = parl.algorithms.DQN(
        model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm,
        obs_dim=obs_dim,
        act_dim=action_dim,
        e_greed=0.2,  # explore
        e_greed_decrement=1e-6
    )
    if os.path.exists('./model_dir'):
        agent.restore('./model_dir')
    # test part, run 5 episodes and average
    eval_reward = []
    for i in range(5):
        env.init()
        episode_reward = 0
        isOver = False
        step = 0
        while not isOver:
            if step == 0:
                reward = env.act(None)
                done = False
            else:
                obs = list(env.getGameState().values())
                action = agent.predict(obs)
                if action == 1:
                    act = actions["up"]
                else:
                    act = None
                reward = env.act(act)
                isOver = env.game_over()
                episode_reward += reward
            step += 1
            eval_reward.append(episode_reward)
            if step > MAX_STEP:
                break;
        env.reset_game()
    return np.mean(eval_reward)

if __name__ == '__main__':
    # main()
    test()