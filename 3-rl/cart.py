#!/usr/bin/python3

import argparse
import logging
import sys

import numpy as np

import gym
#import gym.scoreboard.scoring
from gym import wrappers, logger


################################################
# CS482: this is the function that changes the
# continuous values of the state of the cart-
# pole into a single integer state value, you'll
# have to adjust this for the mountain car task
################################################


def discretize_state( x, xdot, theta, thetadot ):
    one_degree = 0.0174532
    six_degrees = 0.1047192
    twelve_degrees = 0.2094384
    fifty_degrees = 0.87266

    box = 0
    if x < -2.4 or x > 2.4 or theta < -twelve_degrees or theta > twelve_degrees:
        return -1

    if x < -0.08:
        box = 0
    elif x < 0.08:
        box = 1
    else:
        box = 2

    box *= 3
    if xdot < -0.5:
        box += 0
    elif xdot < 0.5:
        box +=1
    else:
        box +=2

    box *= 6
    if theta < -six_degrees:
        box += 0
    if theta < -one_degree:
        box += 1
    elif theta < 0:
        box += 2
    elif theta < one_degree:
        box += 3
    elif theta < six_degrees:
        box += 4
    else:
        box += 5

    box *= 3
    if thetadot < -fifty_degrees:
        box += 0
    elif thetadot < fifty_degrees:
        box += 1
    else:
        box += 2

    return box

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)

    ############################################
    # CS482: This is the line you'll have to
    # change to switch to the mountain car task
    ############################################
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    args = parser.parse_args()

    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # You can set the level to logging.DEBUG or logging.WARN if you
    # want to change the amount of output.
    logger.setLevel(logging.INFO)

    env = gym.make(args.env_id)
    outdir = '/tmp/' + 'qagent' + '-results'
    env = wrappers.Monitor(env, outdir, write_upon_reset=True, force=True)

    env.seed(0)

    ############################################
    # CS482: This initial Q-table size should 
    # change to fit the number of actions (columns)
    # and the number of observations (rows)
    ############################################


    Q = np.zeros([162, env.action_space.n])

    ############################################
    # CS482: Here are some of the RL parameters
    # you can adjust to change the learning rate
    # (alpha) and the discount factor (gamma)
    ############################################

    alpha = 0.5
    gamma = 0.97

    n_episodes = 50001
    for episode in range(n_episodes):
        tick = 0
        reward = 0
        done = False
        state = env.reset()
        s = discretize_state(state[0], state[1], state[2], state[3])
        while done != True:
            tick += 1
            action = 0
            ri = -999
            for q in range(env.action_space.n):
                if Q[s][q] > ri:
                    action = q
                    ri = Q[s][q]
            state, reward, done, info = env.step(action)
            sprime = discretize_state(state[0], state[1], state[2], state[3])
            predicted_value = np.max(Q[sprime])
            if sprime < 0:
                predicted_value = 0
                reward = -5

            Q[s,action] += 0

            s = sprime

        if episode % 1000 == 0:
            alpha *= .996

        ############################################
        # CS482: When switching to the mountain car
        # task, you will have to change this to
        # reflect the success/failure of the mountain
        # car task
        ############################################
       
        if tick < 199:
            print ("fail " + str(tick) )
        else:
            print ("success")