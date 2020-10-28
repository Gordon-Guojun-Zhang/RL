import numpy as np

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--alg', type=str, default="sample_avg", help="sample_avg | constant")
args = parser.parse_args()
print(args)

qstar = np.array([0.] * 10)
mu, sigma = 0, 0.01 # mean and standard deviation
q_avg = np.array([0.] * 10)  # q value
q_const = np.array([0.] * 10)  # q value
epsilon = 0.1
alpha = 0.1
epoch = 10000
total_reward_avg = 0.
total_reward_const = 0.
avg_reward_avg = np.array([0.] * epoch)
avg_reward_const = np.array([0.] * epoch)
total_right_avg = 0.
total_right_const = 0.
avg_right_avg = np.array([0.] * epoch)
avg_right_const = np.array([0.] * epoch)

for i in range(epoch):
    s = np.random.normal(mu, sigma, 10) # random walk
    qstar += s 
    if np.random.rand(1)[0] < epsilon:
        action_avg = np.random.randint(10)
    else:
        '''random tie breaking'''
        action_avg = np.random.choice(np.flatnonzero(q_avg == q_avg.max()))
    if np.random.rand(1)[0] < epsilon:
        action_const = np.random.randint(10)
    else:
        '''random tie breaking'''
        action_const = np.random.choice(np.flatnonzero(q_const == q_const.max()))
    reward_avg = np.random.normal(qstar[action_avg], 1)
    reward_const = np.random.normal(qstar[action_const], 1)

    q_avg[action_avg] += (reward_avg - q_avg[action_avg])/(i + 1)
    q_const[action_const] += alpha * (reward_const - q_const[action_const])
    total_reward_avg += reward_avg
    total_reward_const += reward_const
    avg_reward_avg[i] = total_reward_avg/(i + 1)
    avg_reward_const[i] = total_reward_const/(i + 1)
    if q_avg.argmax() == qstar.argmax():
       total_right_avg += 1. 
    if q_const.argmax() == qstar.argmax():
       total_right_const += 1. 
    avg_right_avg[i] = total_right_avg/(i + 1)
    avg_right_const[i] = total_right_const/(i + 1)

print("qstar value: ", qstar)

import matplotlib.pyplot as plt
ax1 = plt.subplot(121)
ax1.set_title("average reward")
ax1.plot(avg_reward_avg, label='sample mean')
ax1.plot(avg_reward_const, label='constant step')
ax1.legend()

ax2 = plt.subplot(122)
ax2.set_title("optimal action")
ax2.plot(avg_right_avg, label='sample mean')
ax2.plot(avg_right_const, label='constant step')
ax2.legend()

plt.show()
