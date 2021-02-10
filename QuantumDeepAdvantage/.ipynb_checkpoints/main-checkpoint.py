from Agent.dqn import *
from Environment.Environment import *
import matplotlib.pyplot as plt

NUM_QUBITS = 2
NUM_SIM = 10

env = Environment(NUM_QUBITS) 
bot = dqn(num_qubits=NUM_QUBITS)

def learning(bot, env):
  #for sim_times in range(NUM_SIM):
  reward = 0
  state  = env.state
  ct = 0
  while not env.is_terminated():
    bot.learn_from_transition(state, reward, env.is_terminated())
    move = bot.get_action(state)
    env.step(move)
    state  = env.state
    reward = env.reward()
    bot.total_reward += reward
    ct+=1

  bot.total_reward += env.reward()
  print("It's Done")
  print("The total reward is ", bot.total_reward)
  env.reset()
  # bot.reset()
  return bot.total_reward

reward_array = []
episodes = 100
for num in range(episodes):
  print("Times is "+str(num))
  r = learning(bot, env)
  reward_array.append(r)

xAxis = [ num for num in range(episodes)]
_, ag = plt.subplots()
ag.plot(xAxis, reward_array, color = 'blue', label="DQN")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ag.set(xlabel='episodes', ylabel='average return(agents)',
      title='Learning curve')
ag.grid()
plt.show()





