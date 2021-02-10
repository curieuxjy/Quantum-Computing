#################################################################
# Copyright (C)                                                 #
# 2019 Qiskit Team                                              #
# Permission given to modify the code as long as you keep this  #
# declaration at the top                                        #
#################################################################

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers

from typing import Dict, Tuple, Sequence, List
import copy

from Agent.network.nets import *

class dqn:
  """
  Deep Q Network

  Action Space: {x1, x2, y1, y2, z1, z2, h1, h2, c12, c21}

  Attribute
  self.num_qubits: 
  self.input_dim:

  Methods
  parse_action: convert 0 to 9 to specific gate and its argument
  """
  def __init__(self, num_qubits=2, num_action=12, gamma=0.99, alpha=10e-2, epsilon=0.01):
      self.num_qubits = num_qubits
      self.input_sz = 2 ** self.num_qubits 
      self.input_dim = ( 1, self.input_sz )
      self.num_action = num_action 
      self.init = False
      self.gamma = gamma
      self.alpha = alpha
      self.epsilon = epsilon

      self.net_instance = vanila_neural_net(self.input_sz, self.num_action, self.input_dim, self.alpha)
      self.q_network = self.net_instance.init_model()

      self.total_reward = 0
      self.win_times = 0
    
  def parse_action(self, action_num):
    if action_num == 0 or action_num == 1:
      return ["X", action_num]
    elif action_num == 2 or action_num == 3:
      return ["Y", action_num%self.num_qubits]
    elif action_num == 4 or action_num == 5:
      return ["Z", action_num%self.num_qubits]
    elif action_num == 6 or action_num == 7:
      return ["H", action_num%self.num_qubits]
    elif action_num == 8 or action_num == 9:
      return ["T", action_num%self.num_qubits]

    # It can be better!!! (Only good in 2 qubits)
    return [ "CX", [action_num%self.num_qubits, 1-(action_num%self.num_qubits)] ]
  
  def find_max_val_indx(self, q_values):
    init_flag = False
    indx_list = []
    max_val:float = None
    for indx in range(self.num_action):
      if not init_flag:
        max_val = q_values[indx] 
        indx_list.append(indx)
        init_flag = True
      else:
        if max_val < q_values[indx]:
          max_val = q_values[indx]
          indx_list = [indx]
        elif max_val == q_values[indx]:
          indx_list.append(indx)
    
    return np.random.choice(indx_list) 

  def get_action(self, state):
    self.prev_state = copy.deepcopy(state.reshape(1, self.input_sz))
    favor_action = None
    if np.random.uniform(0, 1) < self.epsilon:
      favor_action = np.random.choice(range(self.num_action))
    else:
      q_values = self.q_network.predict(self.prev_state)[0]
      favor_action = self.find_max_val_indx(q_values)

    self.prev_action = favor_action
    return self.parse_action(favor_action)

  def learn_from_transition(self, next_state, reward, terminate):
    if not self.init:
      self.init = True
      return

    state = self.prev_state
    n_state = copy.deepcopy(next_state.reshape(self.input_dim))
    action = self.prev_action
    q_table = self.q_network.predict(state)

    q_values = 0
    if not terminate:
      q_values = np.max(q_table[0])
      # print("q_values is ",q_values)
    else:
      self.init = False
      self.prev_action = None
      self.prev_state = None

    q_table[0][action] = reward + self.gamma * q_values
    self.q_network.fit(state, q_table, batch_size=1, verbose=0)

  def reset(self):
    self.init = False
    self.q_network = self.net_instance.init_model()
    # self.q_network.save_weights(filepath +'train_' + str(ag_times) + '.h5')

class drqn(dqn):

  def __init__(self, num_qubits, gamma=0.9, alpha=10e-2):
    super().__init__(num_qubits=num_qubits, gamma=gamma, alpha=alpha)
    self.net_instance = lstm(self.input_sz, self.num_action, self.input_dim, self.alpha)
    self.q_network = self.net_instance.init_model()

  # convert 1 * 2^n array into 2 * 2^n array
  def complexToReal(self, complexArray):
    return np.array([[[complexArray.real[indx], complexArray.imag[indx]] for indx in range(len(complexArray))]])

  def get_action(self, state):
    self.prev_state = copy.deepcopy(self.complexToReal(state))
    favor_action = None
    if np.random.uniform(0, 1) < self.epsilon:
      favor_action = np.random.choice(range(self.num_action))
    else:
      q_values = self.q_network.predict(self.prev_state)[0]
      favor_action = self.find_max_val_indx(q_values)

    self.prev_action = favor_action
    return self.parse_action(favor_action)

  def learn_from_transition(self, next_state, reward, terminate):
    if not self.init:
      self.init = True
      return

    state = self.prev_state
    n_state = copy.deepcopy(self.complexToReal(next_state))
    action = self.prev_action
    q_table = self.q_network.predict(state)

    q_values = 0
    if not terminate:
      q_values = np.max(q_table[0])
    else:
      self.init = False
      self.prev_action = None
      self.prev_state = None

    q_table[0][action] = reward + self.gamma * q_values
    self.q_network.fit(state, q_table, batch_size=1, verbose=0)
