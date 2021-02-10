import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
from typing import Dict, Tuple, Sequence, List

class vanila_neural_net:
  """
  A simple neural network
  
  Attributes
  ----------
  input_sz: int
    the number of neurons in the input layer

  output_sz: int 
    the number of neurons in the output layer

  input_dim: int 
    the size of the input vector
    
  Methods
    model: layers of the simple neural network
  
  """
  def __init__(self, input_sz, output_sz, input_dim, alpha, dropout=0.0) -> None:
    self.input_sz = input_sz
    self.output_sz = output_sz
    self.input_dim = input_dim
    self.alpha = alpha
    self.dropout = dropout

  def init_model(self) -> None:
    model = models.Sequential()
    model.add(layers.Dense( self.input_sz, activation='relu', input_dim=self.input_sz))
    model.add(layers.Dropout( self.dropout ))

    model.add(layers.Dense( self.output_sz, activation='relu'))
    model.compile(optimizer=optimizers.RMSprop(lr=self.alpha),
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    return model

class lstm(vanila_neural_net):

  def __init__(self, input_sz, output_sz, input_dim, alpha, num_hidden_layer=1, dropout=0.0) -> None:
    super().__init__(input_sz,output_sz, input_dim, alpha)

  def init_model(self) -> None:
    model = models.Sequential()
    model.add(layers.LSTM(
        self.input_sz*2, 
        activation='relu',
        recurrent_initializer='glorot_uniform',
        dropout=self.dropout,
        input_shape=(4,2)
     ))
    model.add(layers.LayerNormalization())
    model.add(layers.Dense( self.output_sz, activation='relu'))
    model.compile(optimizer=optimizers.Adam(lr=self.alpha),
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    return model



