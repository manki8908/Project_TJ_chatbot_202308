import torch
import torch.nn as nn
from torch.autograd import Variable



class lstm_reg(nn.Module):

      def __init__(self, input_size, hidden_size, num_layers, output_size=1):
          super(lstm_reg, self).__init__()

          #self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
          #self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.15, bidirectional=True)
          self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.15)
          self.reg = nn.Linear(hidden_size, output_size)

      def forward(self, x):
          x, _ = self.rnn(x)
          s, b, h = x.shape
          x = x.view(s*b, h)
          x = self.reg(x)
          x = x.view(s, b, -1)
          return x

class lstm_reg_v2(nn.Module):

      def __init__(self, input_size, hidden_size, num_layers, output_size=1, bidirectional=False):
          super(lstm_reg_v2, self).__init__()
          self.input_size = input_size
          self.hidden_size = hidden_size
          self.output_size = output_size
          self.num_layers = num_layers

          if bidirectional:
             self.num_directions = 2
          else:
             self.num_directions = 1

          self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
          self.reg = nn.Linear(self.hidden_size*self.num_directions, self.output_size)

      def init_hidden(self, batch_size):
          # .. Set initial states
          h0 = Variable(torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size)).to(device)
          c0 = Variable(torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size)).to(device)

          return h0, c0

      def forward(self, x):
          # Set initial hidden and cell states 
          h0, c0 = self.init_hidden(x.size(1))

          # .. Forward propagate LSTM
          out, _ = self.rnn(x, (h0, c0) )
 
          # .. Decode the hidden state, many to one
          s, b, h = out.shape
          out = out.view(s*b, h)
          out = self.reg(out)
          out = out.view(s, b, -1)
          return out




class general_ltsm_for_1input(nn.Module):

      def __init__(self, input_size, hidden_size, num_layers, output_size=1, bidirectional=False):
          super(lstm_reg_v1, self).__init__()

          self.input_size = input_size
          self.hidden_size = hidden_size
          self.output_size = output_size
          self.num_layers = num_layers
 
          if bidirec:
             self.num_directions = 2
          else:
             self.num_directions = 1
      
          self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
          self.linear = nn.Linear(self.hidden_size*self.num_directions, self.output_size)

      def init_hidden(self, batch_size):
          #(num_layers*num_dirctions, batch_size, hidden_size)

          hidden = Variable(torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size))
          cell = Variable(torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size))

          return hidden, cell


      def forward(self, x):

          # Set initial hidden and cell states 
          hidden, cell = self.init_hidden(x.size(1))


          # out: tensor of shape (seq_length, batch_size, hidden_size)
          out, (hidden, cell) = self.rnn(x, (hidden, cell))

 
          # Decode the hidden state, many to one
          s, b, h = out.shape
          out = out.view(s*b, h)
          out = self.linear(out)
          out = out.view(s, b, -1)
          return x

