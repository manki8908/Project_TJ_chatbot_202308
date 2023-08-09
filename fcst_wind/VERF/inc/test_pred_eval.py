"""
-------------------------------------------------------------------------
  Purpose : To evaluation model predict test using test data
  Author  : KIM MK
  Content : 
            1. Device configuration initialize
            2. Prediction

  History : 
            Code by 2019.05.09  - ManKi Kim
-------------------------------------------------------------------------
"""

import numpy as np
import torch
from torch.autograd import Variable



class Eval_prediction:

      def __init__(self): 

          # .. Device configuration
          torch.set_num_threads(4)
          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


      def __call__(self, model_name, test_data):

          # .. Model load
          net = torch.load(model_name)
          model = net.eval()

          # .. make tensor
          test_tensor = torch.from_numpy(test_data)
          var_data = Variable(test_tensor)

          # .. Model run
          pred_test = model(var_data)
          output = pred_test[:,0,0].data.numpy()

          return output
