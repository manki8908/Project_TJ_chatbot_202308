import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.counter_epoch = 0       # 19.05.08 KMK
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, model_outdir, model_name):

        score = val_loss
        self.counter_epoch += 1      # 19.05.08 KMK

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_outdir, model_name)
        elif score > self.best_score:
            self.counter += 1
            #print 'EarlyStopping counter: {self.counter} out of {self.patience}'
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.counter_epoch%100 == 0:      # 19.05.08 KMK
               self.best_score = score
               self.save_checkpoint(val_loss, model, model_outdir, model_name)
               self.counter = 0
            #self.best_score = score
            #self.save_checkpoint(val_loss, model, model_outdir, model_name)
            #self.counter = 0

    def save_checkpoint(self, val_loss, model, outdir, name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print 'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
        torch.save(model, outdir + name)
        self.val_loss_min = val_loss
