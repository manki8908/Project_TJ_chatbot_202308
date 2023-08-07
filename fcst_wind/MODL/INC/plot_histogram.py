#.. module
import os
import numpy as np
import scipy
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import copy


# .. histogram
def plot_histogram(run_stn_id, var_name, save_name, plot_x, plot_y, subtitle, lims):
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(9,5))
    #fig, axes = plt.subplots(nrows=2, ncols=4)
    #fig.subplots_adjust(left=0.2)
    fig.suptitle(subtitle, y=1)
    #plt.title(subtitle)
    plt.style.use('ggplot')

    count_idx = 0
    for i in range(2):
        for j in range(4):
            print("count_idx: ", count_idx)
            if count_idx <= 5:
               axes[i,j].hist(plot_x[:,:,count_idx].ravel(), bins=20)
               axes[i,j].set_title(var_name[count_idx] )
            elif count_idx==6:
               axes[i,j].hist(plot_y[:,:,0].ravel(), bins=20)
               axes[i,j].set_title(var_name[count_idx] )
            count_idx = count_idx + 1
   
    for ax1 in axes:
        for ax2 in ax1:
            ax2.set_ylim(lims)
            #ax2.set_aspect('equal')

    #plt.subplots_adjust(top=0.6)
    # .. save fig
    plt.tight_layout()
    plt.savefig(save_name)
    #plt.show()
    plt.clf()
