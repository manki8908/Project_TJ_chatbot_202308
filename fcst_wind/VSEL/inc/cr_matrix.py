import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import fs_mj as fs


def plot_correlation_matrix(selv_nwp, selv_header, save_name):

    print("======== plot correlation matrix ==========")

    pd_input = pd.DataFrame(data=selv_nwp, columns=selv_header)
    
    print(pd_input)
    print(pd_input.shape)
    
    df = pd_input.corr(method='pearson')
    print(df.shape)
    print(df)
    
    plt.subplots( figsize=(10,10) )
    plt.title("Correlation matrix, var list from RF over mean FI")
    mask = np.zeros_like(df, dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True
    
    sns.heatmap(df.round(1),
                cmap='RdYlBu_r',
                annot = True,
                mask=mask,
                linewidths=.5,
                cbar_kws={"shrink": .5},
                annot_kws={"size": 18},
                vmin=-1, vmax=1 )

    plt.tight_layout()
    plt.savefig(save_name)


