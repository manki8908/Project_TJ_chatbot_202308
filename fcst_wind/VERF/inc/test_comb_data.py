def Combine_data(data1, data2):

    import numpy as np

    # .. missing value replace
    np.place(data1, data2==-999., -999.)
    np.place(data2, data1==-999., -999.)
    np.place(data1, data1==-999., np.nan)
    np.place(data2, data2==-999., np.nan)

    # .. combine data
    a, b, c = data1.shape
    comb_data = np.empty((a+1,b,c))
    comb_data
