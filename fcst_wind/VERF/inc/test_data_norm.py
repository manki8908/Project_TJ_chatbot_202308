def Normalize1D(data):

    import numpy as np

    max_data = np.nanmax(data)
    min_data = np.nanmin(data)
    data_scale = max_data - min_data
    data = list(map(lambda x: (x - min_data) / data_scale, data))

    return np.array(data)


def De_normalize1D(norm_data, data):

    import numpy as np

    max_data = np.nanmax(data)
    min_data = np.nanmin(data)
    data_scale = max_data - min_data
    rescale_data = list(map(lambda x: (data_scale * x) + min_data, norm_data))

    return np.array(rescale_data)



def Normalize_Min_Max(data):

    import numpy as np
    from skilearn.preprocessing import MinMaxScaler

    func = preprocessing.MinMaxScaler()
   
    rescale_data = func.fit_transform(data)

    return np.array(rescale_data)


def Inverse_Normalize_Min_Max(data):

    import numpy as np
    from skilearn.preprocessing import MinMaxScaler

    func = preprocessing.MinMaxscaler()

    rescale_data = func.inverse_transfrom(data)

    return np.array(rescale_data)

