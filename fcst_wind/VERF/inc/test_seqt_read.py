def Squential_read(nwp_name, obs_name, num_stn, num_his, num_fct):

    import numpy as np
    import readf90

    ## .. call data
    nwp, obs = [], []
    nwp = readf90.read_binary(nwp_name, num_stn, num_his, num_fct)
    obs = readf90.read_binary(obs_name, num_stn, num_his, num_fct)

    return np.array(nwp), np.array(obs)



def Squential_read_var(input_size, data_name, num_stn, num_his, num_fct):

    import numpy as np
    import readf90
    import sys

    ## .. call data
    try:
        data = readf90.read_binary(data_name, input_size, num_stn, num_his, num_fct)
    except:
        sys.exit('Error: Could not found input data: '+ data_name)
        

    return np.array(data)



def Squential_read_var_pdate(input_size, data_name, num_stn, num_his, num_fct):

    import numpy as np
    import read_plus_datef90
    import sys

    ## .. call data
    try:
        data, date = read_plus_datef90.read_binary(data_name, input_size, num_stn, num_his, num_fct)
    except:
        sys.exit('Error: Could not found input data: '+ data_name)


    return np.array(data), np.array(date)
