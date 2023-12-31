def r2_3dim(y_true, y_pred):

    from sklearn.metrics import r2_score
    import numpy as np

    #print ("===================================")
    #print ( 'ypred_shape', y_pred.shape )
    #print ( 'ytrue_shape', y_true.shape )

    return r2_score( y_true.flatten(), y_pred.flatten() )



def mse_3dim(y_true, y_pred):

    from sklearn.metrics import mean_squared_error
    import numpy as np

    return mean_squared_error( y_true.flatten(), y_pred.flatten() )


def mae_3dim(y_true, y_pred):

    from sklearn.metrics import mean_absolute_error
    import numpy as np

    return mean_absolute_error( y_true.flatten(), y_pred.flatten() )
