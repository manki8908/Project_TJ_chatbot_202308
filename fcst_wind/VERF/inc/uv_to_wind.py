import numpy as np

def uv_to_wind(u, v):

    wspd = -999.
    wdir = -999.

    #print( "U: ", u, "V: ", v )


    # .. wind speed
    wspd = np.sqrt(u*u + v*v)


    # .. wind direction
    if (u != -99) or (u != np.nan):
       if v != 0.:
          wdir = 180./3.141592*np.arctan(u/v)
       

    if v > 0. : wdir = wdir + 180.
    if (v < 0.) and (u >= 0.): wdir = wdir + 360
    if (v == 0.) and (u > 0.): wdir = 270.
    if (v == 0.) and (u < 0.): wdir = 90.
    if wspd < 0.5: wdir=0.

    if wdir > 360.: wdir = wdir - 360.
    if wdir < 0.: wdir = wdir + 360.


    return wspd, wdir
   
