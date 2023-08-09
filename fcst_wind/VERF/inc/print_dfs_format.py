#.. module
import os
import sys
import numpy as np
import pandas as pd


def print_out_dfs_format( bicr_fcst, date, num_stn, obs_stn_id ):

    #-------------------------------------------------------------------------
    # .. Write bias corrected forecast using DFS format


    # .. make directory
    YM = date[0:6]
    DD = date[6:8]
    HH = date[8:10]

    out_dir = '../DAOU/HIST/' + YM + '/' + DD + '/'
    os.system(" mkdir -p " + out_dir)

    # .. cut fcst time
    bicr_fcst = bicr_fcst[:,6:100]

    # .. make forecast date index
    fcsth = [ "+%03dH" % i for i in range(6,100,1) ]
    sdate_utc = pd.to_datetime( date, format='%Y%m%d%H' )
    sdate_kst = pd.to_datetime( date, format='%Y%m%d%H' ) + pd.to_timedelta( 9, 'H' )
    sdate_kst_plus_6h = pd.to_datetime( date, format='%Y%m%d%H' ) + pd.to_timedelta( 15, 'H' )
    fdate = pd.date_range( sdate_kst_plus_6h, periods=len(fcsth), freq='1H' )


    # .. make write string for each line
    line1 = '{}{}{}{}'.format( str(sdate_utc.strftime('%Y%m%d%H')),'+000HOUR     ',
                               str(sdate_kst.strftime('%Y%m%d%H')),'LST' )

    # .. print file
    out_file = out_dir + 'DFS_SHRT_STN_GDPS_PMOS_TCNM_T1H.' + date + '00'

    try:
       print_out = open( out_file, 'w' )
    except:
       print ( "Can not open ", out_file )

    # .. line1 print
    print ( line1, file=print_out )

    # .. line2 print
    print ( 'STNID', end='  ', file=print_out )
    for i in range(len(fcsth)):
        if i < len(fcsth) - 1:
           print ( fcsth[i], end='  ', file=print_out )
        else:
           print ( fcsth[i], file=print_out )

    # .. line3 print
    print ( 'STNID', end=' ', file=print_out )
    for i in range(len(fdate)):
        if i < len(fdate)-1:
           print ( fdate[i].strftime('%m%d%H'), end=' ', file=print_out )
        else:
           print ( fdate[i].strftime('%m%d%HLST'), file=print_out )

    # .. line4 print
    for i in range(num_stn):
        print ( "%5d" % obs_stn_id[i], end=' ', file=print_out )
        for j in range(len(fcsth)):
            #if j < num_fct-1:
            if j < len(fcsth)-1:
               print ( "%6.1f" % bicr_fcst[i,j], end=' ', file=print_out )
            else:
               print ( "%6.1f" % bicr_fcst[i,j], file=print_out  )

    print_out.close()

