#.. module
import os
import numpy as np
import pandas as pd
import copy
import sys


def bias_rmse_avgall_fcst_print(out_nam, out_dir, val_per, g128, lstm, ecnp, emos, best):

        out_file = out_dir + out_nam + '_all_each_fcst_' + val_per

        try:
           print_out = open( out_file, 'w' )
        except:
           print ( "Can not open ", out_file )

        fcsth = [ "+%03dH" % i for i in range(6,90,3) ]

        # .. line2 print
        print ( 'MODEL', end='  ', file=print_out )
        for i in range(len(fcsth)):
            if i < len(fcsth) - 1:
               print ( fcsth[i], end='  ', file=print_out )
            else:
               print ( fcsth[i], file=print_out )

        # .. line4 print
        print ( "%5s" % "G128", end=' ', file=print_out )
        for j in range(len(fcsth)):
                if j < len(fcsth)-1:
                   print ( "%6.1f" % g128[j], end=' ', file=print_out )
                else:
                   print ( "%6.1f" % g128[j], file=print_out  )

        print ( "%5s" % "LSTM", end=' ', file=print_out )
        for j in range(len(fcsth)):
                if j < len(fcsth)-1:
                   print ( "%6.1f" % lstm[j], end=' ', file=print_out )
                else:
                   print ( "%6.1f" % lstm[j], file=print_out  )

        print ( "%5s" % "ECNP", end=' ', file=print_out )
        for j in range(len(fcsth)):
                if j < len(fcsth)-1:
                   print ( "%6.1f" % ecnp[j], end=' ', file=print_out )
                else:
                   print ( "%6.1f" % ecnp[j], file=print_out  )

        print ( "%5s" % "EMOS", end=' ', file=print_out )
        for j in range(len(fcsth)):
                if j < len(fcsth)-1:
                   print ( "%6.1f" % emos[j], end=' ', file=print_out )
                else:
                   print ( "%6.1f" % emos[j], file=print_out  )
        print ( "%5s" % "BEST", end=' ', file=print_out )
        for j in range(len(fcsth)):
                if j < len(fcsth)-1:
                   print ( "%6.1f" % best[j], end=' ', file=print_out )
                else:
                   print ( "%6.1f" % best[j], file=print_out  )

        print_out.close()

        return print("print complete " + out_file)



def bias_rmse_avgmonth_fcst_print(out_nam, out_dir, val_per, lstm):

        out_file = out_dir + out_nam + '_month_each_fcst_' + val_per

        n_month = lstm.shape[0]

        try:
           print_out = open( out_file, 'w' )
        except:
           print ( "Can not open ", out_file )

        fcsth = [ "+%03dH" % i for i in range(6,90,3) ]

        # .. line2 print
        print ( 'MODEL', end='  ', file=print_out )
        for i in range(len(fcsth)):
            if i < len(fcsth) - 1:
               print ( fcsth[i], end='  ', file=print_out )
            else:
               print ( fcsth[i], file=print_out )

        # .. line3 print
        for i in range(n_month):
                print ( "%5s" % "LSTM", end=' ', file=print_out )
                for j in range(len(fcsth)):
                      if j < len(fcsth)-1:
                           print ( "%6.1f" % lstm[i,j], end=' ', file=print_out )
                      else:
                            print ( "%6.1f" % lstm[i,j], file=print_out  )

        return print("print complete " + out_file)
