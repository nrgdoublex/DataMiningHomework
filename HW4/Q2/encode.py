from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd

df = open("Q2vi_predict.csv")
df_out = open("Q2vi_predict_1.csv",'w')

for line in df.readlines():
    l = line.rstrip('\n').split(',')
    if l[0] == 'user_id':
        df_out.write("%s\r" %line.rstrip('\n'))
    else:
        value = float(l[1])
        if value % 1 == 0:
            value = int(value)
            df_out.write("%s,%d\r" %(l[0],value))
        else:
            df_out.write("%s,%.1f\r" %(l[0],value))
