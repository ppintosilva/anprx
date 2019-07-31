"""Test module for data computing methods."""

from   anprx.compute       import displacement
from   anprx.compute       import all_ods_displacement

import os
import numpy               as     np
import pandas              as     pd


#   Expected Trips - Vehicle 1: 'AA00AAA'
#
#   vehicle | to  | td  | oo | od | dp | dn
#   ----------------------------------------------------------------------------
#       1   |  0  | 90  | 1  |  2 | 1  | 0
#       2   |  5  | 85  | 2  |  1 | 0  | 1
#       3   |  6  | 91  | 3  |  3 | 0  | 0
#       4   |  10 | 160 | 4  | 15 | 11 | 0
#       5   |  12 | 95  | 5  |  4 | 0  | 1
#       6   |  14 | 97  | 6  |  5 | 0  | 1
#       7   |  15 | 130 | 7  | 10 | 4  | 1
#       8   |  20 | 105 | 8  | 7  | 1  | 2
#       9   |  22 | 135 | 9  | 11 | 3  | 1
#       10  |  25 | 110 | 10 | 8  | 1  | 3
#       11  |  30 | 145 | 11 | 13 | 3  | 1
#       12  |  35 | 155 | 12 | 14 | 3  | 1
#       13  |  40 | 100 | 13 | 6  | 0  | 7
#       14  |  55 | 120 | 14 | 9  | 0  | 5
#       15  |  68 | 140 | 15 | 12 | 0  | 3
#   ----------------------------------------------------------------------------

#
# Human algorithm:
#   (dp) - for each row in od (order at destination), count how many rows BELOW
#          have a number lower than my current number (e.g. row 1, od = 2,
#          and there is only 1 row below with a value below 2 -> dp = 1)
#
#   (dn) - for each row in od (order at destination), count how many rows ABOVE
#          have a number greater
#
# sum(dp) - sum(dn) = 0
#
# Queue displacement can then be obtained by subtracting dn from dp: d = dp - dn
#
# In practice however, we bound maximum displacement at a buffer_size to make
# computations faster. Hence the sum of total positive and negative
# displacements might not be zero.

order_departure = np.arange(0,15) + 1
order_arrival   = [2,1,3,15,4,5,10,7,11,8,13,14,6,9,12]
t_origin        = [0,5,6,10,12,14,15,20,22,25,30,35,40,55,68]
t_dest          = [90,85,91,160,95,97,130,105,135,110,145,155,100,120,140]
expected_dp     = [1,0,0,11,0,0,4,1,3,1,3,3,0,0,0]
expected_dn     = [0,1,0,0,1,1,1,2,1,3,1,1,7,5,3]

t_origin = pd.Series(t_origin, dtype = np.float64)
t_dest   = pd.Series(t_dest, dtype = np.float64)

expected_dp = pd.Series(expected_dp, dtype = np.uint16)
expected_dn = pd.Series(expected_dn, dtype = np.uint16)

baseline_date = pd.to_datetime('21000101', format='%Y%m%d', errors='coerce')

df = pd.DataFrame({
    'vehicle' : order_departure,
    'origin'  : ['A'] * 15,
    'destination'  : ['B'] * 15,
    't_origin' : t_origin,
    't_destination' : t_dest
})

df['t_origin'] = df['t_origin']\
    .apply(lambda x: baseline_date + pd.to_timedelta(x, unit = 's'))

df['t_destination'] = df['t_destination']\
    .apply(lambda x: baseline_date + pd.to_timedelta(x, unit = 's'))

df['travel_time'] = df['t_destination'] - df['t_origin']

def test_displacement():

    ndf = displacement(df)

    pd.testing.assert_series_equal(ndf['dp'], expected_dp, check_names = False)
    pd.testing.assert_series_equal(ndf['dn'], expected_dn, check_names = False)


def test_displacement_all_pairs():

    df1 = df
    df2 = df.copy()

    df2['origin'] = "C"
    df2['destination'] = "D"

    tdf = pd.concat([df1,df2], axis = 0)

    ndf = all_ods_displacement(tdf, parallel = False)
    ndf2 = all_ods_displacement(tdf, parallel = True)

    for name, group in ndf.groupby(['origin', 'destination']):
        pd.testing.assert_series_equal(group['dp'].reset_index(drop = True),
                                       expected_dp, check_names = False)
        pd.testing.assert_series_equal(group['dn'].reset_index(drop = True),
                                       expected_dn, check_names = False)

    for name, group in ndf2.groupby(['origin', 'destination']):
        pd.testing.assert_series_equal(group['dp'].reset_index(drop = True),
                                       expected_dp, check_names = False)
        pd.testing.assert_series_equal(group['dn'].reset_index(drop = True),
                                       expected_dn, check_names = False)
