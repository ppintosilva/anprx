"""Test module for data computing methods."""

from   anprx.trips       import displacement
from   anprx.trips       import all_ods_displacement
from   anprx.flows       import discretise_time
from   anprx.flows       import get_flows

import os
import numpy               as     np
import pandas              as     pd

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

    for name, group in ndf.groupby(['origin', 'destination']):
        pd.testing.assert_series_equal(group['dp'].reset_index(drop = True),
                                       expected_dp, check_names = False)
        pd.testing.assert_series_equal(group['dn'].reset_index(drop = True),
                                       expected_dn, check_names = False)

    ndf2 = all_ods_displacement(tdf, parallel = True)

    for name, group in ndf2.groupby(['origin', 'destination']):
        pd.testing.assert_series_equal(group['dp'].reset_index(drop = True),
                                       expected_dp, check_names = False)
        pd.testing.assert_series_equal(group['dn'].reset_index(drop = True),
                                       expected_dn, check_names = False)



# Trips:
# ------
#   vehicle | origin | dest | to   | td   | tt  | av_speed | distance | dp | dn
#   ----------------------------------------------------------------------------
#       1   |  A     |   B  |  0   |  90  |  90 | 45.0     | 1125     | 0  | 0
#       2   |  A     |   B  |  10  |  110 |  90 | 40.5     | 1125     | 0  | 0
#       3   |  A     |   B  |  15  |  135 |  120| 33.75    | 1125     | 1  | 0
#       4   |  A     |   B  |  5   |  95  |  95 | 45.0     | 1125     | 0  | 0
#       5   |  A     |   B  | 32   |  122 |  90 | 45.0     | 1125     | 0  | 1
#   ----------------------------------------------------------------------------
#   vehicle | origin | dest | to   | td   | tt  | av_speed | distance
#   ----------------------------------------------------------------------------
#       6   |  C     |   B  |  0   |  200 | 200 |   54     | 3000     | 2  | 0
#       7   |  C     |   B  | 35   |  195 | 160 |  67.5    | 3000     | 0  | 2
#       8   |  C     |   B  | 12   |  262 | 250 |  43.2    | 3000     | 1  | 0
#       9   |  C     |   B  |  73  |  298 | 225 |  48.0    | 3000     | 0  | 0
#       10  |  C     |   B  | 5    | 185  | 180 |  60.0    | 3000     | 0  | 1
#   ----------------------------------------------------------------------------

baseline_date = pd.to_datetime('21000101', format='%Y%m%d', errors='coerce')
freq = "30S"

fake_trips = pd.DataFrame({
    'vehicle'       : pd.Series(np.arange(10) + 1, dtype = object),
    'origin'        : np.concatenate([np.repeat('A', 5), np.repeat('C', 5)]),
    'destination'   : np.repeat('B', 10),
    't_origin'      : [0,10,15,5,32,0,35,12,73,5],
    't_destination' : [90,110,135,95,122,200,195,262,298,185],
    'distance'      : np.concatenate([np.repeat(1125,5), np.repeat(3000, 5)]),
    'trip'          : np.repeat(1,10),
    'trip_step'     : np.repeat(2,10),
    'trip_length'   : np.repeat(3,10)
})

fake_trips["od"] = fake_trips["origin"] + "_" + fake_trips["destination"]

fake_trips['t_origin'] = fake_trips['t_origin']\
    .apply(lambda x: baseline_date + pd.to_timedelta(x, unit = 's'))
fake_trips['t_destination'] = fake_trips['t_destination']\
    .apply(lambda x: baseline_date + pd.to_timedelta(x, unit = 's'))

fake_trips['travel_time'] = fake_trips['t_destination'] - fake_trips['t_origin']
fake_trips['av_speed'] = (fake_trips['distance'] * 3.6) \
                         /(fake_trips['travel_time'].dt.total_seconds())


# Expected Flows:
# ------
#   origin | dest | period| flow|  mean_avspeed
#   ----------------------------------------------------------------------------
#    A     |   B  |   0   |  4  | (45,40.5,33.75,45)
#    A     |   B  |   30  |  5  | (45,40.5,33.75,45,45)
#    A     |   B  |   60  |  5  | (45,40.5,33.75,45,45)
#    A     |   B  |   90  |  4  | (40.5,33.75,45,45)
#    A     |   B  |   120 |  2  | (33.75,45.0)
#    A     |   B  |   150 |  0  | np.nan
#    A     |   B  |   180 |  0  | np.nan
#    A     |   B  |   210 |  0  | np.nan
#    A     |   B  |   240 |  0  | np.nan
#    A     |   B  |   270 |  0  | np.nan
#   ----------------------------------------------------------------------------
#    C     |   B  |   0   |  3  |  3/3000   |(54,43.2,60)        | 7 | 3/7
#    C     |   B  |   30  |  4  |  4/3000   |(54,67.5,43.2,60)   | 9 | 4/9
#    C     |   B  |   60  |  5  |  5/3000   |(54,67.5,43.2,48,60)| 10| 1/2
#    C     |   B  |   90  |  5  |  5/3000   |(54,67.5,43.2,48,60)| 9 | 5/9
#    C     |   B  |   120 |  5  |  5/3000   |(54,67.5,43.2,48,60)| 7 | 5/7
#    C     |   B  |   150 |  5  |  5/3000   |(54,67.5,43.2,48,60)| 5 | 1.0
#    C     |   B  |   180 |  5  |  5/3000   |(54,67.5,43.2,48,60)| 5 | 1.0
#    C     |   B  |   210 |  2  |  2/3000   |(43.2,48)           | 2 | 1.0
#    C     |   B  |   240 |  2  |  2/3000   |(43.2,48)           | 2 | 1.0
#    C     |   B  |   270 |  1  |  1/3000   |(48)                | 1 | 1.0
#   ----------------------------------------------------------------------------


expected_flows = pd.DataFrame({
    'origin'        : np.concatenate([np.repeat('A', 10), np.repeat('C', 10)]),
    'destination'   : np.repeat('B', 20),
    'period'        : list(range(0,300,30)) * 2,
    'flow'          : pd.Series([4,5,5,4,2,0,0,0,0,0,
                                 3,4,5,5,5,5,5,2,2,1], dtype = np.uint32),
    'avspeed'       : [[45,40.5,33.75,45], [45,40.5,33.75,45,45],
                       [45,40.5,33.75,45,45], [40.5,33.75,45,45],
                       [33.75,45], np.nan, np.nan, np.nan, np.nan,np.nan,
                       [54,43.2,60], [54,67.5,43.2,60],
                       [54,67.5,43.2,48,60], [54,67.5,43.2,48,60],
                       [54,67.5,43.2,48,60], [54,67.5,43.2,48,60],
                       [54,67.5,43.2,48,60], [43.2,48], [43.2,48], [48.0]]
})

expected_flows['period'] = expected_flows['period']\
    .apply(lambda x: baseline_date + pd.to_timedelta(x, unit = 's'))

expected_flows['mean_avspeed'] = expected_flows['avspeed']\
    .apply(lambda x: np.mean(np.array(x)))

expected_flows['sd_avspeed'] = expected_flows['avspeed']\
    .apply(lambda x: np.nan if np.all(pd.isnull(x)) or len(x) == 1\
                            else np.std(np.array(x), ddof=1))

expected_flows = expected_flows.drop(columns = ['avspeed'])


def test_displacement_trips():
    disp_trips = all_ods_displacement(fake_trips, parallel = False)

    expected_dp_l = pd.Series([0, 0, 1, 0, 0,
                               2, 0, 1, 0, 0], dtype = np.float64)
    expected_dn_l = pd.Series([0, 0, 0, 0, 1,
                               0, 2, 0, 0, 1], dtype = np.float64)

    pd.testing.assert_series_equal(disp_trips['dp'].reset_index(drop = True),
                                   expected_dp_l, check_names = False)
    pd.testing.assert_series_equal(disp_trips['dn'].reset_index(drop = True),
                                   expected_dn_l, check_names = False)



def test_flows():
    observed_flows = get_flows(
        fake_trips,
        freq,
        skip_explicit = False
    )

    observed_flows = observed_flows\
        .drop(columns = ['skew_avspeed'])\
        .sort_values(by = ['origin', 'destination', 'period'])\
        .reset_index(drop = True)

    names =['origin', 'destination', 'period',
            'flow', 'mean_avspeed', 'sd_avspeed']

    pd.testing.assert_frame_equal(
        observed_flows,
        expected_flows,
        check_less_precise = 5,
        check_dtype = True)

def test_flows_skip_explicit():
    observed_flows = get_flows(
        fake_trips,
        freq,
        skip_explicit = True
    )

    names =['origin', 'destination', 'period',
            'flow', 'mean_avspeed', 'sd_avspeed']

    pd.testing.assert_frame_equal(
        observed_flows.drop(columns = ['skew_avspeed']),
        expected_flows\
            .loc[expected_flows.flow > 0]\
            .reset_index(drop = True),
        check_less_precise = 5,
        check_dtype = True)
