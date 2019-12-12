from   .utils       import log

import os
import time
import psutil
import pandas       as pd
import numpy        as np
import scipy.stats  as stats
import logging      as lg


def get_periods(trips, freq):
    start_period = trips['t_origin'].dropna().min().floor(freq)
    end_period = trips['t_destination'].dropna().max().ceil(freq)

    return pd.date_range(start = start_period,
                         end   = end_period,
                         freq  = freq)

def log_memory(name, df):
    log("{}: shape = {}, memory = {:,.2f} MB"\
            .format(name, df.shape, df.memory_usage(index=True).sum()/1e6),
        level = lg.INFO)

def discretise_time(trips, freq, sort = True):
    process = psutil.Process(os.getpid())

    start_time = time.time()
    nrows = len(trips)

    log("Discretising time of trips (len = {}) into {} periods"\
            .format(len(trips), freq),
        level = lg.INFO)

    start_period = trips['t_origin'].dropna().min().floor(freq)
    end_period = trips['t_destination'].dropna().max().ceil(freq)

    log("Start period = {}, end period = {}"\
            .format(start_period, end_period),
        level = lg.INFO)

    periods = get_periods(trips, freq)

    trips = trips.assign(
            period_o = trips.t_origin.dt.floor(freq),
            period_d = trips.t_destination.dt.ceil(freq)
        )

    # Cases where origin is NA, we want the floor of period_d instead of ceiling
    trips.loc[trips.trip_step == 1, 'period_d'] = \
        trips.loc[trips.trip_step == 1, 't_destination'].dt.floor(freq)

    log_memory("trips", trips)

    # To handle trip steps that span several time intervals, we augment
    # the original dataframe by adding as many rows per trip step as many
    # intervals it spans
    # E.g:
    # ----------------------------------------------------------------
    # INPUT
    # ----------------------------------------------------------------
    # vehicle origin destination trip trip_step o_time    d_time
    # AAA     33     34          1    2         00:00:00  00:15:00
    # ----------------------------------------------------------------
    # OUTPUT
    # ----------------------------------------------------------------
    # vehicle origin destination trip trip_step t
    # AAA     33     34          1    2         00:05:00
    # AAA     33     34          1    2         00:10:00
    # AAA     33     34          1    2         00:15:00
    # ----------------------------------------------------------------
    # The intuition being that we see the vehicle travelling that route
    # during those 3 time periods, so it counts +1 towards the flow
    # of vehicles (and towards other summary statistics, e.g. mean_avspeed)
    # for those 3 distinct time steps

    to_expand = (trips.trip_step > 1) & \
                (trips.trip_step < trips.trip_length) & \
                (trips.period_o != trips.period_d)

    tmp  = trips[to_expand]

    log_memory("tmp", tmp)

    tmp2_step1 = trips[(~to_expand) & (trips.trip_step == 1)]\
                .rename(columns = {'period_d' : 'period'})\
                .drop(columns=['period_o'])

    log_memory("tmp2_step1", tmp2_step1)

    tmp2_others = trips[(~to_expand) & (trips.trip_step != 1)]\
                .rename(columns = {'period_o' : 'period'})\
                .drop(columns=['period_d'])

    log_memory("tmp2_others", tmp2_others)

    tmp2 = pd.concat([tmp2_step1, tmp2_others])

    log_memory("tmp2", tmp2)

    log("Total process memory: {:,.1f} MB [BEFORE DEL 1]"\
            .format(process.memory_info().rss/1e6),
        level = lg.INFO)

    # release memory
    del tmp2_step1, tmp2_others

    log("Total process memory: {:,.1f} MB [BEFORE DEL 2]"\
            .format(process.memory_info().rss/1e6),
        level = lg.INFO)

    if len(tmp) > 0:
        dfs =[ tmp[(tmp.period_o <= p) & \
                   (tmp.period_d > p)]\
                  .assign(period = p) for p in periods]

        total_memory = np.array([ df.memory_usage(index=True).sum()/1e6 \
                                  for df in dfs ]).sum()

        log("Total memory in 'dfs' ({} dataframes): {:,.2f}"\
                .format(len(dfs), total_memory),
            level = lg.INFO)

        del tmp

        log("Total process memory: {:,.1f} MB [BEFORE DEL 3]"\
                .format(process.memory_info().rss/1e6),
            level = lg.INFO)

        tmp = pd.concat(dfs)

        # release memory
        del dfs

        log("Total process memory: {:,.1f} MB [BEFORE DEL 4]"\
                .format(process.memory_info().rss/1e6),
            level = lg.INFO)

        tmp.drop(columns=['period_o','period_d'], inplace = True)

        log_memory("tmp", tmp)

        # merge expanded and not-expanded dataframes
        trips = pd.concat([tmp, tmp2])

        # release memory
        del tmp, tmp2

        log("Total process memory: {:,.1f} MB [BEFORE RETURN]"\
                .format(process.memory_info().rss/1e6),
            level = lg.INFO)

        if sort:
            trips = trips.sort_values(['vehicle', 'trip', 't_destination'])\
                         .reset_index(drop = True)
        else:
            trips = trips.reset_index(drop = True)

        log_memory("trips", trips)

    else:
        trips = tmp2

    log("Discretised time in {:,.2f} sec. Added {} rows. Total rows = {}."\
            .format(time.time() - start_time, len(trips) - nrows, len(trips)),
        level = lg.INFO)

    return trips


def get_flows(trips, freq,
              agg_displacement = False,
              remove_na = False,
              try_discretise = True,
              od_separator = '_',
              skip_explicit = False,
              single_precision = False):
    """
    Aggregate trip data to compute flows.
    """
    process = psutil.Process(os.getpid())
    start_time = time.time()

    log("Aggregating trips into flows: column 'period' in trips : {}"\
            .format('period' in trips.columns),
        level = lg.INFO)

    if 'period' not in trips.columns and try_discretise:
        trips = discretise_time(trips, freq)

    log("Preparing to aggregate trips into flows.", level = lg.INFO)

    aggregator = {
        'flow'         : ('av_speed', 'size'),
        'density'      : ('distance', 'first'),
        'mean_avspeed' : ('av_speed', np.mean),
        'sd_avspeed'   : ('av_speed', np.std ),
        'skew_avspeed' : ('av_speed', stats.skew),
        'mean_tt'      : ('travel_time', np.mean),
        'sd_tt'        : ('travel_time', np.std ),
        'skew_tt'      : ('travel_time', stats.skew)
   }

    if agg_displacement:
        log("Including displacement in aggregated metrics", level = lg.INFO)
        aggregator['mean_dp'] = ('dp', np.mean)
        aggregator['sd_dp']   = ('dp', np.std)
        aggregator['mean_dn'] = ('dn', np.mean)
        aggregator['sd_dn']   = ('dn', np.std)

    # Whether to remove steps with missing origin and destination
    if remove_na:
        nrows = len(trips)

        na_od = (trips.trip_step == 1) | (trips.trip_step == trip_length)
        trips = trips.drop(na_od)

        frows = nrows - len(trips)
        log(("Removing first (origin = na) and last (dest = na) steps of "
            "every trip, before aggregating. Removed {} rows ({}%). "
            "Total = {}.")\
                .format(frows, frows/nrows*100, len(trips)),
            level = lg.INFO)

    trips = trips.assign(travel_time = trips.travel_time.dt.total_seconds())

    flows = trips\
            .groupby(['od', 'period'])\
            .agg(**aggregator)

    # Remove last period as the interval is open and does not include the
    # final period
    periods = get_periods(trips, freq)[:-1]

    log("Total process memory: {:,.1f} MB [BEFORE DEL TRIPS]"\
            .format(process.memory_info().rss/1e6),
        level = lg.INFO)

    del trips

    log("Total process memory: {:,.1f} MB [AFTER DEL TRIPS]"\
            .format(process.memory_info().rss/1e6),
        level = lg.INFO)

    flows['density'] = flows['flow']/(flows['density']/1000)

    if not skip_explicit:
        flows = expand_flows(flows, periods)

        log("Total process memory: {:,.1f} MB [AFTER REINDEX]"\
                .format(process.memory_info().rss/1e6),
            level = lg.INFO)

    else:
        flows.reset_index(inplace = True)
        log("SKIP filling missing combinations of (od,period) with zero flows: {}".format(flows.columns.values),
            level = lg.INFO)

    # making sure flow is of type int
    flows['flow'] = flows['flow'].astype(np.uint32)

    # Retrieve 'origin' and 'destination' back from 'od'
    flows['origin'], flows['destination'] = \
        flows['od'].str.split(od_separator, 1).str

    # Move 'origin' and 'destination' columns to the front
    cols = flows.columns.tolist()
    flows = flows[cols[-2:] + cols[:-2]]

    log("Computing rates and flows at destination.", level = lg.INFO)

    log_memory("flows", flows)

    flow_d = flows.groupby(['destination','period'])['flow']\
                  .agg(flow_destination = ('flow', 'sum'))\
                  .reset_index()

    log_memory("flows_destination", flow_d)

    log("Total process memory: {:,.1f} MB [BEFORE MERGE]"\
            .format(process.memory_info().rss/1e6),
        level = lg.INFO)

    flows = pd.merge(flows, flow_d,
                     on = ['destination', 'period'], how = 'left')

    log("Total process memory: {:,.1f} MB [AFTER MERGE]"\
            .format(process.memory_info().rss/1e6),
        level = lg.INFO)

    flows['rate'] = flows['flow']/flows['flow_destination']

    if single_precision:
        non_float_cols = ['origin', 'destination', 'period', 'od',
                          'flow', 'flow_destination']
        float_cols = set(flows.columns.tolist()) - set(non_float_cols)
        for col in float_cols:
            flows[col] = flows[col].astype(np.float32)


    log(("Aggregated trips into flows in {:,.2f} sec.")\
            .format(time.time() - start_time),
        level = lg.INFO)

    return flows


def expand_flows(flows, periods, assert_expected_nrows = True):
    """
    Expand flows by filling in zero flows explicitly (missing combinations of
    origin, destination, period).
    """
    # We want every combination of origin,destination,period to show up in the
    # flows, even if the flow is zero. This is useful later, for calculations
    # and makes 'missing' data explicit.
    log("Filling missing combinations of (od,period) with zero flows.",
        level = lg.INFO)

    # Used below
    unique_ods = flows.index.levels[0]

    # Cartesian product of unique values of 'od', and 'period'
    # Using 'od' instead of 'origin' and 'destination' prevents od combinations
    # that don't show up in the data to be included in the cartesian product
    mux = pd.MultiIndex.from_product([flows.index.levels[0], periods],
                                     names = ['od', 'period'])

    # reindex and fill with np.nan
    flows = flows.reindex(mux, fill_value=np.NaN)\
                 .fillna({'flow' : 0})\
                 .reset_index()

    # Not using fillna, because of cases that should remain with na:
    # od pairs for which there is no distance and therefore no speed and density
    flows.loc[flows.flow == 0, 'density'] = 0

    expected_nrows = len(periods) * len(unique_ods)

    log(("Expected rows: {} (nperiods . unique_ods = {} . {}). "
         "Observed rows: {}.")\
            .format(expected_nrows, len(periods), len(unique_ods), len(flows)),
        level = lg.INFO)

    if assert_expected_nrows:
        assert len(flows) == expected_nrows

    return(flows)

def sum_flows_datasets(list_flows):

    # every dataset needs to have the same columns and temporal resolution
    pass

def aggregate_flows(
    flows,
    by_hour = False,
    by_hour_range = False,
    by_weekday = True):
    """
    Sum all observed flows, per origin and destination and additional groups.

    When computing descriptive statistics over several datasets, we first compute
    partial sums over individual datasets, record the total number of periods,
    and then compute the average.
    """

    # Check if temporal resolution allows by_hour and by_hour_range, by_weekday

    flows = trips\
            .groupby(['od', 'period'])\
            .agg(**aggregator)
