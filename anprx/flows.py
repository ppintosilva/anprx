"""Methods for transforming and aggregating wrangled ANPR data."""
# ------------------------------------------------------------------------------

from   .utils       import log

import os
import time
import pandas       as pd
import numpy        as np
import scipy.stats  as stats
import logging      as lg

# ------------------------------------------------------------------------------

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

    tmp2_step1 = trips[(~to_expand) & (trips.trip_step == 1)]\
                .rename(columns = {'period_d' : 'period'})\
                .drop(columns=['period_o'])

    tmp2_others = trips[(~to_expand) & (trips.trip_step != 1)]\
                .rename(columns = {'period_o' : 'period'})\
                .drop(columns=['period_d'])

    tmp2 = pd.concat([tmp2_step1, tmp2_others])

    if len(tmp) > 0:
        dfs =[ tmp[(tmp.period_o <= p) & \
                   (tmp.period_d > p)]\
                  .assign(period = p) for p in periods]

        total_memory = np.array([ df.memory_usage(index=True).sum()/1e6 \
                                  for df in dfs ]).sum()

        tmp = pd.concat(dfs)

        tmp.drop(columns=['period_o','period_d'], inplace = True)

        log_memory("tmp", tmp)

        # merge expanded and not-expanded dataframes
        trips = pd.concat([tmp, tmp2])

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
              aggregator = None,
              try_discretise = True,
              remove_na = False,
              skip_explicit = True):
    """
    Aggregate trip data to compute flows.
    """
    start_time = time.time()

    log("Aggregating trips into flows: column 'period' in trips : {}"\
            .format('period' in trips.columns),
        level = lg.INFO)

    if 'period' not in trips.columns and try_discretise:
        trips = discretise_time(trips, freq)

    log("Preparing to aggregate trips into flows.", level = lg.INFO)

    if aggregator is None:
        aggregator = {
            'flow'         : ('av_speed', 'size'),
            'mean_avspeed' : ('av_speed', np.mean),
            'sd_avspeed'   : ('av_speed', np.std ),
            'skew_avspeed' : ('av_speed', stats.skew)
       }

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

    flows = trips\
            .groupby(['origin', 'destination', 'period'])\
            .agg(**aggregator)

    if skip_explicit:
        flows.reset_index(inplace = True)
        log(("SKIP filling missing combinations of (origin,destination,period)"
            " with zero flows: {}").format(flows.columns.values),
            level = lg.INFO)
    else:
        # Remove last period as the interval is open and does not include the
        # final period
        periods = get_periods(trips, freq)[:-1]
        flows = expand_flows(flows, periods)

    # making sure flow is of type int
    flows['flow'] = flows['flow'].astype(np.uint32)

    log_memory("flows", flows)

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
    log("Filling missing combinations of (o,d,period) with zero flows.",
        level = lg.INFO)

    # unique od combinations that show up in the data
    unique_ods = list(set(map(lambda x: (x[0], x[1]), flows.index.tolist())))

    # Cartesian product of unique values of 'od', and 'period'
    # Using 'od' instead of 'origin' and 'destination' prevents od combinations
    # that don't show up in the data to be included in the cartesian product
    combs_duple = pd.MultiIndex.from_product([unique_ods, periods]).tolist()

    combs_triple = list(map(lambda x: (x[0][0], x[0][1], x[1]), combs_duple))

    mux = pd.MultiIndex.from_tuples(
        combs_triple,
        names = ['origin', 'destination', 'period']
    )

    # reindex and fill with np.nan
    flows = flows.reindex(mux, fill_value=np.NaN)\
                 .fillna({'flow' : 0})\
                 .reset_index()

    expected_nrows = len(periods) * len(unique_ods)

    log(("Expected rows: {} (nperiods . unique_ods = {} . {}). "
         "Observed rows: {}.")\
            .format(expected_nrows, len(periods), len(unique_ods), len(flows)),
        level = lg.INFO)

    if assert_expected_nrows:
        assert len(flows) == expected_nrows

    return(flows)
