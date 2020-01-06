"""Methods for transforming and aggregating wrangled ANPR data."""
# ------------------------------------------------------------------------------

from   .utils       import log
from   .cameras     import NA_CAMERA

import os
import time
import pandas       as pd
import numpy        as np
import logging      as lg


def get_periods(trips, freq):
    start_period = trips['t_origin'].dropna().min().floor(freq)
    end_period = trips['t_destination'].dropna().max().floor(freq)

    return pd.date_range(start = start_period,
                         end   = end_period,
                         freq  = freq)

def log_memory(name, df):
    log("{}: shape = {}, memory = {:,.2f} MB"\
            .format(name, df.shape, df.memory_usage(index=True).sum()/1e6),
        level = lg.INFO)

def discretise_time(
    trips,
    freq,
    apply_pthreshold = False,
    pthreshold = .02,
    same_period = False,
    sort = False):
    """
    Discretise time
    """
    start_time = time.time()
    nrows = len(trips)

    interval_size = pd.tseries.frequencies.to_offset(freq)

    log("Discretising time of trips (len = {}) into {} periods"\
            .format(len(trips), freq),
        level = lg.INFO)

    periods = get_periods(trips, freq)

    log("start period = {}, end period = {}"\
            .format(pd.Interval(periods[0], periods[0] + interval_size),
                    pd.Interval(periods[-1], periods[-1] + interval_size)),
        level = lg.INFO)

    # if we're aggregating over time periods of length equal or longer than an
    # offset then we consider that trips always start and end in the same period
    # This simplifies everything and hence we don't need to create multiple
    # entries for the same trip step
    if same_period:
        first_step = (trips.trip_step == 1)

        # Floor to closest Monday
        if interval_size == pd.tseries.offsets.Day(n=7):
            dow = trips.t_origin.dt.dayofweek

            trips = trips.assign(period = \
                (trips.t_origin - pd.to_timedelta(dow, unit='d')).dt.floor('D'))
            # fix for when t_origin is null
            tdest_first = trips.loc[first_step, 't_destination']
            dow = tdest_first.dt.dayofweek

            trips.loc[first_step, 'period'] = \
                (tdest_first - pd.to_timedelta(dow, unit='d')).dt.floor('D')
        else:
            trips = trips.assign(period = trips.t_origin.dt.floor(freq))
            # fix for when t_origin is null
            trips.loc[first_step, 'period'] = \
                trips.loc[first_step, 't_destination'].dt.floor(freq)

        log(("Discretised time in {:,.2f} sec. No new rows were added because "
            "frequency is large enough that we can consider that trips always "
            "start and end in the same time period.")\
                .format(time.time() - start_time),
            level = lg.INFO)
        # we can return
        return trips

    # else we consider that a trip step can count towards the total vehicle
    # flow between a origin and destination during multiple time periods
    trips = trips.assign(
        period_o = trips.t_origin.dt.floor(freq),
        period_d = trips.t_destination.dt.floor(freq)
    )

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

    trips_to_expand  = trips[to_expand]

    trips_not_expand_step1 = \
        trips[(~to_expand) & (trips.trip_step == 1)]\
            .rename(columns = {'period_d' : 'period'})\
            .drop(columns=['period_o'])

    trips_not_expand_step_last = \
        trips[(~to_expand) & (trips.trip_step != 1)]\
            .rename(columns = {'period_o' : 'period'})\
            .drop(columns=['period_d'])

    # sweet release
    del trips

    trips_not_expand = pd.concat([trips_not_expand_step1,
                                  trips_not_expand_step_last])

    # sweet release
    del trips_not_expand_step1, trips_not_expand_step_last

    log("Computed trips not to expand.")

    if len(trips_to_expand) > 0:
        dfs =[ trips_to_expand[(trips_to_expand.period_o <= p) & \
                               (trips_to_expand.period_d >= p)]\
                  .assign(period = p) for p in periods]

        trips_to_expand = pd.concat(dfs)

        # sweet release
        del dfs

        log("Computed trips to expand.")
        log_memory("trips", trips_to_expand)

        if apply_pthreshold:

            def calc_overlap(x):
                return \
                (min(x.t_destination, x.period + interval_size) \
                 - max(x.t_origin, x.period))/interval_size

            # total proportion of the time interval covered by the observation
            trips_to_expand['period_overlap'] = \
                trips_to_expand.apply(calc_overlap, axis=1)

            log(("Calculated steps with period_overlap "
                 "lower than threshold {}.").format(pthreshold))

            # Don't count towards the flow if less than ptreshold percent
            # of the period is covered by the trip step
            trips_to_expand = \
                trips_to_expand[trips_to_expand.period_overlap >= pthreshold]

            log(("Filtered steps with period_overlap "
                 "lower than threshold {}.").format(pthreshold))

            trips_to_expand.drop(columns=['period_overlap'], inplace = True)

        else:
            log("Skipped applying ptreshold.")

        trips_to_expand.drop(columns=['period_o','period_d'],
                             inplace = True)

        # merge expanded and not-expanded dataframes
        trips = pd.concat([trips_to_expand, trips_not_expand])

        log("Concatenated all trips together.")
        log_memory("trips", trips)

        # sweet release
        del trips_to_expand, trips_not_expand

        if sort:
            trips = trips.sort_values(['vehicle', 'trip', 't_destination'])\
                         .reset_index(drop = True)
        else:
            trips = trips.reset_index(drop = True)

        log_memory("trips", trips)

    else:
        trips = trips_not_expand

    log("Discretised time in {:,.2f} sec. Added {} rows. Total rows = {}."\
            .format(time.time() - start_time, len(trips) - nrows, len(trips)),
        level = lg.INFO)

    return trips


def get_flows(trips,
              aggregator = None,
              remove_na = False):
    """
    Aggregate trip data to compute flows.
    """
    start_time = time.time()

    log("Aggregating trips into flows: column 'period' in trips : {}"\
            .format('period' in trips.columns),
        level = lg.INFO)

    if 'period' not in trips.columns:
        raise ValueError(
            ("Column 'period' not in trips dataframe. "
             "Have you discretised time?"))

    log("Preparing to aggregate trips into flows.", level = lg.INFO)

    if aggregator is None:
        aggregator = {
            'flow'          : ('av_speed', 'size'),
            'median_avspeed': ('av_speed', np.median),
            'mean_avspeed'  : ('av_speed', np.mean),
            'sd_avspeed'    : ('av_speed', np.std )
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
    else:
        # keep origin and destination as NAs in aggregation
        trips = trips.fillna(value = {
            'origin'        : NA_CAMERA,
            'destination'   : NA_CAMERA
        })
        log("Replacing NA origins and destinations with {}".format(NA_CAMERA))

    flows = trips\
            .groupby(['origin', 'destination', 'period'])\
            .agg(**aggregator)


    flows.reset_index(inplace = True)

    log_memory("flows", flows)

    log(("Aggregated trips into flows in {:,.2f} sec.")\
            .format(time.time() - start_time),
        level = lg.INFO)

    return flows


def expand_flows(flows, assert_expected_nrows = True):
    """
    Expand flows by filling in zero flows explicitly (missing combinations of
    origin, destination, period).
    """
    # We want every combination of origin,destination,period to show up in the
    # flows, even if the flow is zero. This is useful later, for calculations
    # and makes 'missing' data explicit.
    log("Filling missing combinations of (o,d,period) with zero flows.",
        level = lg.INFO)

    periods = flows['period'].unique()

    # unique od combinations that show up in the data
    unique_ods = list(set(list(zip(flows.origin, flows.destination))))

    # set index so then we can reindex
    flows = flows.set_index(['origin','destination','period'])

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

    # converting back to int after reindex (w/fill np.nan) caused float casting
    flows['flow'] = flows['flow'].astype('int64')

    expected_nrows = len(periods) * len(unique_ods)

    log(("Expected rows: {} (nperiods . unique_ods = {} . {}). "
         "Observed rows: {}.")\
            .format(expected_nrows, len(periods), len(unique_ods), len(flows)),
        level = lg.INFO)

    if assert_expected_nrows:
        assert len(flows) == expected_nrows

    return(flows)
