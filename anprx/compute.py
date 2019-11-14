"""Methods for transforming and aggregating wrangled ANPR data."""
# ------------------------------------------------------------------------------

from   .utils               import log

import gc
import os
import math
import psutil
import time
import numpy                as np
import osmnx                as ox
import pandas               as pd
import scipy.stats          as stats
import pandas.api.types     as ptypes
import networkx             as nx
import geopandas            as gpd
import logging              as lg

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import ray

# ------------------------------------------------------------------------------

def transform_anpr(anpr, od_separator = '_'):
    """
    Transform data to edge format rather than node format:

    before: vehicle | camera | timestamp | confidence
    after : vehicle | origin | destination | traveltime | t_origin | t_dest
    """
    start_time = time.time()

    anpr = anpr.assign(destination   = \
        anpr.groupby('vehicle')['camera'].shift(-1))

    anpr = anpr.assign(t_destination = \
        anpr.groupby('vehicle')['timestamp'].shift(-1))

    anpr = anpr.assign(travel_time   = \
        anpr.t_destination - anpr.timestamp)

    # Rename columns
    anpr = anpr.rename(index=str,
                       columns = {
                        "camera":"origin",
                        "timestamp":"t_origin"})

    # Replace nans in destination with 'NA' so that we can concatenate
    # origin and destination appropriately
    anpr = anpr.fillna({'destination'   : 'NA'})

    # Computing 'od' column useful to merge with camera pairs dataframe
    anpr['od'] = anpr['origin'] + od_separator + anpr['destination']

    # Sort by vehicle, t_origin: (necessary for cumulative operations)
    anpr = anpr.sort_values(by = ['vehicle','t_origin'])\
               .reset_index(drop = True)

    # # Cumulative observations
    # trips = trips.assign(observation = \
    #     trips.assign(observation = True)\
    #          .groupby('vehicle')['observation'].cumsum().astype('uint16'))

    # Reorder columns
    anpr = anpr.reindex(columns =
        ['vehicle', 'origin', 'destination', 'od', 'travel_time',
         't_origin', 't_destination', 'confidence'])

    log("Transformed anpr data in node format to edge format in {:,.2f} sec"\
            .format(time.time() - start_time),
        level = lg.INFO)

    return anpr

def trip_identification(
    anpr,
    camera_pairs,
    speed_threshold = 3.0, # km/h : 3 km/h = 1 km/20h
    duplicate_threshold = 300.0,
    maximum_av_speed = 140.0, # km/h
    od_separator = '_'
):
    """
    Identify trips in a wrangled batch of ANPR data.

    Workflow:
        a. Group by vehicle
        b. Compute time difference between consecutive observations
        c. Compute camera-pair
        d. Add distance and average speed columns
        e. Compute trips (av.speed threshold)
        f. Group by Vehicle,Trip
        g. Remove duplicates
        h. Filter observations where vehicle travels too fast
        i. Reset trip ids per vehicle to avoid trips with zero observations.

    Parameters
    ----------
    anpr : pd.DataFrame
        wrangled anpr data
    camera_pairs : pd.DataFrame
        camera pairs data
    speed_threshold : float
        minimum average speed to get from origin to destination (km/h).
        The default is 3 km/h = 1 km/20 min. It is advised to use a low
        absolute value here. Any trips that are failed to be identified can
        later be corrected using the displacement metric.
    duplicate_threshold : float
        consecutive observations at the same camera with
        travel time below this threshold are considered DUPLICATES.
    maximum_av_speed: float
        Observations with average speeds above this value
        are discarded (probable camera error).
    Returns
    -------
    trips
        The resulting trip annotated anpr data.

    """
    start_time = time.time()

    # Assert dtypes
    str_cols = ['vehicle', 'camera']
    num_cols = ['confidence']
    dt_cols  = ['timestamp']

    log("Running asserts and checks...", level = lg.INFO)

    assert all(ptypes.is_string_dtype(anpr[col]) for col in str_cols)
    assert all(ptypes.is_numeric_dtype(anpr[col]) for col in num_cols)
    assert all(ptypes.is_datetime64_any_dtype(anpr[col]) for col in dt_cols)

    camera_diff = set(anpr['camera'].unique()) - \
                  set(camera_pairs['origin'].unique())

    if len(camera_diff) > 0:
        log(("There are cameras in the anpr dataset that are not present in "
            "the camera-pairs dataset: {}.")\
                .format(camera_diff),
            level = lg.ERROR)
        assert len(camera_diff) == 0

    log("Transforming anpr data into edge format.", level = lg.INFO)

    # Transform raw anpr data to edge format
    trips = transform_anpr(anpr, od_separator)

    partial_time = time.time()

    # Preparing Camera Pairs for merge
    camera_pairs = camera_pairs.fillna({'origin' : 'NA' ,'destination' : 'NA'})

    ## adding 'od' column
    camera_pairs['od'] = camera_pairs['origin'] + od_separator + \
                         camera_pairs['destination']

    ## dropping any unused columns (e.g. 'path', 'origin', 'destination',
    ## 'direction_origin', 'direction_destination', 'valid'). This avoids
    ## storing unecessary information that will be lost anyway when aggregating

    ## It can always be obtained by merging with camera pairs again.
    camera_pairs = camera_pairs[['od', 'distance']]

    # Merge with camera pairs df to get distance, direction and validness
    trips = trips.merge(camera_pairs, how = "left", on = "od")

    log(("Merged anpr with camera pairs to gain distance and direction data "
         "in {:,.2f} sec").format(time.time() - partial_time),
        level = lg.INFO)

    partial_time = time.time()

    # replace 'NA' in destination with np.nan
    trips.loc[trips.destination == "NA", 'destination'] = np.nan

    log(("Replaced 'NA' with nan in destination column in {:,.2f} sec")\
            .format(time.time() - partial_time),
        level = lg.INFO)

    partial_time = time.time()

    # And compute average speed in km/h
    trips['av_speed'] = \
        (trips.distance * 0.001)/(trips.travel_time / np.timedelta64(1, 'h'))

    # Duplicates
    # ----------------------------------------------------------------------
    # Labelling duplicate observations as the ones which have the same origin
    # and destination occurring within X seconds, where X is the input
    # variable duplicate_thresholds
    #
    #   vehicle | origin | destination | travel time | duplicate
    #   -------   ------   -----------   -----------   ---------
    #       A       33          34          90.0 sec      False
    #       A       34          34          0.5 sec       True
    #       A       34          102         3 hours       False
    #
    # NOTE: The recorded/saved timestamp at camera 34 will be that of the FIRST
    #       observation!!! The reasoning goes that the vehicle may be stopped
    #       at a traffic light or junction for some time, thus affecting the
    #       following trip step's travel time but not the previous one. Since the
    #       vehicle was at the same location for longer the subsequent travel
    #       should depict this.
    #
    #       However, this choice may not be adequate depending on the use case.
    #       Especially if the difference between the first and second
    #       observations is considerable: 30+ seconds. This may affect the
    #       accuracy of models relying on travel times, as there may be
    #       significant differences in travel times depending on which
    #       timestamp is kept.
    #

    trips['duplicate'] = \
        (trips.origin == trips.destination) & \
        (trips['travel_time'].dt.total_seconds() < duplicate_threshold)

    # tricky bit: correct the timestamp 't_origin' of the
    #             next valid observation of that vehicle
    # ---
    # index+1 is guaranteed to return a row for the exact same vehicle,
    # because the last observation of any group of observations in the dataset
    # always have NA as destination, where the dataset is grouped by vehicle
    # ---
    # This becomes extra tricky when working with several duplicates in a row
    # If there was only one duplicate at a time, the fix would be:
    #
    #   trips.loc[trips[trips.duplicate].index+1, 't_origin'] = \
    #       trips.loc[trips.duplicate, 't_origin'].values
    # ---

    # But we know that there are many [True -> False] transitions as there
    # are [False -> True]. I.e. for a given duplicate, there is always a
    # subsequent non-duplicate observation. By finding these transitions,
    # we are effectively finding the first duplicates and then the corresponding
    # valid observation.
    # ASSUMPTION: trips are sorted by vehicle and timestamp
    # (otherwise, this won't work)

    v = trips['duplicate'].astype(np.uint8)
    transitions = v - v.shift(1).fillna(0)

    # Transition True to False means the first valid observation after a
    # sequence of duplicates
    duplicate_reset_ind  = np.where(transitions == -1)[0]
    # Transition False -> True means the first duplicate after a sequence of
    # valid observations
    first_duplicates_ind = np.where(transitions == 1)[0]

    trips.loc[duplicate_reset_ind, 't_origin'] = \
        trips.loc[first_duplicates_ind, 't_origin'].values

    trips.loc[duplicate_reset_ind, 'travel_time'] = \
        trips.loc[duplicate_reset_ind, 't_destination'] - \
        trips.loc[duplicate_reset_ind, 't_origin']

    trips.loc[duplicate_reset_ind, 'av_speed'] = \
        (trips.loc[duplicate_reset_ind, 'distance'] * 0.001) / \
        (trips.loc[duplicate_reset_ind, 'travel_time'] / np.timedelta64(1, 'h'))

    log("Identified {} duplicates using temporal threshold = {:,.1f} sec"\
            .format(len(trips.loc[trips.duplicate == True]),
                    duplicate_threshold),
        level = lg.INFO)

    log("Describing travel time of duplicates:\n{}"\
            .format(trips.loc[trips.duplicate==True, 'travel_time'].describe()),
        level = lg.INFO)

    # Drop duplicates
    nrows = len(trips)
    trips = trips.drop(trips.loc[trips['duplicate'] == True].index)
    trips = trips.drop(columns = 'duplicate')

    if len(trips) == 0:
        log("No more rows, after filtering duplicate observations.",
            level = lg.WARNING)
        return trips

    frows = nrows - len(trips)
    log(("Filtered {}/{} ({:,.2f} %) observations labelled as duplicates "
        "in {:,.2f} sec. Total: {}")\
            .format(frows, nrows, frows/nrows*100,
                    time.time() - partial_time, len(trips)),
        level = lg.INFO)

    partial_time = time.time()

    # Unfeasible observations
    # --------------------------------------------------------------------------
    # Drop observations whose average speed is greater than "physically"
    # possible. Caused probably by camera errors.
    #
    # Care must be taken when removing rows that are in edge format, otherwise
    # the following observation will have an incorrect origin or destination
    #
    #   vehicle | origin | destination | travel time | duplicate
    #   -------   ------   -----------   -----------   ---------
    #       A       33          34          90.0 sec      False
    #       A       34          34          0.5 sec       True
    #       A       34          102         3 hours       False

    nrows = len(trips)

    # we use the same strategy as above (duplicates)
    # we want to modify the origin/t_origin field of the first valid
    # observation after a sequence of invalid observations
    v = (trips['av_speed'] > maximum_av_speed).astype(np.uint8)

    transitions = (v - v.shift(1).fillna(0))

    reset_ind  = transitions == -1
    first_ind  = transitions == 1

    trips.loc[reset_ind, 'origin'] = \
        trips.loc[first_ind, 'origin'].values

    trips.loc[reset_ind, 't_origin'] = \
        trips.loc[first_ind, 't_origin'].values

    trips.loc[reset_ind, 'od'] = \
        trips.loc[reset_ind, 'origin'] + od_separator + \
        trips.loc[reset_ind, 'destination']

    trips.loc[reset_ind, 'travel_time'] = \
        trips.loc[reset_ind, 't_destination'] - \
        trips.loc[reset_ind, 't_origin']

    trips.loc[reset_ind] = \
        trips.loc[reset_ind,
                  ['vehicle','origin','destination','od',
                   'travel_time','t_origin','t_destination','confidence']]\
             .merge(camera_pairs, how = "left", on = "od")\
             .set_index(trips.loc[reset_ind].index)

    trips.loc[reset_ind, 'av_speed'] = \
        (trips.loc[reset_ind, 'distance'] * 0.001) / \
        (trips.loc[reset_ind, 'travel_time'] / np.timedelta64(1, 'h'))

    trips = trips.drop(trips.loc[trips['av_speed'] > maximum_av_speed].index)

    if len(trips) == 0:
        log("No more rows, after filtering unfeasible observations.",
            level = lg.WARNING)
        return trips

    frows = nrows - len(trips)
    log(("Filtered {}/{} ({:,.2f} %) observations labelled as unfeasible "
         "(when a vehicle moves unrealistically fast between two cameras) "
         "in {:,.2f} sec. Total: {}")\
            .format(frows, nrows, frows/nrows*100,
                    time.time() - partial_time, len(trips)),
        level = lg.INFO)

    partial_time = time.time()

    # Trip Identification:
    #   Currently based on average speed: if the average speed of the vehicle
    #   travelling the shortest path between origin and destination is lower than
    #   a threshold (default: 1 km / 20 min), then
    #
    #   Consecutive observations on the same camera are not allowed, and are
    #   considered as separate trips. The reason behind this is that we can't
    #   calculate the shortest path for a self-camera pair. Hence, has we have
    #   no distance, then we can't compute the average speed. We therefore
    #   assume that trips can't have the same camera as consecutive observations.
    #   If this assumption needs to be relaxed, then a travel time thresholding
    #   approach can be used, as we did above when identifying duplicates.
    #
    #
    #   vehicle | origin | destination | av speed    | trip
    #   -------   ------   -----------   -----------   ---------
    #       A       33          34          90 sec        1
    #       A       34          102         3 hours       2
    #       A       102         33          4 min         2
    #
    # ====

    trips['trip'] = \
        (trips['av_speed'] < speed_threshold) | \
        (trips.origin == trips.destination)

    # Assign trip ids
    trips['trip'] = trips.groupby('vehicle')['trip'].shift(1).fillna(False)
    # trips start at index 1
    trips['trip'] = trips.groupby('vehicle')['trip'].cumsum()\
                         .astype('uint64')+1

    # Add origin=NA first dummy observation to every trip
    fdf = trips.groupby(['vehicle', 'trip']).nth(0).reset_index()
    fdf['destination'  ]  = fdf['origin']
    fdf['origin'       ]  = np.nan
    fdf['rest_time'    ]  = np.nan
    fdf['travel_time'  ]  = pd.NaT
    fdf['t_destination']  = fdf['t_origin']
    fdf['t_origin'     ]  = pd.NaT
    fdf['confidence'   ]  = np.nan
    fdf['od'           ]  = "NA" + od_separator + fdf['destination']
    fdf['distance'     ]  = np.nan
    fdf['av_speed'     ]  = np.nan

    nrows = len(trips)

    trips = pd.concat([trips, fdf], ignore_index = True, sort = False)

    log(("Augmented each trip with one extra row, total : {} representing the "
         "missing info about the true origin and destination of each trip.")\
            .format(len(trips) - nrows),
        level = lg.INFO)

    # Now we sort again for cumcount coming up
    # We sort by time at destination because when sorting NAs, these come last,
    # so if we sorted by t_origin, the first step (t_origin=NA) would be
    # wrongly be sorted as last in the trip. Sorting by t_destination instead,
    # fixes this (t_destination=NA is last in the trip sequence).
    trips = trips.sort_values(by = ['vehicle', 'trip', 't_destination'])\
                 .reset_index(drop = True)

    # Adding info cols about trips
    trips['trip_length'] = trips.groupby(['vehicle','trip'])['origin']\
                                .transform('size').astype('uint16')

    trips['trip_step'  ] = trips.groupby(['vehicle','trip'])\
                                .cumcount().astype('uint16')+1

    # rest time
    ntrips = dict(trips.groupby(['vehicle'])['trip']\
                       .unique().apply(lambda x: len(x)))

    trips['ntrips'] = trips.vehicle.apply(lambda x: ntrips[x])

    trips.loc[(trips.trip > 1) & (trips.trip_step == 1), 'rest_time'] = \
        trips.loc[(trips.trip < trips.ntrips) & \
                  (trips.trip_step == trips.trip_length), 'travel_time'].values

    trips['rest_time'] = trips['rest_time'].astype('timedelta64[ns]')

    # Add nans and NaTs for appropriate variables at the last step of each group
    trips.loc[trips.trip_step == trips.trip_length,
      ['destination', 'distance', 'av_speed']] = np.nan

    trips.loc[trips.trip_step == trips.trip_length,
      ['t_destination', 'travel_time']] = pd.NaT

    trips.loc[trips.trip_step == trips.trip_length, 'od'] = \
        trips[trips.trip_step == trips.trip_length]['od']\
             .str.split(pat=od_separator).str[0] + od_separator + "NA"

    log(("Added trips to each vehicle and fixed the corresponding first/last "
         "steps in {:,.2f} sec").format(time.time() - partial_time),
        level = lg.INFO)

    log("Identified trips from raw anpr data in {:,.2f} sec"\
            .format(time.time() - start_time),
        level = lg.INFO)

    return trips

def all_ods_displacement(
    df,
    buffer_size = 100,
    parallel = True,
    shutdown_ray = True
):
    """
    Calculate displacement for all origin-destination pairs in dataframe.
    """
    start_time = time.time()

    dfs = []

    if parallel:
        j = 0
        jobs = []

        if not ray.is_initialized():
            ray.init()

    has_trips = 'trip' in df.columns
    if has_trips:
        # remove first and last steps of every trip step
        time_na = (df.trip_step == 1) | (df.trip_step == df.trip_length)
        to_keep = df[time_na]
        to_compute = df[~time_na]
        dfs.append(to_keep)

    else:
        to_compute = df

    groups = to_compute.groupby(['origin', 'destination'])
    ngroups = len(groups.groups)
    checkpoints =  np.unique(np.linspace(1, ngroups,10)\
                               .round().astype(np.int32))

    log(("Calculating displacement for {} groups and {} observations. "
         "Checkpoints = {}. Parallelize = {}. Has trips = {}. "
         "This may take a while.")\
            .format(ngroups, len(to_compute), len(checkpoints),
                    parallel, has_trips),
        level = lg.INFO)

    # group counter
    j = 0

    # Main loop, split anpr/trips into groups
    for group, group_df in groups:
        # Progress update
        j += 1
        log_progress = (j in checkpoints)

        if parallel:
            log_data = {}

            if log_progress:
                log_data['log'             ] = True
                log_data['groups_processed'] = j
                log_data['groups_total'    ] = ngroups
                log_data['checkpoint'      ] = np.argwhere(checkpoints==j)[0][0]
            else:
                log_data['log'] = False

            # the actual job submission
            job_id = rdisplacement.remote(group_df,
                                          log_data,
                                          buffer_size)
            jobs.append(job_id)
        else:
            # the actual computation
            newdf = displacement(group_df, buffer_size)
            dfs.append(newdf)

            if log_progress:
                log("Checkpoint {}: {}/{} groups processed (~{}%)."\
                        .format(np.argwhere(checkpoints==j)[0][0],
                                j, ngroups, j/ngroups*100),
                    level = lg.INFO)


    if parallel:
        # the actual computation
        dfs.extend(ray.get(jobs))
        # shutdown
        if shutdown_ray:
            ray.shutdown()

    # not all dfs have the same columns
    # (to_keep does not have the new displacement columns)
    displacements = pd.concat(dfs, sort = False).sort_index()

    log("Computed displacements for all observations in {:,.2f} sec"\
            .format(time.time() - start_time),
        level = lg.INFO)

    return displacements

def displacement(df, buffer_size = 100):
    """
    Calculate displacement of vehicles travelling from A to B.
    """
    newdf = df.reset_index(drop = False)
    size = len(newdf)

    dps = np.zeros([size], dtype = np.uint16)
    dns = np.zeros([size], dtype = np.uint16)

    for i, row in newdf.iterrows():
        # bound df so that we don't run expensive query on the entire group dataframe
        # pandas doesn't throw out of bounds error so it's alright
        # to use out of bound indexes in either direction
        wdf = newdf.loc[(i - buffer_size):(i + buffer_size)]

        dps[i] = np.sum((wdf.t_origin      > row["t_origin"]) & \
                        (wdf.t_destination < row["t_destination"]))

        dns[i] = np.sum((wdf.t_origin      < row["t_origin"]) & \
                        (wdf.t_destination > row["t_destination"]))

    newdf = newdf.assign(dp = dps)
    newdf = newdf.assign(dn = dns)
    newdf = newdf.set_index('index')

    return newdf

@ray.remote
def rdisplacement(df, log_data, buffer_size = 100):
    """
    Calculate displacement of vehicles travelling from A to B.
    """
    newdf = df.reset_index(drop = False)
    size = len(newdf)

    dps = np.zeros([size], dtype = np.uint16)
    dns = np.zeros([size], dtype = np.uint16)

    for i, row in newdf.iterrows():
        # bound df so that we don't run expensive query on the entire group
        # dataframe pandas doesn't throw out of bounds error so it's alright
        # to use out of bound indexes in either direction
        wdf = newdf.loc[(i - buffer_size):(i + buffer_size)]

        dps[i] = np.sum((wdf.t_origin      > row["t_origin"]) & \
                        (wdf.t_destination < row["t_destination"]))

        dns[i] = np.sum((wdf.t_origin     < row["t_origin"]) & \
                        (wdf.t_destination > row["t_destination"]))

    newdf = newdf.assign(dp = dps)
    newdf = newdf.assign(dn = dns)
    newdf = newdf.set_index('index')

    if log_data['log']:
            q = log_data['groups_processed']/log_data['groups_total']*100
            log("Checkpoint {}: {}/{} groups processed (~{}%)."\
                    .format(log_data['checkpoint'],
                            log_data['groups_processed'],
                            log_data['groups_total'],
                            q),
                level = lg.INFO)

    return newdf

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

    unique_ods = flows.index.levels[0]

    if not skip_explicit:
        # We want every combination of origin,destination,period to show up in the
        # flows, even if the flow is zero. This is useful later, for calculations
        # and makes 'missing' data explicit.
        log("Filling missing combinations of (od,period) with zero flows.",
            level = lg.INFO)

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

        log("Total process memory: {:,.1f} MB [AFTER REINDEX]"\
                .format(process.memory_info().rss/1e6),
            level = lg.INFO)

        expected_nrows = len(periods) * len(unique_ods)

        log(("Expected rows: {} (nperiods . unique_ods = {} . {}). "
             "Observed rows: {}.")\
                .format(expected_nrows, len(periods), len(unique_ods), len(flows)),
            level = lg.INFO)

        assert len(flows) == expected_nrows

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
