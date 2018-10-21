################################################################################
# Module: utils.py
# Description: Global settings, configuration, logging and caching
# License: MIT, see full license in LICENSE.txt
# Author: Pedro Pinto da Silva
# Web: https://github.com/pedroswits/pydummy
################################################################################
# Based on: Geoff Boeing's OSMnx package
# https://github.com/gboeing/osmnx/blob/master/osmnx/utils.py
################################################################################

import io
import os
import sys
import time
import json
import hashlib
import inspect
import unicodedata
import osmnx as ox
import logging as lg
import datetime as dt

###
###

settings = {
    "app_name" : "anprx",

    "app_folder" : os.path.expanduser("~/.anprx"),
    "data_folder_name" : "data",
    "logs_folder_name" : "logs",
    "cache_folder_name" : "cache",
    "images_folder_name" : "images",

    "log_to_file" : True,
    "log_to_console" : False,
    "cache_http" : True,

    "log_default_level" : lg.INFO,

    "default_user_agent" : "Python anprx package (https://github.com/pedroswits/anprx)",
    "default_referer" : "Python anprx package (https://github.com/pedroswits/anprx)",
    "default_accept_language" : "en"
}
"""
anprx's global settings.
"""

###
###

def init_osmnx():
    """
    Configure osmnx's settings to match anprx's settings.
    """

    osmnx_folder = os.path.join(settings["app_folder"], "osmnx")
    if not os.path.exists(osmnx_folder):
        os.makedirs(osmnx_folder)

    ox.config(
        data_folder = os.path.join(osmnx_folder, "data"),
        logs_folder = os.path.join(osmnx_folder, "logs"),
        imgs_folder = os.path.join(osmnx_folder, "images"),
        cache_folder = os.path.join(osmnx_folder, "cache"),
        use_cache = True,
        log_file = True,
        log_console = False)    

###
###
# Run this here globally so that we don't have to run it everytime before we call an osmnx statement.

init_osmnx()

###
###

class InvalidSetting(ValueError):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)

class ImmutableSetting(ValueError):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)

###
###

def config(**kwargs):
    """
    Configure anprx's settings

    Parameters
    ---------
    **kwargs
        keyword arguments that exist in settings

    Raises
    ------
    InvalidSetting
        if keyword argument is not in settings

    Returns
    -------
    None
    """

    for key, value in kwargs.items():
        if not key in settings:
            raise InvalidSetting("No such setting: {}".format(key))

        if key == "app_name":
            raise ImmutableSetting("App name should not be changed.")

        if key in {"app_folder", "logs_folder_name"}:
            clean_logger()

        settings[key] = value
        log('Config: {} = {}'.format(key, value))

###
###

def create_folders(app_folder = None,
                   logs_folder_name = None,
                   data_folder_name = None,
                   cache_folder_name = None,
                   images_folder_name = None):
    """
    Creates app folders: parent, data, logs and cache

    Parameters
    ----------
    app_folder : string
        location of main app directory

    logs_folder_name : string
        name of folder containing logs

    data_folder_name : string
        name of folder containing data

    cache_folder_name : string
        name of folder containing cached http responses

    images_folder_name : string
        name of folder containing saved images

    Returns
    -------
    None
    """
    if app_folder is None:
        app_folder = settings["app_folder"]
    if logs_folder_name is None:
        logs_folder_name = settings["logs_folder_name"]
    if data_folder_name is None:
        data_folder_name = settings["data_folder_name"]
    if cache_folder_name is None:
        cache_folder_name = settings["cache_folder_name"]
    if images_folder_name is None:
        images_folder_name = settings["images_folder_name"]

    if not os.path.exists(app_folder):
        os.makedirs(app_folder)

    init_osmnx()

    logs_folder = os.path.join(app_folder, logs_folder_name)

    if not os.path.exists(logs_folder):
        os.mkdir(logs_folder)

    data_folder = os.path.join(app_folder, data_folder_name)

    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    cache_folder = os.path.join(app_folder, cache_folder_name)

    if not os.path.exists(cache_folder):
        os.mkdir(cache_folder)

    images_folder = os.path.join(app_folder, images_folder_name)

    if not os.path.exists(images_folder):
        os.mkdir(images_folder)

###
###

def make_str(value):
    """
    Convert a passed-in value to unicode if Python 2, or string if Python 3.

    Parameters
    ----------
    value : any
        the value to convert to unicode/string

    Returns
    -------
    unicode or string
    """
    try:
        # for python 2.x compatibility, use unicode
        return unicode(value)
    except NameError:
        # python 3.x has no unicode type, so if error, use str type
        return str(value)

###
###

def log(message,
        level = None,
        name = settings["app_name"],
        filename = settings["app_name"]):
    """
    Write a message to the log file and/or print to the the console.

    Parameters
    ----------
    message : string
        the content of the message to log

    level : int
        one of the logger.level constants

    name : string
        name of the logger

    filename : string
        name of the log file

    Returns
    -------
    None
    """
    if level is None:
        level = settings["log_default_level"]

    func = inspect.currentframe().f_back.f_code

    # if logging to file is turned on
    if settings["log_to_file"]:
        # get the current logger (or create a new one, if none), then log
        # message at requested level
        logger = get_logger(level=level, name=name, filename=filename)

        complete_message = "{:>17} -> {:27} {}"\
            .format(
                os.path.basename(func.co_filename) +
                ':' + str(func.co_firstlineno),
                func.co_name + '()',
                message)

        if level == lg.DEBUG:
            logger.debug(complete_message)
        elif level == lg.INFO:
            logger.info(complete_message)
        elif level == lg.WARNING:
            logger.warning(complete_message)
        elif level == lg.ERROR:
            logger.error(complete_message)

    # if logging to console is turned on, convert message to ascii and print to
    # the console
    if settings["log_to_console"]:
        # capture current stdout, then switch it to the console, print the
        # message, then switch back to what had been the stdout. this prevents
        # logging to notebook - instead, it goes to console
        standard_out = sys.stdout
        sys.stdout = sys.__stdout__

        # convert message to ascii for console display so it doesn't break
        # windows terminals
        message = unicodedata.normalize('NFKD', make_str(message)).encode('ascii', errors='replace').decode()
        print(message)
        sys.stdout = standard_out

###
###

def get_logger(level = None,
               name = settings["app_name"],
               filename = settings["app_name"]):
    """
    Create a logger or return the current one if already instantiated.

    Parameters
    ----------
    level : int
        one of the logger.level constants

    name : string
        name of the logger

    filename : string
        name of the log file

    Returns
    -------
    logger.logger
    """

    if level is None:
        level = settings["log_default_level"]

    logger = lg.getLogger(name)

    # if a logger with this name is not already set up
    if not getattr(logger, 'is_set', None):
        # if the logs folder does not already exist, create it
        create_folders()

        # get today's date and construct a log filename
        todays_date = dt.datetime.today().strftime('%Y_%m_%d')
        log_filename = os.path.join(settings["app_folder"], settings["logs_folder_name"], '{}_{}.log'.format(filename, todays_date))

        # create file handler and log formatter and set them up
        handler = lg.FileHandler(log_filename, encoding='utf-8')
        formatter = lg.Formatter('%(asctime)s %(levelname)10s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.is_set = True

    return logger

###

def clean_logger(name = settings["app_name"]):
    """
    Removes all handlers associated with a given logger

    Parameters
    ----------
    name : string
        name of the logger

    Returns
    -------
    logger.logger
    """

    logger = lg.getLogger(name)

    handlers = logger.handlers
    for handler in handlers:
        logger.removeHandler(handler)

    logger.is_set = False

    return logger

###
###

def save_to_cache(url, response_json):
    """
    Save an HTTP response json object to the cache.

    If the request was sent to server via POST instead of GET, then URL should be a GET-style representation of request. Users should always pass OrderedDicts instead of dicts of parameters into request functions, so that the parameters stay in the same order each time, producing the same URL string, and thus the same hash. Otherwise the cache will eventually contain multiple saved responses for the same request because the URL's parameters appeared in a different order each time.

    Parameters
    ----------
    url : string
        the url of the request

    response_json : dict
        the json response

    Returns
    -------
    None
    """
    if settings["cache_http"]:
        if response_json is None:
            log('Saved nothing to cache because response_json is None')
        else:
            # create the folder on the disk if it doesn't already exist
            create_folders()

            # hash the url (to make filename shorter than the often extremely
            # long url)
            filename = hashlib.md5(url.encode('utf-8')).hexdigest()
            cache_path_filename = os.path.join(settings["app_folder"], settings["cache_folder_name"], os.extsep.join([filename, 'json']))

            # dump to json, and save to file
            json_str = make_str(json.dumps(response_json))
            with io.open(cache_path_filename, 'w', encoding='utf-8') as cache_file:
                cache_file.write(json_str)

            log('Saved response to cache file "{}"'.format(cache_path_filename))

###
###

def get_from_cache(url):
    """
    Retrieve a HTTP response json object from the cache.

    Parameters
    ----------
    url : string
        the url of the request

    Returns
    -------
    dict
        response_json
    """
    # if the tool is configured to use the cache
    if settings["cache_http"]:
        # determine the filename by hashing the url
        filename = hashlib.md5(url.encode('utf-8')).hexdigest()

        cache_path_filename = os.path.join(settings["app_folder"], settings["cache_folder_name"], os.extsep.join([filename, 'json']))
        # open the cache file for this url hash if it already exists, otherwise
        # return None
        if os.path.isfile(cache_path_filename):
            with io.open(cache_path_filename, encoding='utf-8') as cache_file:
                response_json = json.load(cache_file)
            log('Retrieved response from cache file "{}" for URL "{}"'.format(cache_path_filename, url))
            return response_json

###
###

def get_http_headers(user_agent=None, referer=None, accept_language=None):
    """
    Update the default requests HTTP headers with OSMnx info.

    Parameters
    ----------
    user_agent : str
        the user agent string, if None will set with OSMnx default

    referer : str
        the referer string, if None will set with OSMnx default

    accept_language : str
        make accept-language explicit e.g. for consistent nominatim result sorting

    Returns
    -------
    dict
        headers
    """

    if user_agent is None:
        user_agent = settings["default_user_agent"]
    if referer is None:
        referer = settings["default_referer"]
    if accept_language is None:
        accept_language = settings["default_accept_language"]

    headers = requests.utils.default_headers()
    headers.update({'User-Agent': user_agent, 'referer': referer, 'Accept-Language': accept_language})
    return headers


def save_fig(fig,
             axis,
             filename,
             file_format = 'png',
             dpi = 300):
    """
    Save a figure to disk.

    Parameters
    ----------
    fig : figure

    axis : axis

    filename : str
        name of the file

    file_format : str
        format of the file (e.g. 'png', 'jpg', 'svg')

    dpi : int
        resolution of the image file

    Returns
    -------
    None
    """
    start_time = time.time()

    if not filename:
        raise ValueError("Please define a filename")

    path_filename = os.path.join(
        settings["app_folder"],
        settings["images_folder_name"],
        os.extsep.join([filename, file_format]))

    if file_format == 'svg':
        # if the file_format is svg, prep the fig/ax a bit for saving
        axis.axis('off')
        axis.set_position([0, 0, 1, 1])
        axis.patch.set_alpha(0.)
        fig.patch.set_alpha(0.)
        fig.savefig(path_filename, bbox_inches=0, format=file_format, facecolor=fig.get_facecolor(), transparent=True)

    else:
        extent = 'tight'

        fig.savefig(path_filename,
                    dpi=dpi,
                    bbox_inches=extent,
                    format=file_format,
                    facecolor=fig.get_facecolor(),
                    transparent=True)

    log('Saved the figure to disk in {:,.2f} seconds'\
            .format(time.time()-start_time),
        level = lg.INFO)
