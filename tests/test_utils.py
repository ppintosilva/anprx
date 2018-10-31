import io
import os
import ast
import json
import shutil
import hashlib
import requests
import osmnx as ox
import logging as lg
import datetime as dt

import pytest
import anprx.utils

###

def destroy_folders():
    app_folder = anprx.utils.settings["app_folder"]

    if os.path.exists(app_folder):
        shutil.rmtree(app_folder)

    assert not os.path.exists(app_folder)

###

def assert_file_structure():
    app_folder = anprx.utils.settings["app_folder"]
    logs_folder = os.path.join(app_folder, anprx.utils.settings["logs_folder_name"])
    data_folder = os.path.join(app_folder, anprx.utils.settings["data_folder_name"])
    cache_folder = os.path.join(app_folder, anprx.utils.settings["cache_folder_name"])
    images_folder = os.path.join(app_folder,
    anprx.utils.settings["images_folder_name"])
    osmnx_folder = os.path.join(app_folder, "osmnx")

    assert os.path.exists(app_folder)
    assert os.path.exists(logs_folder)
    assert os.path.exists(data_folder)
    assert os.path.exists(cache_folder)
    assert os.path.exists(images_folder)
    assert os.path.exists(osmnx_folder)

def assert_log_file():
    log_filename = os.path.join(anprx.utils.settings["app_folder"], anprx.utils.settings["logs_folder_name"], '{}_{}.log'.format(anprx.utils.settings["app_name"], dt.datetime.today().strftime('%Y_%m_%d')))

    assert os.path.exists(log_filename)

###

def test_init_osmnx():
    app_folder = anprx.utils.settings["app_folder"]
    osmnx_folder = os.path.join(app_folder, "osmnx")

    assert ox.settings.data_folder == \
            os.path.join(osmnx_folder, "data")
    assert ox.settings.logs_folder == \
            os.path.join(osmnx_folder, "logs")
    assert ox.settings.imgs_folder == \
            os.path.join(osmnx_folder, "images")
    assert ox.settings.cache_folder == \
            os.path.join(osmnx_folder, "cache")
    assert ox.settings.use_cache
    assert ox.settings.log_file
    assert not ox.settings.log_console

###

def test_get_app_folder():
    assert anprx.utils.settings["app_folder"] == os.path.expanduser("~/.anprx")

###

def test_create_folders():
    anprx.utils.config(app_folder = "/tmp/anprx___")
    anprx.utils.create_folders()
    assert_file_structure()
    destroy_folders()

###

def test_config_immutable_setting():
    with pytest.raises(anprx.utils.ImmutableSetting):
        anprx.utils.config(app_name = "test")

###

def test_config_invalid_setting():
    with pytest.raises(anprx.utils.InvalidSetting):
        anprx.utils.config(verbose = True)

###

def test_config():
    # Changing the config generates a log entry and if no log files exist, then
    anprx.utils.config(app_folder = "/tmp/anprx")

    assert anprx.utils.settings["log_to_console"] == False
    assert anprx.utils.settings["cache_http"] == True
    assert anprx.utils.settings["log_default_level"] == lg.INFO

    assert_file_structure()
    assert_log_file()
    destroy_folders()

###

def test_cache():
    anprx.utils.config(app_folder = "/tmp/anprx",
                 cache_http = True)

    url = requests.Request('GET', "https://nominatim.openstreetmap.org/?format=json&addressdetails=1&q=Newcastle+A186+Westgate+Rd").prepare().url

    response_json = requests.get(url).json()

    anprx.utils.save_to_cache(url, response_json)

    cache_folder = os.path.join(anprx.utils.settings["app_folder"], anprx.utils.settings["cache_folder_name"])
    cache_file = os.path.join(cache_folder, os.extsep.join([hashlib.md5(url.encode('utf-8')).hexdigest(), 'json']))

    assert os.path.exists(cache_file)

    with io.open(cache_file, 'r', encoding='utf-8') as cache_file_handler:
        cache_content = json.load(cache_file_handler)

    json_str = json.dumps(response_json)
    json_str_from_cache = anprx.utils.get_from_cache(url)

    try:
        # This fails in python2 for awkward reasons
        assert json_str == json.dumps(cache_content)
    except:
        assert response_json == json_str_from_cache

    assert cache_content == json_str_from_cache

    destroy_folders()
