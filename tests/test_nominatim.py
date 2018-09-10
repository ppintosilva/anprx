import anprx
import pytest

def test_nominatim_lookup_ways():
    expected = \
        ['37899441',
         '461119586',
         '4725926',
         '4692270',
         '4655478',
         '2544439',
         '31992849']

    result = anprx.lookup_ways("Newcastle A186 Westgate Rd")

    assert expected == result

def test_nominatim_lookup_ways_fail():
    result = anprx.lookup_ways("AAAAAAAAAAA")

    assert result == []
