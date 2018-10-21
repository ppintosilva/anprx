import anprx
import pytest

def test_nominatim_search_address():
    expected = \
        [37899441,
         461119586,
         4725926,
         4692270,
         4655478,
         2544439,
         31992849]

    result = anprx.search_address("Newcastle A186 Westgate Rd")

    assert expected == result

def test_nominatim_search_address_fail():
    result = anprx.search_address("AAAAAAAAAAA")

    assert result == []
