import anprx
import pytest

def test_nominatim_search_address():
    result = anprx.search_address("Newcastle A186 Westgate Rd")

    assert len(result) > 0

def test_nominatim_search_address_fail():
    result = anprx.search_address("AAAAAAAAAAA")

    assert result == []
