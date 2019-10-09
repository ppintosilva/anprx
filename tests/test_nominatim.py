import pytest
import anprx.nominatim as nominatim
import anprx.exceptions as exceptions

def test_nominatim_search_address():
    result = nominatim.search_address("Newcastle A186 Westgate Rd")

    assert len(result) > 0

def test_nominatim_search_address_fail():
    with pytest.raises(exceptions.EmptyResponseError):
        result = nominatim.search_address("AAAAAAAAAAA")

osmids = [37899441,
          461119586,
          4725926,
          4692270,
          4655478,
          2544439,
          31992849]

def test_lookup_address_exceeds_limit():
    with pytest.raises(ValueError):
        nominatim.lookup_address(osmids * 10, entity = 'W')

def test_lookup_address_invalid_osm_entity():
    with pytest.raises(ValueError):
        nominatim.lookup_address(osmids, entity = 'A')

def test_lookup_address():
    details = nominatim.lookup_address(osmids, entity = 'W')

    assert len(details) <= len(osmids)

    for dict_ in details:
        assert len(dict_.keys()) > 0
        assert {'road', 'importance', 'type', 'suburb'}.issubset(set(dict_.keys()))
