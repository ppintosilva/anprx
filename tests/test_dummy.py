import pytest
import anprx

def test_imports():
    import osmnx

def test_dummy_True():
    assert anprx.dummy_equals(1,1)

def test_dummy_False():
    assert not anprx.dummy_equals(1,2)
