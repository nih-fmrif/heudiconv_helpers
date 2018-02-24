import pandas as pd
import numpy as np
import pytest
from pandas.util.testing import assert_series_equal

from heudiconv_helpers import gen_slice_timings
from heudiconv_helpers.helpers import _set_fields
from heudiconv_helpers.helpers import _get_fields
from heudiconv_helpers.helpers import _del_fields


def test_gen_slice_timings():
    assert gen_slice_timings(1, 5, 1, 'alt+z') == [0.0, 0.6, 0.2, 0.8, 0.4]
    assert gen_slice_timings(1, 5, 1, 'altplus') == [0.0, 0.6, 0.2, 0.8, 0.4]
    assert gen_slice_timings(1, 5, 1, 'alt+z2') == [0.4, 0.0, 0.6, 0.2, 0.8]
    assert gen_slice_timings(1, 5, 1, 'alt-z') == [0.4, 0.8, 0.2, 0.6, 0.0]
    assert gen_slice_timings(1, 5, 1, 'altminus') == [0.4, 0.8, 0.2, 0.6, 0.0]
    assert gen_slice_timings(1, 5, 1, 'alt-z2') == [0.8, 0.2, 0.6, 0.0, 0.4]
    assert gen_slice_timings(1, 5, 1, 'seqplus') == [0.0, 0.2, 0.4, 0.6, 0.8]
    assert gen_slice_timings(1, 5, 1, 'seqminus') == [0.8, 0.6, 0.4, 0.2, 0.0]
    # returns nan if arguments are nan
    assert np.isnan(gen_slice_timings(1, 5, np.nan, 'seqminus'))
    assert np.isnan(gen_slice_timings(np.nan, 5, 1, 'seqminus'))
    assert np.isnan(gen_slice_timings(1, np.nan, 1, 'seqminus'))
    # turns float arguments to int if they are rounded
    assert gen_slice_timings(1.0, 5.0, 1.0, 'seqminus') == [0.8, 0.6, 0.4, 0.2, 0.0]
    with pytest.raises(ValueError):
    	gen_slice_timings(1.9, 5.1, 1.0, 'seqminus') == [0.8, 0.6, 0.4, 0.2, 0.0]


def test_del_fields():
    foo = {'a': {'b': {'c': 20}}, 'd': 30}
    fieldnames = [('a', 'b', 'c')]
    _del_fields(foo, fieldnames)
    assert foo == {'a': {'b': {}}, 'd': 30}

    foo = {'a': {'b': {'c': 20}}, 'd': 30}
    fieldnames = [('a', 'b', 'c'), 'd']
    _del_fields(foo, fieldnames)
    assert foo == {'a': {'b': {}}}

    foo = {'a': {'b': {'c': 20}}, 'd': 30}
    fieldnames = ['d']
    _del_fields(foo, fieldnames)
    assert foo == {'a': {'b': {'c': 20}}}

    foo = {'a': {'b': {'c': 20}}, 'word': 30}
    fieldnames = ['word']
    _del_fields(foo, fieldnames)
    assert foo == {'a': {'b': {'c': 20}}}

    foo = {'a': {'b': {'c': 20}}, 'd': 30}
    fieldnames = [('a', 'd')]
    # some times fields are missing, this can be ignored
    _del_fields(foo, fieldnames)
    assert foo == {'a': {'b': {'c': 20}}, 'd': 30}


def test_set_fields():
    data = {'a': {'b': {'c': 20}}, 'd': 30}
    fieldnames = [('d')]
    data = _set_fields(data, fieldnames, [20])
    assert data == {'a': {'b': {'c': 20}}, 'd': 20}

    data = {'a': {'b': {'c': 20}}, 'd': 30}
    fieldnames = ['d']
    data = _set_fields(data, fieldnames, [20])
    assert data == {'a': {'b': {'c': 20}}, 'd': 20}

    data = {'a': {'b': {'c': 20}}, 'd': 30}
    fieldnames = [('a', 'b', 'c')]
    data = _set_fields(data, fieldnames, [50])
    assert data == {'a': {'b': {'c': 50}}, 'd': 30}

    data = {'a': {'b': {'c': 20}}, 'd': 30}
    fieldnames = [('a', 'b', 'c'), 'd']
    data = _set_fields(data, fieldnames, [50, 'hello'])
    assert data == {'a': {'b': {'c': 50}}, 'd': 'hello'}


def test_get_fields():
    row = pd.Series({'path': 'a/path'})
    data = {'a': {'b': {'c': 20}}, 'd': 30}

    fieldnames = ['d']
    row = _get_fields(row, data, fieldnames)
    assert_series_equal(row.sort_index(),
                        pd.Series({'path': 'a/path',
                                  'd': 30}).
                        sort_index())

    row = pd.Series({'path': 'a/path'})
    fieldnames = [('a', 'b', 'c')]
    row = _get_fields(row, data, fieldnames)
    assert_series_equal(row.sort_index(),
                        pd.Series({'path': 'a/path',
                                  'a_b_c': 20}).
                        sort_index())

    row = pd.Series({'path': 'a/path'})
    fieldnames = [('a', 'b', 'c'), 'd']
    row = _get_fields(row, data, fieldnames)
    assert_series_equal(row.sort_index(),
                        pd.Series({'path': 'a/path',
                                  'a_b_c': 20,
                                   'd': 30}).
                        sort_index())

    row = pd.Series({'path': 'a/path'})
    fieldnames = [('a', 'b', 'd'), 'e']
    row = _get_fields(row, data, fieldnames)
    assert_series_equal(row.sort_index(),
                        pd.Series({'path': 'a/path',
                                  'a_b_d': np.nan,
                                   'e': np.nan}).
                        sort_index())

