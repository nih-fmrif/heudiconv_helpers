import pandas as pd
import numpy as np
import pytest
from pandas.util.testing import assert_series_equal
from pathlib import Path
import json

from heudiconv_helpers import gen_slice_timings, make_heud_call, host_is_hpc
from heudiconv_helpers.helpers import _set_fields
from heudiconv_helpers.helpers import _get_fields
from heudiconv_helpers.helpers import _del_fields
from heudiconv_helpers.helpers import _get_outcmd


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

def get_data_object():
    return json.loads(json.dumps({'a': {'b': {'c': 20}}, 'd': 30}))


def test_del_fields():
    foo = get_data_object()
    fieldnames = [('a', 'b', 'c')]
    _del_fields(foo, fieldnames)
    assert foo == {'a': {'b': {}}, 'd': 30}

    foo = get_data_object()
    fieldnames = [('a', 'b', 'c'), 'd']
    _del_fields(foo, fieldnames)
    assert foo == {'a': {'b': {}}}

    foo = get_data_object()
    fieldnames = ['d']
    _del_fields(foo, fieldnames)
    assert foo == {'a': {'b': {'c': 20}}}

    foo = {'a': {'b': {'c': 20}}, 'word': 30}
    fieldnames = ['word']
    _del_fields(foo, fieldnames)
    assert foo == {'a': {'b': {'c': 20}}}

    foo = get_data_object()
    fieldnames = [('a', 'd')]
    # some times fields are missing, this can be ignored
    _del_fields(foo, fieldnames)
    assert foo == {'a': {'b': {'c': 20}}, 'd': 30}


def test_set_fields():
    data = get_data_object()
    fieldnames = [('d')]
    data = _set_fields(data, fieldnames, [20])
    assert data == {'a': {'b': {'c': 20}}, 'd': 20}

    data = get_data_object()
    fieldnames = ['d']
    data = _set_fields(data, fieldnames, [20])
    assert data == {'a': {'b': {'c': 20}}, 'd': 20}

    data = get_data_object()
    fieldnames = [('a', 'b', 'c')]
    data = _set_fields(data, fieldnames, [50])
    assert data == {'a': {'b': {'c': 50}}, 'd': 30}

    data = get_data_object()
    fieldnames = [('a', 'b', 'c'), 'd']
    data = _set_fields(data, fieldnames, [50, 'hello'])
    assert data == {'a': {'b': {'c': 50}}, 'd': 'hello'}

    data = get_data_object()
    fieldnames = [('a', 'b', 'c'), 'd']
    with pytest.raises(ValueError):
        data = _set_fields(data, fieldnames, 'hello')
    


def test_get_fields():
    row = pd.Series({'path': 'a/path'})
    data = get_data_object()

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
                                  'a__b__c': 20}).
                        sort_index())

    row = pd.Series({'path': 'a/path'})
    fieldnames = [('a', 'b', 'c'), 'd']
    row = _get_fields(row, data, fieldnames)
    assert_series_equal(row.sort_index(),
                        pd.Series({'path': 'a/path',
                                  'a__b__c': 20,
                                   'd': 30}).
                        sort_index())

    row = pd.Series({'path': 'a/path'})
    fieldnames = [('a', 'b', 'd'), 'e']
    row = _get_fields(row, data, fieldnames)
    assert_series_equal(row.sort_index(),
                        pd.Series({'path': 'a/path',
                                  'a__b__d': np.nan,
                                   'e': np.nan}).
                        sort_index())


def test_host_is_hpc():
    assert host_is_hpc(sim=True) is True
    assert host_is_hpc(sim=True, host_simulated="a_host_name") is False


def test_get_outcmd():
    assert f' -o {Path.cwd()}' == _get_outcmd(Path.cwd())


def test_heud_call():
    row = pd.DataFrame({'dicom_template': "the_template",
                 "bids_subj": "the_subj", "bids_ses": "the_sess"},index = [0] )
    basic_kwargs = {
    "project_dir":"proj",
    "output_dir":"outdir",
    "container_image":"img_path"}
    # Should work with a series
    cmd = make_heud_call(row=row.iloc[0,:],
                         **basic_kwargs)
    # Should work with
    with pytest.raises(ValueError):
        cmd = make_heud_call(row=row.iloc[:0,:],
                         **basic_kwargs)

    print(cmd)


def test_heud_dev_call():
    row = pd.Series({'dicom_template': "the_template",
                     "bids_subj": "the_subj", "bids_ses": "the_sess"})
    cmd = make_heud_call(row=row,
                         project_dir="proj",
                         output_dir=Path.cwd(),
                         container_image=Path('sing_path'),
                         conversion=False, minmeta=False,
                         overwrite=True,
                         debug=True,
                         dev=True,
                         dev_dir="path_to_heudiconv",
                         use_scratch=False)


