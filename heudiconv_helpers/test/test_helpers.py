import pandas as pd
import numpy as np
import pytest
from pandas.util.testing import assert_series_equal
from pathlib import Path
import json
import os.path as op

from heudiconv.utils import load_heuristic
from heudiconv_helpers.helpers import hh_load_heuristic


from heudiconv_helpers.helpers import (gen_slice_timings, make_heud_call,
                                       host_is_hpc, validate_heuristics_output,
                                       check_heuristic_script_integrity)
from heudiconv_helpers.helpers import _set_fields
from heudiconv_helpers.helpers import _get_fields
from heudiconv_helpers.helpers import _del_fields
from heudiconv_helpers.helpers import _get_outcmd
from heudiconv_helpers.helpers import __get_seqinfo


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
    assert gen_slice_timings(1.0, 5.0, 1.0, 'seqminus') == [
        0.8, 0.6, 0.4, 0.2, 0.0]
    with pytest.raises(ValueError):
        gen_slice_timings(1.9, 5.1, 1.0, 'seqminus') == [
            0.8, 0.6, 0.4, 0.2, 0.0]


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
                        "bids_subj": "the_subj",
                        "bids_ses": "the_sess"}, index=[0])
    basic_kwargs = {
        "project_dir": "proj",
        "output_dir": "outdir",
        "container_image": "img_path"}
    # Should work with a series
    cmd = make_heud_call(row=row.iloc[0, :],
                         **basic_kwargs)
    # Should work with
    with pytest.raises(ValueError):
        cmd = make_heud_call(row=row.iloc[:0, :],
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


def test_validate_heuristics_output():
    from heudiconv_helpers import helpers as hh
    heuristics_script = Path(hh.__file__).with_name(
        'sample_heuristics.py')
    validate_heuristics_output(heuristics_script=heuristics_script)


def test_validate_heuristics_output_no_arg():
    validate_heuristics_output()


def test_check_heuristic_script_integrity():
    from heudiconv_helpers import helpers as hh
    heuristics_script = Path(hh.__file__).with_name(
        'sample_heuristics.py')
    check_heuristic_script_integrity(heuristics_script=heuristics_script)

    # syntax errors in criterion should be picked up
    temp_heur = Path("heuristic_test.py")
    temp_heur.write_text(
        heuristics_script.read_text().
        replace("'Axial DTI B=1000' == ", "Axial DTI B=1000 == ")
    )
    with pytest.raises(SyntaxError):
        check_heuristic_script_integrity(heuristics_script=temp_heur)
    with pytest.raises(SyntaxError):
        check_heuristic_script_integrity(
            heuristics_script=temp_heur, test_heuristics=True)

    temp_heur.unlink()

    # silly key error mistake
    temp_heur_2 = Path("heuristic_test_2.py")
    temp_heur_2.write_text(
        heuristics_script.read_text().
        replace("info[dti_fmap]", "info['dti_fmap']")
    )
    print(heuristics_script.read_text().
          replace("""info[dti_fmap]""", """info['dti_fmap']"""))
    with pytest.raises(AttributeError):
        check_heuristic_script_integrity(
            heuristics_script=temp_heur_2.as_posix())

    with pytest.raises(KeyError):
        check_heuristic_script_integrity(
            heuristics_script=temp_heur_2, test_heuristics=True)

    check_heuristic_script_integrity(
        heuristics_script=temp_heur_2)

    temp_heur_2.unlink()


def test_load_heuristic():
    from heudiconv_helpers import helpers as hh
    HEURISTICS_PATH = Path(hh.__file__).parent
    from_file = load_heuristic(
        op.join(HEURISTICS_PATH, 'sample_heuristics.py'))

    with pytest.raises(ImportError):
        load_heuristic('unknownsomething')

    with pytest.raises(ImportError):
        load_heuristic(op.join(HEURISTICS_PATH, 'unknownsomething.py'))


def test_hh_load_heuristic():
    from heudiconv_helpers import helpers as hh
    HEURISTICS_PATH = Path(hh.__file__).parent
    from_file = hh_load_heuristic(
        op.join(HEURISTICS_PATH, 'sample_heuristics'))

    with pytest.raises(ImportError):
        hh_load_heuristic('unknownsomething')

    with pytest.raises(ImportError):
        hh_load_heuristic(op.join(HEURISTICS_PATH, 'unknownsomething.py'))
