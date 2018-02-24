from collections import deque
import numpy as np
from pathlib import  Path
import pandas as pd
import json




def coerce_to_int(num,name):
    if np.isclose(num,int(num)):
        num = int(num)
    else:
        raise ValueError("%s must be an integer or a rounded float"%name)
    return num


def gen_slice_timings(tr, nslices, nvolumes=1, pattern='alt+z'):
    """Generate slice timings for all slices collected in some number of volumes.

    Parameters
    ----------
    tr: float
        Repitition Time in same units as desired for output
    nslices: int, float
        Number of slices collected in each tr
    nvolumes: int, float  optional
        Number of volumes you would like slice timings for
    pattern: string, one of ('altplus', 'alt+z', 'alt+z2', 'altminus', 'alt-z',
                             'alt-z2', 'seq+z', 'seqplus', 'seq-z', seqminus')
        The slice sampling pattern, names are taken from AFNI nomenclature

    Returns
    -------
    output: list of floats or np.nan
        List of floats for slice timing in same units as tr.
        np.nan is returned if any of the arguments are nan.
    """
    if any(np.isnan([tr,nslices,nvolumes])):
        return np.nan
    tr = coerce_to_int(tr,"tr")
    nslices = coerce_to_int(nslices,"nslices")
    nvolumes = coerce_to_int(nvolumes,"nvolumes")

    ordered_times = [tt for tt in np.linspace(0, tr, nslices+1)][:-1]
    middle = int((nslices % 2) + len(ordered_times)/2)
    first_half = ordered_times[:middle]
    second_half = ordered_times[middle:]

    if pattern == 'altplus' or pattern == 'alt+z':
        tr_timings = np.zeros(nslices)
        tr_timings[::2] = first_half
        tr_timings[1::2] = second_half
    elif pattern == 'alt+z2':
        tr_timings = np.zeros(nslices)
        tr_timings[::2] = first_half
        tr_timings[1::2] = second_half
        tr_timings = deque(tr_timings)
        tr_timings.rotate(1)
        tr_timings = np.array(tr_timings)
    elif pattern == 'altminus' or pattern == 'alt-z':
        tr_timings = np.zeros(nslices)
        tr_timings[::2] = first_half
        tr_timings[1::2] = second_half
        tr_timings = tr_timings[::-1]
    elif pattern == 'alt-z2':
        tr_timings = np.zeros(nslices)
        tr_timings[::2] = first_half
        tr_timings[1::2] = second_half
        tr_timings = deque(tr_timings)
        tr_timings.rotate(1)
        tr_timings = np.array(tr_timings)[::-1]
    elif pattern == 'seq+z' or pattern == 'seqplus':
        tr_timings = ordered_times
    elif pattern == 'seq-z' or pattern == 'seqminus':
        tr_timings = ordered_times[::-1]
    else:
        raise NotImplementedError('Pattern %s is not implemented yet'%pattern)

    output = np.hstack([np.array(tr_timings) + n for n in range(nvolumes)])
    output = list(float(str(np.round(oo, 6))) for oo in output)
    return output



def write_json(path, data):
    path = Path(path)
    path.write_text(json.dumps(data, indent=0))


def _get_fields(row, data, fieldnames):
    for f_idx, field in enumerate(fieldnames):
        tmp = data
        if isinstance(field, tuple):

            for ii, key in enumerate(field):
                    try:
                        if ii == (len(field) - 1):
                            # tmp[key] = values_to_set[f_idx]
                            row = row.append(pd.Series({'_'.join(field): tmp[key]}))
                        else:
                            tmp[key]
                            tmp = tmp[key]
                    except KeyError:
                        row = row.append(pd.Series({'_'.join(field): np.nan}))
        else:
            try:
                row = row.append(pd.Series({field: data[field]}))
            except KeyError:
                row = row.append(pd.Series({field: np.nan}))

    return row


def _set_fields(data, fieldnames, values_to_set):

    for f_idx, field in enumerate(fieldnames):
        tmp = data
        if isinstance(field, tuple):

            for ii, key in enumerate(field):
                    try:
                        if ii == (len(field) - 1):
                            tmp[key] = values_to_set[f_idx]
                        else:
                            tmp[key]
                            tmp = tmp[key]
                    except KeyError:
                            pass
        else:
            data[field] = values_to_set[f_idx]

    return data


def _del_fields(j_in, fieldnames):
    """
    Remove fields from json data inplace.
    Each field name should be  a tuple.
    """
    for field in fieldnames:
        tmp = j_in
        if isinstance(field, tuple):
            for i, l in enumerate(field):
                try:
                    if i == (len(field) - 1):
                        del tmp[l]
                    else:
                        tmp[l]
                        tmp = tmp[l]
                except KeyError:
                        pass
#                         print(f, ' not found as a json field')
        else:
            try:
                del tmp[field]
            except KeyError:
                        pass
#                         print(f, ' not found as a json field')


def modify_json_fields(row,json_col='json_path',fieldnames=None,action='get',
values_to_set=None):
    """
    For mapping operations with json files across dataframe rows.

    Parameters
    ----------
    row: pd.Series
        Dataframe row containing a column that has a path to a json.
    json_col: string
        Name of Dataframe column that contains the path to an appropriate json
    fieldnames: tuple,str or a list of either/both
       One or more json fieldnames to perform an action with. Nested
        fieldnames should be expressed as a tuple of strings.
    action: string, one of ('get', 'set', 'delete')
        'get' will return an updated row with the retrieved fieldnames. 'set'
    will write to the provided fieldnames in the json file. 'delete' removes
    the provided fieldnames from the json file.
    values_to_set: str,iterable
        Must match fieldnames in number. Can only be used when action is 'set'.

     Returns
    -------
    row: for get the returned row will be modified

    Example usage:
    df.apply(lambda row: modify_json_fields(row,
                         fieldnames = ['SliceTiming',('primary_field','AcquisitionTime')],
                         action = 'set',
                         values_to_set = [[0,0.2,0.4,0.6],"15.20.00"])
    """
    assert fieldnames is not None
    assert action in ['delete', 'get', 'set']

    json_path = Path(row[json_col])
    with json_path.open() as j:
        data = json.load(j)

    if action == 'get':
        row = _get_fields(row, data, fieldnames)
    if action == 'set':
        data = _set_fields(data, fieldnames, values_to_set)
        write_json(json_path, data)
    if action == 'delete':
        _del_fields(data, fieldnames)
        write_json(json_path, data)

    return row

