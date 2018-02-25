from collections import deque
import numpy as np
from pathlib import  Path
import pandas as pd
import json
import platform




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


def host_is_hpc(sim=False, host_simulated="helix.nih.gov"):
    if sim:
        host = host_simulated
    else:
        try:
            host = platform.node()
        except:
            host = ""
    if any(h in host for h in ['biowulf', 'helix', 'felix']):
        hpc = True
    else:
        hpc = False

    return hpc


def test_host_is_hpc():
    assert host_is_hpc(sim=True) == True
    assert host_is_hpc(sim=True, host_simulated="a_host_name") == False


def test_heud_call():
    row = pd.Series({'dicom_template': "the_template",
                     "bids_subj": "the_subj", "bids_ses": "the_sess"})
    # make_heud_call(row,project_dir,output_dir,container_image,heuristics_script=None,conv_dir=None,\
    #                              anon_script=None,conversion=False,minmeta=False, \
    #                     overwrite=True,debug=False,dev=False,use_scratch=False):

    cmds = make_heud_call(row=row,
                         project_dir=Path.cwd(),
                         output_dir=Path.cwd(),
                         container_image=Path('sing_path'),
                         conversion=False, minmeta=False,
                         overwrite=True,
                         debug=False,
                         dev=False,
                         use_scratch=False)
    assert False



def make_heud_call(*, row=None, project_dir=None, output_dir=None,
                   container_image=None, **kwargs):
    """
    Returns command for executing heudiconv in a container.

    Parameters
    ----------
    row: pd.Series
        Dataframe row containing a column that has a path to a json.
    project_dir: pathlib.Path,string
        Should be the base directory (absolute path) of all projects work.
    output_dir: pathlib.Path,string
       Path to output directory relative to the project_dir.
    container_image: pathlib.Path,string
        Path to heudiconv singularity image.
    heuristics_script: None,
    conv_dir: None,
    anon_script: None,
    conversion: bool, default False
        Use dcm2niix to convert the dicoms.
    minmeta: bool, default False
        Remove the majority of the metadata.
    overwrite: bool, default True
        Overwrite previous output
    debug: bool, default False
        Drop into pdb upon error.
    dev: bool, default False
        Mount heudiconv code repo to container.
    use_scratch: bool, default False
        Mount /lscratch/$SLURM_JOB_ID to containers tmp directory.

     Returns
    -------
    row: for get the returned row will be modified

    Example usage:
    df.apply(lambda row: modify_json_fields(row,
                         fieldnames = ['SliceTiming',('primary_field','AcquisitionTime')],
                         action = 'set',
                         values_to_set = [[0,0.2,0.4,0.6],"15.20.00"])
    """


    default_for_kwargs = {
        "heuristics_script": None,
        "conv_dir": None,
        "anon_script": None,
        "conversion": False,
        "minmeta": False,
        "overwrite": True,
        "debug": False,
        "dev": False,
        "use_scratch": False
    }
    default_for_kwargs.update(kwargs)
    print(default_for_kwargs)
    return default_for_kwargs

    if not heuristics_script:
        # use heuristics script inside container
        heuristics_script = '/src/heudiconv/heuristics/convertall.py'
    else:
        heuristics_script = Path(
            '/data').joinpath(heuristics_script).as_posix()
    if host_is_hpc():
        cmd = 'module load singularity;'
    else:
        cmd = ''

    project_dir = Path(project_dir).as_posix()
    output_dir = Path(output_dir).as_posix()
#     cmd = cmd + \
#     'module load Anaconda;source deactivate;' +\
    cmd += \
        ' singularity exec' + \
        ' --bind ' + project_dir + ':/data'

    if dev:
        if host_is_hpc():
            cmd += ' --bind /home/rodgersleejg/Documents/code/heudiconv_project/heudiconv/heudiconv:/opt/conda/envs/neuro/lib/python2.7/site-packages/heudiconv'
        else:
            assert False

    if use_scratch:
        cmd += ' --bind /lscratch/$SLURM_JOB_ID:/tmp'
    else:
        print('Not using scratch.')
        cmd += ' --bind /tmp:/tmp'

    cmd += ' ' + container_image.as_posix() + \
        " bash -c 'source activate neuro; /neurodocker/startup.sh heudiconv" + \
        ' -d ' + row.dicom_template + \
        ' -s ' + row.bids_subj + \
        ' -ss ' + row.bids_ses + \
        ' -f ' + heuristics_script + \
        ' -b'

    if overwrite:
        cmd += ' --overwrite'
    if output_dir is not None:
        output_dir = Path(output_dir).as_posix()
        cmd += ' -o ' + output_dir

    if conv_dir is not None:
        conv_dir = Path(conv_dir).as_posix()
        cmd += ' --conv-outdir ' + conv_dir

    if debug:
        cmd += ' --dbg'

    if minmeta:
        cmd += ' --minmeta'

    if conversion:
        cmd += ' -c dcm2niix'
    else:

        cmd += ' -c none'
    cmd += "'"
    return cmd


def _assemble_cmd_list(cmd_list, kwargs, kwarg_dict):

    return cmd_list


def test_assemble_cmd_list():
    return True
    assert _assemble_cmd_list(cmd_list,)
