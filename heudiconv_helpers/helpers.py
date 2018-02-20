from collections import deque
import numpy as np


def gen_slice_timings(tr, nslices, nvolumes=1, pattern='alt+z'):
    """Generate slice timings for all slices collected in some number of volumes.

    Parameters
    ----------
    tr: float
        Repitition Time in same units as desired for output
    nslices: int
        Number of slices collected in each tr
    nvolumes: int, optional
        Number of volumes you would like slice timings for
    pattern: string, one of ('altplus', 'alt+z', 'alt+z2', 'altminus', 'alt-z',
                             'alt-z2', 'seq+z', 'seqplus', 'seq-z', seqminus')
        The slice sampling pattern, names are taken from AFNI nomenclature

    Returns
    -------
    output: list of floats
        List of floats for slice timing in same units as tr
    """

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
