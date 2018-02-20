from heudiconv_helpers import gen_slice_timings


def test_gen_slice_timings():
    assert gen_slice_timings(1, 5, 1, 'alt+z') == [0.0, 0.6, 0.2, 0.8, 0.4]
    assert gen_slice_timings(1, 5, 1, 'altplus') == [0.0, 0.6, 0.2, 0.8, 0.4]
    assert gen_slice_timings(1, 5, 1, 'alt+z2') == [0.4, 0.0, 0.6, 0.2, 0.8]
    assert gen_slice_timings(1, 5, 1, 'alt-z') == [0.4, 0.8, 0.2, 0.6, 0.0]
    assert gen_slice_timings(1, 5, 1, 'altminus') == [0.4, 0.8, 0.2, 0.6, 0.0]
    assert gen_slice_timings(1, 5, 1, 'alt-z2') == [0.8, 0.2, 0.6, 0.0, 0.4]
    assert gen_slice_timings(1, 5, 1, 'seqplus') == [0.0, 0.2, 0.4, 0.6, 0.8]
    assert gen_slice_timings(1, 5, 1, 'seqminus') == [0.8, 0.6, 0.4, 0.2, 0.0]
