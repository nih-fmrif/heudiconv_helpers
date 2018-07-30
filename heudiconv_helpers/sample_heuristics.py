# coding: utf-8
# sub-<participant_label>[_ses-<session_label>]_task-<task_label>[_acq-<label>][_rec-<label>][_run-<index>][_echo-<index>]_bold.nii[.gz]
import os
from collections import namedtuple
import logging
lgr = logging.getLogger('heudiconv')


def infotoids(seqinfos, outdir):
    # decide on subjid and session based on patient_id
    lgr.info("Processing sequence infos to deduce study/session")

    subject = get_unique(seqinfos, 'patient_id')
    locator = 'none_defined_yet'

    return {
        # TODO: request info on study from the JedCap
        'locator': locator,
        # Sessions to be deduced yet from the names etc TODO
        'session': 'not_working_yet',
        'subject': subject,
    }


def get_unique(seqinfos, attr):
    """Given a list of seqinfos, which must have come from a single study
    get specific attr, which must be unique across all of the entries

    If not -- fail!

    """
    values = set(getattr(si, attr) for si in seqinfos)
    assert (len(values) == 1)
    return values.pop()


def filter_dicom(dcmdata):
    """Return True if a DICOM dataset should be filtered out, else False"""
    comments = getattr(dcmdata, 'ImageComments', '')
    if len(comments):
        if 'reference volume' in comments.lower():
            print("Filter out image with comment '%s'" % comments)
            return True
    return False
    # Another format:return True if dcmdata.StudyInstanceUID in dicoms2skip else False


def filter_files(fn):
    """
    This is used by heudiconv to filter files based on the filename.
    The function returns a boolean for a given filename.
    """
    patterns_to_filter_out = ['README',
                              'requisition',
                              'realtime',
                              'edti',
                              'optional',
                              '3 plane loc',
                              'ASSET cal',
                              'clinical',
                              'plane_loc',
                              'nihpcasl']
#                               'rest_assetEPI']
    return all(pat not in fn for pat in patterns_to_filter_out)


def create_key(template, outtype=('nii.gz'), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes


def infotodict(seqinfo, test_heuristics=False):
    # sometimes seqinfo is a list of seqinfo objects, sometimes it is a seqinfo object
    if not hasattr(seqinfo, 'keys'):
        seqinfo_dict = {'no_grouping': seqinfo}
    else:
        seqinfo_dict = seqinfo

    mprage = create_key(
        'sub-{subject}/{session}/anat/sub-{subject}_{session}_acq-mprage_run-{item:03d}_T1w')
#     dti = create_key(
#         'sub-{subject}/{session}/dwi/sub-{subject}_{session}_run-{item:03d}_dwi')
    rest = create_key(
        'sub-{subject}/{session}/func/sub-{subject}_{session}_task-rest_run-{item:03d}_bold')
    audition = create_key(
        'sub-{subject}/{session}/func/sub-{subject}_{session}_task-audition_run-{item:03d}_bold')

    info = {mprage: [], rest: [], audition: []}
    heurs = {

        "('MPRAGE' in seq.series_description.upper().replace('_','').replace('-','').replace(' ',''))": "info[mprage].append([seq.series_id])",

        "('rest_assetEPI' in seq.series_description)": "info[rest].append([seq.series_id])",
        "('Audition fmri' in seq.series_description)": "info[audition].append([seq.series_id])",

        #         "('edti_cdiflist' in seq.series_description)": "info[dti].append([seq.series_id])",
    }

    if test_heuristics:
        for group_id, seqinfo in seqinfo_dict.items():
            for seq in seqinfo:
                for criterion, action in heurs.items():
                    eval(criterion)
                    eval(action)
                print("The defined heuristics evaluate")
                return None
    for group_id, seqinfo in seqinfo_dict.items():
        if len(seqinfo) > 30:
            print("There are a lot of entries provided here (%s)."
                  " This heuristic file does not handle duplicate"
                  " series_id across the same accession_number."
                  " This can be avoided by passing subject/session"
                  " combinations individually to heudiconv" % len(seqinfo))
        for seq in seqinfo:
            for criterion, action in heurs.items():
                if eval(criterion):
                    eval(action)
                    break
        #

    return info
