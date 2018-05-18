# coding: utf-8
# sub-<participant_label>[_ses-<session_label>]_task-<task_label>[_acq-<label>][_rec-<label>][_run-<index>][_echo-<index>]_bold.nii[.gz]
import os
from collections import namedtuple


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
    patterns_to_filter_out = ['README', 'requisition','realtime']
    return all(pat not in fn for pat in patterns_to_filter_out)


def create_key(template, outtype=('nii.gz'), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes


def infotodict(seqinfo, test_heuristics=False):

    if len(seqinfo) > 30:
            print("There are a lot of entries provided here (%s)." 
                  " This heuristic file does not handle duplicate"
                  " series_id across the same accession_number."
                  " This can be avoided by passing subject/session"
                  " combinations individually to heudiconv"% len(seqinfo))
    t1w = create_key(
        'sub-{subject}/{session}/anat/sub-{subject}_{session}_acq-fspgr_run-{item:03d}_T1w')
    pcasl = create_key(
        'sub-{subject}/{session}/anat/sub-{subject}_{session}_run-{item:03d}_pcasl')
    dti_ap = create_key(
        'sub-{subject}/{session}/dwi/sub-{subject}_{session}_acq-ap_run-{item:03d}_dwi')
    dti_pa = create_key(
        'sub-{subject}/{session}/dwi/sub-{subject}_{session}_acq-pa_run-{item:03d}_dwi')
    dti = create_key(
        'sub-{subject}/{session}/dwi/sub-{subject}_{session}_run-{item:03d}_dwi')
    flair_2d = create_key(
        'sub-{subject}/{session}/anat/sub-{subject}_{session}_acq-2d_run-{item:03d}_FLAIR')
    t2_star = create_key(
        'sub-{subject}/{session}/anat/sub-{subject}_{session}_run-{item:03d}_T2star')
    rest = create_key(
        'sub-{subject}/{session}/func/sub-{subject}_{session}_task-rest_run-{item:03d}_bold')
    cube_t2 = create_key(
        'sub-{subject}/{session}/anat/sub-{subject}_{session}_run-{item:03d}_T2w')
    hippo = create_key(
        'sub-{subject}/{session}/anat/sub-{subject}_{session}_acq-highreshippo_run-{item:03d}_T1w')
    flair_3d = create_key(
        'sub-{subject}/{session}/anat/sub-{subject}_{session}_acq-3d_run-{item:03d}_FLAIR')
    avdc = create_key(
        'derivatives/sub-{subject}/{session}/anat/sub-{subject}_{session}_run-{item:03d}_avdc')
    cbf = create_key(
        'derivatives/sub-{subject}/{session}/anat/sub-{subject}_{session}_run-{item:03d}_cbf')
    fa = create_key(
        'derivatives/sub-{subject}/{session}/anat/sub-{subject}_{session}_run-{item:03d}_fa')
    trace = create_key(
        'derivatives/sub-{subject}/{session}/anat/sub-{subject}_{session}_run-{item:03d}_trace')
    resting_fmap = create_key(
        'sub-{subject}/{session}/fmap/sub-{subject}_{session}_acq-resting_run-{item:03d}_fieldmap')
    dti_fmap = create_key(
        'sub-{subject}/{session}/fmap/sub-{subject}_{session}_acq-dwi_run-{item:03d}_fieldmap')

    info = {t1w: [], pcasl: [], dti_ap: [], dti_pa: [], dti: [], flair_2d: [], t2_star: [],
            rest: [], cube_t2: [], hippo: [], flair_3d: [], avdc: [], cbf: [], fa: [], trace: [],
            resting_fmap: [], dti_fmap: []}
    heurs = {

        "('FSPGR' in seq.series_description.upper())": "info[t1w].append([seq.series_id])",
        "('PCASL' in seq.series_description.upper())": "info[pcasl].append([seq.series_id])",
        "('RESHIPPO' in seq.series_description.upper())": "info[hippo].append([seq.series_id])",
        "('T2 2D' in seq.series_description.upper())": "info[flair_2d].append([seq.series_id])",
        "('T2_FLAIR' in seq.series_description.upper())": "info[flair_2d].append([seq.series_id])",
        "('Ax T2 FLAIR' in seq.series_description)": "info[flair_2d].append([seq.series_id])",
        "('T2 STAR' in seq.series_description.upper())": "info[t2_star].append([seq.series_id])",
        "('CUBE_T2' in seq.series_description.upper())": "info[cube_t2].append([seq.series_id])",
        "('3D FLAIR' in seq.series_description.upper())": "info[flair_3d].append([seq.series_id])",

        "('AXIAL RSFMRI' in seq.series_description.upper())": "info[rest].append([seq.series_id])",

        "('AVDC' in seq.series_description.upper())": "info[avdc].append([seq.series_id])",
        "('CBF' in seq.series_description.upper())": "info[cbf].append([seq.series_id])",
        "('FA' in seq.series_description.upper())": "info[fa].append([seq.series_id])",
        "('TRACE' in seq.series_description.upper())": "info[trace].append([seq.series_id])",

        "('B0 MAP - DTI' in seq.series_description.upper())": "info[dti_fmap]",
        "('B0 MAP - RSFMRI' in seq.series_description.upper())": "info[resting_fmap]",
        "('B0_MAP_RSFMRI' in seq.series_description.upper())": "info[resting_fmap]",
        "('B0 Map - rsfMRI' in seq.series_description)": "info[resting_fmap]",
        "('B0rf' == seq.sequence_name)": "info[resting_fmap]",

        "('AXIAL DTI B=1000' in seq.series_description.upper())": "info[dti_pa].append([seq.series_id])",
        "('AXIAL DTI 24VOLS FLIPPED' in seq.series_description.upper())": "info[dti_ap].append([seq.series_id])",
        "('DTI' in seq.series_description)": "info[dti].append([seq.series_id])",



    }

    if test_heuristics:
        for seq in seqinfo:
            for criterion, action in heurs.items():
                eval(criterion)
                eval(action)
            print("The defined heuristics evaluate")
            return None
    for seq in seqinfo:
        for criterion, action in heurs.items():
            if eval(criterion):
                eval(action)
                break
        #

    return info


