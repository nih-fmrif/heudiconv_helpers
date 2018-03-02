# coding: utf-8
# sub-<participant_label>[_ses-<session_label>]_task-<task_label>[_acq-<label>][_rec-<label>][_run-<index>][_echo-<index>]_bold.nii[.gz]
import os
from collections import namedtuple


def filter_files(fn):
    """
    This is used by heudiconv to filter files based on the filename.
    The function returns a boolean for a given filename.
    """
    patterns_to_filter_out = ['README', 'requisition']
    return all(pat not in fn for pat in patterns_to_filter_out)


def create_key(template, outtype=('nii.gz'), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes


def infotodict(seqinfo):

    t1w = create_key(
        'sub-{subject}/{session}/anat/sub-{subject}_{session}_acq-fspgr_run-{item:03d}_T1w')
    pcasl = create_key(
        'sub-{subject}/{session}/anat/sub-{subject}_{session}_acq-_run-{item:03d}_pcasl')
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
        'sub-{subject}/{session}/fmap/sub-{subject}_{session}_acq-highreshipporun-{item:03d}_T1w')
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

    for s in seqinfo:
        if ('Axial DTI B=1000' == s.series_description):
            info[dti_pa].append([s.series_id])
        elif ('Axial DTI 24vols flipped' == s.series_description):
            info[dti_ap].append([s.series_id])
        elif ('DTI' in s.series_description):
            info[dti].append([s.series_id])

        if ('FSPGR' in s.series_description.upper()):
            info[t1w].append([s.series_id])
        elif ('pCASL' == s.series_description):
            info[pcasl].append([s.series_id])
        elif ('ResHippo' in s.series_description):
            info[hippo].append([s.series_id])
        elif ('T2 2D' in s.series_description):
            info[flair_2d].append([s.series_id])
        elif ('T2 Star' in s.series_description):
            info[t2_star].append([s.series_id])
        elif ('CUBE_T2' in s.series_description):
            info[cube_t2].append([s.series_id])
        elif ('3D FLAIR' in s.series_description):
            info[flair_3d].append([s.series_id])

        elif ('Axial rsfMRI' in s.series_description):
            info[rest].append([s.series_id])

        elif ('AvDC' in s.series_description):
            info[avdc].append([s.series_id])
        elif ('CBF' in s.series_description):
            info[cbf].append([s.series_id])
        elif ('FA' in s.series_description):
            info[fa].append([s.series_id])
        elif ('Trace' in s.series_description):
            info[trace].append([s.series_id])

        elif ('B0 Map - DTI' == s.series_description):
            info['dti_fmap']
        elif ('B0 Map - rsfMRI' == s.series_description):
            info['resting_fmap']

    return info
