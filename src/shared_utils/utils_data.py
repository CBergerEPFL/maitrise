import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from petastorm import make_reader
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)


def format_data_to_xarray(data_path: str, save_path: str | None = None):
    """Format data to xarray format.

    Args:
        data_path (str): Path to data.
        save_path (str): Path to save data. If save_path is not None, data will be saved to save_path.

    Returns:
        ds_ecg (xarray): Data in xarray format.
    """
    if (save_path is not None) and (not os.path.exists(save_path)):
        os.makedirs(save_path)

    if "file://" not in data_path:
        path_petastorm = f"file:///{data_path}"
    else:
        path_petastorm = data_path
    # Load data
    array_signal = []
    array_name = []
    array_quality = []
    array_fs = []
    array_sex = []
    with make_reader(path_petastorm) as reader:
        for idx, sample in enumerate(reader):
            if idx == 0:
                lead_names = sample.signal_names.astype(str)
            array_signal.append(sample.signal)
            array_name.append(sample.noun_id.decode("utf-8"))
            array_quality.append(sample.signal_quality.decode("utf-8"))
            array_fs.append(sample.sampling_frequency)
            array_sex.append(sample.sex.decode("utf-8"))

    ds_ecg = xr.Dataset(
        data_vars=dict(
            signal=(["id", "time", "lead_name"], np.array(array_signal)),
            data_quality=(["id"], np.array(array_quality)),
            fs=(["id"], np.array(array_fs)),
            sex=(["id"], np.array(array_sex)),
        ),
        coords=dict(
            id=(["id"], np.array(array_name)),
            time=(["time"], np.arange(0, 5000)),
            lead_names=(["lead_names"], lead_names),
        ),
        attrs=dict(description="ecg with quality description"),
    )

    if save_path is not None:
        ds_ecg.to_netcdf(os.path.join(save_path,"ecg_data.nc"))

    return ds_ecg

def format_data_to_xarray_2020(data_path: str, save_path: str | None = None):
    """Format data to xarray format.This is the same function as before but adapted for the Classification of 12-leads ECGgs ; the physionet/computing in Cardiology Challenge 2020” dataset

    Args:
        data_path (str): Path to data.
        save_path (str): Path to save data. If save_path is not None, data will be saved to save_path.

    Returns:
        ds_ecg (xarray): Data in xarray format.
    """
    if (save_path is not None) and (not os.path.exists(save_path)):
        os.makedirs(save_path)

    if "file://" not in data_path:
        path_petastorm = f"file:///{data_path}"
    else:
        path_petastorm = data_path
    # Load data
    array_signal = []
    array_name = []
    array_fs = []
    array_sex = []
    with make_reader(path_petastorm) as reader:
        for idx, sample in enumerate(reader):
            if idx == 0:
                lead_names = sample.signal_names.astype(str)
            if len(sample.signal[:,0])!=5000:
                continue

            array_signal.append(sample.signal)
            array_name.append(sample.noun_id.decode("utf-8"))
            array_fs.append(sample.sampling_frequency)
            array_sex.append(sample.sex.decode("utf-8"))

    ds_ecg = xr.Dataset(
        data_vars=dict(
            signal=(["id", "time", "lead_name"], np.array(array_signal)),
            fs=(["id"], np.array(array_fs)),
            sex=(["id"], np.array(array_sex)),
        ),
        coords=dict(
            id=(["id"], np.array(array_name)),
            time=(["time"], np.arange(0, 5000)),
            lead_names=(["lead_names"], lead_names),
        ),
        attrs=dict(description="ecg with pathologies description"),
    )

    if save_path is not None:
        ds_ecg.to_netcdf(os.path.join(save_path,"ecg_data_2020.nc"))

    return ds_ecg
