import numpy as np
from pathlib import Path
import h5py
import os
from torch.utils.data import DataLoader, Dataset
import pickle
import pprint

pp = pprint.PrettyPrinter()


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def extract_pos_s21_offset(
    file_list,
    file_list_TEST,
    data_file_name,
    cyl_length,  # cylinder dependent
    MAX_POS=None,  # dataset dependent
    MIN_POS=None,  # dataset dependent
    use_complex=False,
    use_print=True,
    use_save=True,
):

    """
    extract positions and d21 from all files, and then scale the positions

    # offset rule: (for v5 LEG19B_Testzyl_12910433)
    # 1. if max_pos is around cyl_length, then use it as standard
    #    as for small positive min_pos,
    #       if simple subtraction of the diff between max_pos and cyl_length doesn't lead to negative min_pos, then use it
    #       otherwise, consider scaling to make sure the range for pos is [0, cyl_length]
    #    as for negative min_pos,
    #       consider scaling directly to make sure the range for pos is [0, cyl_length]

    # 2. if max position is far off cyl_length, then first check outlier and then apply 1.

    """

    pos = []
    s21 = []
    dict_all = {}

    j = 0
    k = 0

    for i, file in enumerate(file_list + file_list_TEST):

        filename = file.split(os.sep)[-1][:-5]
        h51 = h5py.File(file, "r")

        _pos = np.array(h51["pos"])

        _s21_ = np.array(h51["s21"], dtype=np.complex64)

        # change to (real,imag,real,imag) instead of (real,...,real,imag,...,imag) as before
        if not use_complex:
            n = _s21_.shape[0]
            f = _s21_.shape[1] * 2
            _s21 = np.zeros((n, f), dtype=np.float32)
            _s21[:, 0::2] = _s21_.real.astype(np.float32)
            _s21[:, 1::2] = _s21_.imag.astype(np.float32)
        else:
            _s21 = _s21_

        ## ignore outliers
        if MAX_POS:
            mask = _pos < MAX_POS  # ignore position that is bigger than usual
            _pos = _pos[mask]
            _s21 = _s21[mask, :]

        if MIN_POS:
            mask = _pos > MIN_POS  # ignore position that is bigger than usual
            _pos = _pos[mask]
            _s21 = _s21[mask, :]

        # offset
        _diff = _pos.max() - cyl_length
        if abs(_diff) < 3:  # TODO: check if this is reasonable
            # if max_pos is around cyl_length, then use it as standard

            # if simple subtraction of the diff between max_pos and cyl_length doesn't lead to negative min_pos, then use it
            if _pos.min() - _diff >= 0:
                _pos = _pos - _diff
            # otherwise, consider scaling to make sure the range for pos is [0, cyl_length]
            else:
                _pos = (_pos - _pos.min()) / (_pos.max() - _pos.min()) * cyl_length

        pos.append(_pos)
        s21.append(_s21)

        _dict = {}
        _dict["filename"] = filename
        _dict["length"] = _pos.shape[0]
        dict_all[i] = _dict

    ## extract pos and s21 from file_list_TEST and append them to the end

    pos = np.hstack(pos)
    s21 = np.vstack(s21)

    ## rounding and change dtype
    pos = np.around(pos * 1000, decimals=1).astype(np.float32)

    if use_print:
        print("position:")
        print(pos.shape)
        print("s21:")
        print(s21.shape)
        pp.pprint(dict_all)

    ## save data
    if use_save:
        dict_raw = {}
        dict_raw["pos"] = pos
        dict_raw["s21"] = s21
        dict_raw["lengths"] = [dict_all[i]["length"] for i in range(len(dict_all))]
        dict_raw["filenames"] = [dict_all[i]["filename"] for i in range(len(dict_all))]

        with open(data_file_name, "wb") as f:
            pickle.dump(dict_raw, f, pickle.HIGHEST_PROTOCOL)
        print("data saved")


def extract_pos_s21(
    file_list,
    file_list_TEST,
    data_file_name,
    MAX_POS=None,
    MIN_POS=None,
    use_complex=False,
    use_print=True,
    use_save=True,
):

    """
    extract positions and d21 from all files
    """

    pos = []
    s21 = []
    dict_all = {}

    j = 0
    k = 0
    ## extract pos and s21 from file_list and file_list_TEST
    for i, file in enumerate(file_list + file_list_TEST):

        filename = file.split(os.sep)[-1][:-5]
        h51 = h5py.File(file, "r")

        _pos = np.array(h51["pos"])

        _s21_ = np.array(h51["s21"], dtype=np.complex64)

        # change to (real,imag,real,imag) instead of (real,...,real,imag,...,imag) as before
        if not use_complex:
            n = _s21_.shape[0]
            f = _s21_.shape[1] * 2
            _s21 = np.zeros((n, f), dtype=np.float32)
            _s21[:, 0::2] = _s21_.real.astype(np.float32)
            _s21[:, 1::2] = _s21_.imag.astype(np.float32)
        else:
            _s21 = _s21_

        ## ignore outliers
        if MAX_POS:
            mask = _pos < MAX_POS  # ignore position that is bigger than usual
            _pos = _pos[mask]
            _s21 = _s21[mask, :]

        if MIN_POS:
            mask = _pos > MIN_POS  # ignore position that is smaller than usual
            _pos = _pos[mask]
            _s21 = _s21[mask, :]

        pos.append(_pos)
        s21.append(_s21)

        _dict = {}
        _dict["filename"] = filename
        _dict["length"] = _pos.shape[0]
        dict_all[i] = _dict

    pos = np.hstack(pos)
    s21 = np.vstack(s21)

    ## rounding and change dtype
    pos = np.around(pos * 1000, decimals=1).astype(np.float32)

    if use_print:
        print("position:")
        print(pos.shape)
        print("s21:")
        print(s21.shape)
        pp.pprint(dict_all)

    ## save data
    if use_save:
        dict_raw = {}
        dict_raw["pos"] = pos
        dict_raw["s21"] = s21
        dict_raw["lengths"] = [dict_all[i]["length"] for i in range(len(dict_all))]
        dict_raw["filenames"] = [dict_all[i]["filename"] for i in range(len(dict_all))]

        with open(data_file_name, "wb") as f:
            pickle.dump(dict_raw, f, pickle.HIGHEST_PROTOCOL)
        print("data saved")


def load_dict(data_file_name):
    with open(data_file_name, "rb") as f:
        dict_ = pickle.load(f)
    return dict_


def load_pos_s21(data_file_name, n):
    # load data, and ust the last n files for test

    with open(data_file_name, "rb") as f:
        dict_raw = pickle.load(f)
    pos = dict_raw["pos"]
    s21 = dict_raw["s21"]
    lengths = dict_raw["lengths"]
    filenames = dict_raw["filenames"]

    # extract data just for test
    l = sum(lengths[-n:])
    pos_TEST = pos[-l:]
    s21_TEST = s21[-l:, :]
    pos = pos[:-l]
    s21 = s21[:-l, :]

    return pos, s21, pos_TEST, s21_TEST


def save_pre(data_file_name, scaler, idx_train, idx_test, RANDOM_STATE):

    """save preprocessed data for NN"""

    dict_NN_prep = {}
    dict_NN_prep["scaler"] = scaler
    dict_NN_prep["idx_train"] = idx_train
    dict_NN_prep["idx_test"] = idx_test
    dict_NN_prep["RANDOM_STATE"] = RANDOM_STATE

    # save data
    with open(data_file_name, "wb") as f:
        pickle.dump(dict_NN_prep, f, pickle.HIGHEST_PROTOCOL)


def load_pre(data_file_name):
    """load preprocessed data for NN"""

    with open(data_file_name, "rb") as f:
        dict_NN_prep = pickle.load(f)

    scaler = dict_NN_prep["scaler"]
    idx_train = dict_NN_prep["idx_train"]
    idx_train = dict_NN_prep["idx_train"]
    idx_test = dict_NN_prep["idx_test"]
    RANDOM_STATE = dict_NN_prep["RANDOM_STATE"]

    return scaler, idx_train, idx_test, RANDOM_STATE


# PreprocessedDataset
class PreprocessedDataset(Dataset):
    """Load data and labels"""

    def __init__(
        self,
        features,
        labels,
        # mean,
        # std
    ):
        "Initialization"
        self.features = features
        self.labels = labels
        # self.mean = mean
        # self.std = std

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.features)

    def __getitem__(self, index):
        "Generates one sample of data"
        x = self.features[index, :]
        y = self.labels[index]
        ## Normalize the data
        # x -= self.mean
        # x /= self.std
        return x, y


def get_dataloader(is_train, batch_size, x, y):
    ds = PreprocessedDataset(
        x,
        y,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True if is_train else False,
    )
    return dl
