import numpy as np
import warnings


def genhurst(S, q):

    L = len(S)
    if L < 100:
        warnings.warn("Data series very short!")

    H = np.zeros((len(range(5, 20)), 1))
    k = 0

    for Tmax in range(5, 20):

        x = np.arange(1, Tmax + 1, 1)
        mcord = np.zeros((Tmax, 1))

        for tt in range(1, Tmax + 1):
            dV = S[np.arange(tt, L, tt)] - S[np.arange(tt, L, tt) - tt]
            VV = S[np.arange(tt, L + tt, tt) - tt]
            N = len(dV) + 1
            X = np.arange(1, N + 1)
            Y = VV
            mx = np.sum(X, dtype=np.float64) / N
            SSxx = np.sum(X**2, dtype=np.float64) - N * mx**2
            my = np.sum(Y, dtype=np.float64) / N
            SSxy = np.sum(np.multiply(X, Y), dtype=np.float64) - N * mx * my
            cc1 = SSxy / SSxx
            cc2 = my - cc1 * mx
            ddVd = dV - cc1
            VVVd = VV - np.multiply(cc1, np.arange(1, N + 1, dtype=np.float64)) - cc2
            mcord[tt - 1] = np.mean(np.abs(ddVd) ** q, dtype=np.float64) / np.mean(np.abs(VVVd) ** q, dtype=np.float64)

        mx = np.mean(np.log10(x), dtype=np.float64)
        SSxx = np.sum(np.log10(x) ** 2, dtype=np.float64) - Tmax * mx**2
        my = np.mean(np.log10(mcord), dtype=np.float64)
        SSxy = np.sum(np.multiply(np.log10(x), np.transpose(np.log10(mcord))), dtype=np.float64) - Tmax * mx * my
        H[k] = SSxy / SSxx
        k = k + 1

    mH = np.mean(H, dtype=np.float64) / q

    return mH


def is_flatline(sig):
    cond = np.where(np.diff(sig.copy()) != 0.0, np.nan, True)
    if np.isnan(cond).any():
        return False
    else:
        return True


def HurstD_index(dico_signal, name_lead, fs):
    H_lead = {}
    H_array = np.array([])
    for i in name_lead:
        if is_flatline(dico_signal[i]):
            H_lead[i] = (2, dico_signal[i])
            H_array = np.append(H_array, 2)
        else:
            H = genhurst(dico_signal[i], 1)
            H_lead[i] = (2 - H, dico_signal[i])
            H_array = np.append(H_array, 2 - H)
    return H_lead, np.mean(H_array)
