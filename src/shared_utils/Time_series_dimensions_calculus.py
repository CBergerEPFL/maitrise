import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import desolver.backend as D
import hfda
from math import isnan


def system_coordinates_reader(Path_to_data, Attractor_name, num_attractor=0):
    path = Path_to_data + f"/{Attractor_name}_attractors"
    df = pd.read_csv(path + f"/{Attractor_name}__{num_attractor}.csv")
    df_n = df.to_numpy()
    xyzs = df_n[:, 1:4]
    t = df_n[:, 0]
    return xyzs, t


def Mean_taux(signal, taux, hprime, h=0):
    N = len(signal)
    if taux + hprime + h < N and taux + h < N - 1:
        return np.mean((signal[int(h + taux) : int(taux + hprime + h)]))
    elif taux + hprime + h > N and taux + h < N - 1:
        return np.mean((signal[int(h + taux) : -1]))
    else:
        return 0


def Variance_taux(signal, taux, hprime, h=0):

    N = len(signal)
    if taux + hprime + h <= N and taux + h < N - 1:
        # return (1/hprime)*np.sum((signal[int(h+taux):int(taux+hprime+h)]-Mean_taux(signal,taux,hprime,h))**2)
        return np.var(signal[int(h + taux) : int(taux + hprime + h)])
    elif taux + hprime + h > N and taux + h < N - 1:
        # return (1/hprime)*np.sum((signal[int(taux):-1]-Mean_taux(signal,taux,hprime,0))**2)
        return np.var(signal[int(taux) : -1])
    else:
        return 0


def I1(c, signal, fs, tab, h, hprime, step_c, t0=0):
    if t0 > c or t0 < 0:
        return print("error : t0 is outside the range you want to calculate")
    elif c > len(signal) or c < 0:
        return "error : c is outside the range of your signal"
    else:
        if len(tab) == 0:
            I1c = (
                (1 / (h * len(signal)))
                * step_c
                * np.abs(
                    Mean_taux(signal, t0 * fs, hprime * len(signal), h * len(signal))
                    - Mean_taux(signal, t0 * fs, hprime * len(signal))
                )
            )
        else:
            I1c = tab[-1]
            I1c += (
                (1 / (h * len(signal)))
                * step_c
                * np.abs(
                    Mean_taux(signal, c * fs, hprime * len(signal), h * len(signal))
                    - Mean_taux(signal, c * fs, hprime * len(signal))
                )
            )

    return I1c


def I2(c, signal, fs, tab, h, hprime, step_c, t0=0):
    if t0 > c or t0 < 0:
        return "error : t0 is outside the range you want to calculate"
    elif c > len(signal) or c < 0:
        return "error : c is outside the range of your signal"
    else:
        if len(tab) == 0:
            I2c = (
                (1 / (h * len(signal)))
                * step_c
                * np.abs(
                    Variance_taux(signal, t0 * fs, hprime * len(signal), h * len(signal))
                    - Variance_taux(signal, t0 * fs, hprime * len(signal))
                )
            )
        else:
            I2c = tab[-1]
            I2c += (
                (1 / (h * len(signal)))
                * step_c
                * np.abs(
                    Variance_taux(signal, c * fs, hprime * len(signal), h * len(signal))
                    - Variance_taux(signal, c * fs, hprime * len(signal))
                )
            )

    return I2c


def discrepancies_mean_curve(signal_tot, fs, h, hprime, step, t0=0):
    c = np.arange(t0, (len(signal_tot) / fs) + t0, step)
    I1_val = np.array([])
    I2_val = np.array([])
    for j in c:
        if (
            j * fs + hprime * len(signal_tot) + h * len(signal_tot) > len(signal_tot)
            and j * fs + h * len(signal_tot) > len(signal_tot) - 1
        ):
            c = c[c < j]
            break
        else:
            I1_val = np.append(I1_val, I1(j, signal_tot, fs, I1_val, h, hprime, step, t0))
            I2_val = np.append(I2_val, I2(j, signal_tot, fs, I2_val, h, hprime, step, t0))
    return I1_val, I2_val, c


def Interval_calculator_all(dico_signal, name_signal, fs, t0=0):
    h = 0.001
    hprime = 0.005
    dic_segment_lead = {}
    for i in name_signal:
        I1c, I2c, c = discrepancies_mean_curve(dico_signal[i], fs, h, hprime, 1 / fs, t0)
        # c1 = c[np.isclose(I1c, [np.max(I1c[I1c !=np.nan]) / 2], atol=0.001)]
        # c2 = c[np.isclose(I2c, [np.max(I2c[I2c !=np.nan]) / 2], atol=0.001)]
        c1 = c[I1c < 0.1]
        c2 = c[I2c < 0.1]
        cs = np.minimum(np.mean(c1), np.mean(c2))
        dic_segment_lead[i] = (cs - t0) * fs
    return dic_segment_lead


def Interval_calculator_lead(signal, fs, t0=0):
    h = 0.001
    hprime = 0.005
    dic_segment_lead = {}
    I1c, I2c, c = discrepancies_mean_curve(signal, fs, h, hprime, 1 / fs, t0)
    c1 = c[np.isclose(I1c, [np.max(I1c[I1c != np.nan]) / 2], atol=0.001)]
    c2 = c[np.isclose(I2c, [np.max(I2c[I2c != np.nan]) / 2], atol=0.001)]
    cs = np.minimum(np.mean(c1), np.mean(c2))
    dic_segment_lead = (cs - t0) * fs
    return dic_segment_lead


def TSD_index(dico_signal, name_lead, fs, t0=0):

    ###Index Creation :TSD
    ###The label will be as follow : mean(TSD) < 1.25 = Acceptable;mean(SDR of all lead) >1.25 = Unacceptable
    ##For each lead, we will return a more precise classification based on the folloying rules:
    ## TSD<1.25 = Good quality ; 1.25<TSD<1.40 = Medium quality; TSD>1.4 = Bad quality
    # dico_seg = Interval_calculator(dico_signal,name_lead,fs,t0)
    dico_D = {}
    D_arr = np.array([])
    for i in name_lead:
        if is_flatline(dico_signal[i]):
            dico_D[i] = (2, dico_signal[i])
            D_arr = np.append(D_arr, 2)
        else:
            Dv, _ = TSD_mean_calculator(dico_signal[i], 1 / fs)
            dico_D[i] = (Dv, dico_signal[i])
            D_arr = np.append(D_arr, Dv)
    return dico_D, np.mean(D_arr)


def flatline_sig(sig):
    series = sig.copy()
    index = np.where(np.diff(series) != 0.0, False, True)
    index = np.append(index, False)
    copy_sig = sig.copy()
    return copy_sig[index != False]


def is_flatline(sig):
    cond = np.where(np.diff(sig.copy()) != 0.0, np.nan, True)
    if np.isnan(cond).any():
        return False
    else:
        return True


def TSD_index_lead(signal, fs, t0=0):

    ###Index Creation :TSD for 1 lead
    ###The label will be as follow : mean(TSD) < 1.25 = Acceptable;mean(SDR of all lead) >1.25 = Unacceptable
    ##For each lead, we will return a more precise classification based on the folloying rules:
    ## TSD<1.25 = Good quality ; 1.25<TSD<1.40 = Medium quality; TSD>1.4 = Bad quality
    # dico_seg = Interval_calculator(dico_signal,name_lead,fs,t0)
    t = np.linspace(t0, int(len(signal) / fs), len(signal))
    if is_flatline(signal):
        return 2
    else:
        Dv, _ = TSD_mean_calculator(signal, 1 / fs)
        return Dv


def Lm_q(signal, m, k, fs):
    N = len(signal)
    n = np.floor((N - m) / k).astype(np.int64)
    norm = (N - 1) / (n * k * (1 / fs))
    sum = np.sum(np.abs(np.diff(signal[m::k], n=1)))
    Lmq = (sum * norm) / k
    return Lmq


def Lq_k(signal, k, fs):
    calc_L_series = np.frompyfunc(lambda m: Lm_q(signal, m, k, fs), 1, 1)
    L_average = np.average(calc_L_series(np.arange(1, k + 1)))
    return L_average


def Dq(signal, kmax, fs):
    calc_L_average_series = np.frompyfunc(lambda k: Lq_k(signal, k, fs), 1, 1)

    k = np.arange(1, kmax + 1)
    L = calc_L_average_series(k).astype(np.float64)

    D, _ = -np.polyfit(np.log2(k), np.log2(L), 1)

    return D


def TSD_plot(dico_lead, name_lead, fs):

    D_lead = {}
    for i in name_lead:
        sig = dico_lead[i]
        segment_length = Interval_calculator_lead(sig, fs)
        if isnan(segment_length) or segment_length < 100:
            print("WARNING : Segment Length = 100")
            segment_length = 1000
        else:
            print("Optimal Segment Length : ", segment_length)

        X = np.c_[[sig[int((w - 1)) : int((w) + segment_length)] for w in range(1, int(len(sig) - segment_length))]]

        L1 = np.array([Lq_k(X[i, :], 1, fs) for i in range(X.shape[0])])
        L2 = np.array([Lq_k(X[i, :], 2, fs) for i in range(X.shape[0])])
        Ds = (np.log(L1) - np.log(L2)) / (np.log(2))

        # while (w + int(segment_length)) <= len(sig):
        #     sig_c = sig[int((w - 1)) : int((w) + segment_length)]
        #     L1 = Lq_k(sig_c, 1, fs)
        #     L2 = Lq_k(sig_c, 2, fs)
        #     Dv = (np.log(L1) - np.log(L2)) / (np.log(2))
        #     Ds = np.append(Ds, Dv)
        #     w += 1
        D_lead[i] = Ds

    for i in name_lead:
        plt.figure()
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 15))
        w_length = range(0, len(D_lead[i]))
        ax[0].plot(w_length / fs, D_lead[i], label=i)
        ax[0].set_title(f"TSD time Evolution of Lead {i.decode('utf8')}")
        ax[0].set_xlabel("Time (sec)")
        ax[0].set_ylabel("TSD value")
        ax[0].grid()
        ax[1].plot(np.linspace(0, int(len(dico_lead[i]) / fs), len(dico_lead[i])), dico_lead[i], label=i)
        ax[1].set_title(f"Lead {i.decode('utf8')}")
        ax[1].set_xlabel("Time (sec)")
        ax[1].set_ylabel("Voltage Amplitude")
        ax[1].grid()
        plt.show()


def TSD_mean_calculator(signal, dt=0.01):
    segment_length = Interval_calculator_lead(signal, 1 / dt)
    if isnan(segment_length) or segment_length < 100:
        segment_length = 1000
    X = np.c_[[signal[int((w - 1)) : int((w) + segment_length)] for w in range(1, int(len(signal) - segment_length))]]
    L1 = np.array([Lq_k(X[i, :], 1, 1 / dt) for i in range(X.shape[0])])
    L2 = np.array([Lq_k(X[i, :], 2, 1 / dt) for i in range(X.shape[0])])
    Ds = (np.log(L1) - np.log(L2)) / (np.log(2))
    return np.mean(Ds), np.std(Ds)


def add_observational_noise(sig, SNR):
    Power_sig = (1 / len(sig)) * np.sum(np.abs(sig) ** 2, dtype=np.float64)
    P_db = 10 * np.log10(Power_sig)
    noisedb = P_db - SNR
    sd_db_watts = 10 ** (noisedb / 10)
    # sd_noise = np.sqrt(Power_sig/(SNR))
    noise = np.random.normal(0, np.sqrt(sd_db_watts), len(sig))
    sig_noisy = sig + noise
    return sig_noisy


def TSDvsNoiseLevel_array(noise_level, path_to_data, list_attractor=["lorenz", "rossler"]):
    Dmean = {}
    SD_D = {}

    for i in noise_level:
        if i == 0:
            for name in list_attractor:
                mid_Dmean = np.array([])
                mid_SD = np.array([])
                for j in range(0, len(os.listdir(path_to_data + f"/{name}_attractors"))):
                    coord, _ = system_coordinates_reader(path_to_data, name, j)
                    Obs = coord[:, 0]
                    Mean_TSD, SD_TSD = TSD_mean_calculator(Obs, 100)
                    mid_Dmean = np.append(mid_Dmean, Mean_TSD)
                    mid_SD = np.append(mid_SD, SD_TSD)
                Dmean[name] = np.array([np.mean(mid_Dmean)])
                SD_D[name] = np.array([np.mean(mid_SD)])

        else:
            for name in list_attractor:
                mid_Dmean = np.array([])
                mid_SD = np.array([])
                for j in range(0, len(os.listdir(path_to_data + f"/{name}_attractors"))):
                    coord, _ = system_coordinates_reader(path_to_data, name, j)
                    Obs = coord[:, 0]
                    noise_obs = add_observational_noise(Obs, 1 / i)
                    Mean_TSD, SD_TSD = TSD_mean_calculator(noise_obs, 100)
                    mid_Dmean = np.append(mid_Dmean, Mean_TSD)
                    mid_SD = np.append(mid_SD, SD_TSD)
                Dmean[name] = np.append(Dmean[name], np.mean(mid_Dmean))
                SD_D[name] = np.array(SD_D[name], np.mean(mid_SD))

    return Dmean, SD_D


def plt_TSDvsNoise(noise_lev, path_to_data, attractors_sel):
    Great_mean, Great_SD = TSDvsNoiseLevel_array(noise_lev, path_to_data, attractors_sel)
    fig, ax = plt.subplots(len(attractors_sel) - 1, 2, figsize=(20, 10))
    for i, j in zip(attractors_sel, range(len(attractors_sel))):
        ax[j].errorbar(noise_lev, Great_mean[i], Great_SD[i])
        ax[j].set_xlabel("Noise level")
        ax[j].set_ylabel("mean TSD value")
        ax[j].set_title(f"TSD vs noise level for {i} system")
        ax[j].set_ylim([1.9, 2.1])
        ax[j].grid()

    plt.figure()
    for i in attractors_sel:
        plt.plot(noise_lev, Great_mean[i])
    plt.legend([i for i in attractors_sel])
    plt.title("Mean TSD value evolution with noise level for both system")
    plt.xlabel("Noise level")
    plt.ylabel("mean TSD value")
    plt.ylim([1.9, 2.1])
    plt.grid()
    plt.show()


def TSD_ECG(dico_lead, name_lead, segment_length, fs):

    D = np.array([])
    for i in name_lead:
        w = 1
        Ds = np.array([])
        sig = dico_lead[i]
        while (w * segment_length * fs) <= len(sig):
            sig_c = sig[int((w - 1) * segment_length * fs) : int((w) * segment_length * fs)]
            L1 = Lq_k(sig_c, 1, fs)
            L2 = Lq_k(sig_c, 2, fs)
            Dv = (np.log(L1) - np.log(L2)) / (np.log(2))
            Ds = np.append(Ds, Dv)
            w += 1
        D = np.append(D, np.mean(Ds))

    return D


def RMS(tab_val):
    N = len(tab_val)
    # print("la taille : ",N)
    square_sum = 0
    for j in range(len(tab_val)):
        for i in range(j + 1, len(tab_val)):
            if (j + 1) == len(tab_val):
                break
            else:
                square_sum += np.abs(tab_val[j] - tab_val[i]) ** 2
    norm = 1 / (N**2 - N)
    rms_val = np.sqrt(norm * square_sum)
    return rms_val


def RMS_array_creator(dico_signal, name, c_ax, fs):
    RMS_val = np.array([])
    for j in c_ax:
        # print("le temps : ",j)
        D = TSD_ECG(dico_signal, name, j, fs)
        RMS_val = np.append(RMS_val, RMS(D))

    return RMS_val


def plt_TSDvsdyn_Noise(dico_attractor, noise_lev, attractors_sel, n_simulation):
    Great_mean, Great_SD = TSDvsNoiseLevel_dyn(dico_attractor, attractors_sel, noise_lev, n_simulation)
    fig, ax = plt.subplots(len(attractors_sel) - 1, 2, figsize=(20, 10))
    for i, j in zip(attractors_sel, range(len(attractors_sel))):
        ax[j].plot(noise_lev, Great_mean[i], "ob")
        ax[j].errorbar(noise_lev, Great_mean[i], Great_SD[i])
        ax[j].set_xlabel("Noise level")
        ax[j].set_ylabel("mean TSD value")
        ax[j].set_title(f"TSD vs noise level for {i} system with Dynamical noise")
        ax[j].grid()

    plt.figure()
    for i in attractors_sel:
        plt.plot(noise_lev, Great_mean[i])
    plt.legend([i for i in attractors_sel])
    plt.title("Mean TSD value evolution with noise level for both system with Dynamical noise")
    plt.xlabel("Noise level")
    plt.ylabel("mean TSD value")
    plt.grid()
    plt.show()
