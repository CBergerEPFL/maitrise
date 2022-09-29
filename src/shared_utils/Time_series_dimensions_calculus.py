import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import desolver.backend as D


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


def Lm_q(signal, m, k, fs):
    N = len(signal)
    n = np.floor((N - m) / k).astype(np.int64)
    norm = ((N - 1) / (n * k)) / (k * (1 / fs))
    sum = np.sum(np.abs(np.diff(signal[m::k], n=1)))
    Lmq = sum * norm
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


def TSD_plot(dico_lead, name_lead, segment_length, fs):

    D_lead = {}
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
        D_lead[i] = Ds

    w_length = [w * segment_length for w in range(0, int((len(t) / fs) * (1 / segment_length)))]

    for i in name_lead:
        plt.plot(w_length, D_lead[i], label=i)
    plt.xlabel("Time interval")
    plt.ylabel("TSD value")
    plt.legend(loc="best", bbox_to_anchor=(1, 1))
    plt.grid()
    plt.show()


def TSD_mean_calculator(signal, segment_length=100, dt=0.01):
    w = 1
    Ds = np.array([])
    while (w * segment_length) <= len(signal):
        sig_c = signal[int((w - 1) * segment_length) : int((w) * segment_length)]
        L1 = Lq_k(sig_c, 1, 1 / dt)
        L2 = Lq_k(sig_c, 2, 1 / dt)
        Dv = (np.log(L1) - np.log(L2)) / (np.log(2))
        Ds = np.append(Ds, Dv)
        w += 1
    return np.mean(Ds), np.std(Ds)


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
