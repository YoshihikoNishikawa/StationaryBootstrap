#
#   Copyright (c) 2021, Yoshihiko Nishikawa, Jun Takahashi, and Takashi Takahashi
#   Date: * Nov. 2021
#
#   A python3 code for the stationary bootstrap method [D. N. Politis and J. P. Romano (1994)] with estimating an
#   optimal parameter [D. N. Politis and H. White (2004), A. Patton, D. N. Politis, and H. White (2009)].
#
#   URL: https://github.com/YoshihikoNishikawa/StationaryBootstrap
#   See LICENSE for copyright information
#
#   If you use this code or find it useful, please cite ***
#


import numpy as np
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
import matplotlib.font_manager
import argparse
import os
from numba import njit, prange


del matplotlib.font_manager.weight_dict['roman']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 12
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

np.random.seed(1000)  # Fixed seed for PRNG
# Will calculate alpha \% confidence interval using the percentile/bootstrap-t/BCa method
alpha = math.erf(1.0 / 2.0 ** .5)  # = 68% confidence interval
number_bsamples = 1000  # Number of bootstrap samples
length_threshold = 10**6.0

parser = argparse.ArgumentParser(
    description='Stationary bootstrap analysis for 1d timeseries, '
    + 'input a filepath of the timeseries to be analyzed')
parser.add_argument(
    'arg1', help='Filepath of the timeseries you wish to analyze')
parser.add_argument('-p', '--parallel', action='store_true',
                    help='use this option if you wish to parallelize building pseudo timeseries')
parser.add_argument('-t', '--thinning', action='store_true',
                    help='use this option if you wish to thin the timeseries out using the correlation time')
parser.add_argument('-bca', '--bcamethod', action='store_true',
                    help='use this option if you wish to use the BCa method')
parser.add_argument('-bt', '--btmethod', action='store_true',
                    help='use this option if you wish to use the bootstrap-t method')

args = parser.parse_args()

if args.bcamethod and args.btmethod:
    print('\033[31m', 'Use only one of -bca and -bt options.', '\033[0m')
    quit()


def read_timeseries():
    print('-- Read a trajectory')
    file = args.arg1

    trajectory = []
    print('\tFile:\033[31m', file, '\033[0m')
    read_data = open(file, "r")
    for line in read_data.readlines():
        list = line.split()
        trajectory.append(float(list[0]))

    trajectory = np.array(trajectory)

    print('\targs.parallel is\033[34m', args.parallel, '\033[0m')

    print('\tThe length of the timeseries is', len(trajectory))
    if len(trajectory) > length_threshold:
        print('\tThe timeseries is too long, will be thinned out to make the length', int(
            length_threshold))
        thinning = int(len(trajectory) / length_threshold)
        trajectory = trajectory[::thinning]

    if args.thinning:
        print(
            '\t\033[31mThe timeseries will be thinned out using its correlation time\033[0m')
        trajectory = thin_trajectory(trajectory)
        print('\t\033[31mAfter thinning, the length is',
              len(trajectory), '\033[0m')

    return trajectory


def calculate_autocorrelation(trajectory, truncate_length):
    print('\t-- Calculating unnormalized autocorrelation function')
    truncated_trajectory = trajectory[:truncate_length]
    mean_data = np.mean(truncated_trajectory)
    autocorrelation = np.correlate(
        truncated_trajectory - mean_data, truncated_trajectory - mean_data, 'full') / len(truncated_trajectory)
    autocorrelation = autocorrelation[len(truncated_trajectory) - 1:]
    np.savetxt('autocorrelation.txt', np.transpose(autocorrelation))
    print('\t\tDone')
    return autocorrelation


def thin_trajectory(trajectory):
    normalized_autocorrelation = calculate_autocorrelation(
        trajectory, trajectory.shape[0])
    normalized_autocorrelation = normalized_autocorrelation / \
        normalized_autocorrelation[0]
    list_index = np.array(
        np.where(abs(normalized_autocorrelation) > 0.1))
    tau = np.max(list_index)
    print('\tEstimated correlation time:', tau)
    tau = (tau + 1) // 2

    return trajectory[::tau]


def find_bandwidth(autocorrelation):
    c = 2
    threshold = c * \
        np.sqrt(np.log10(len(autocorrelation)) / len(autocorrelation))
    K = np.max([5, int(np.sqrt(np.log10(len(autocorrelation))))])

    normalized_autocorrelation = autocorrelation / autocorrelation[0]
    list_index = np.array(
        np.where(abs(normalized_autocorrelation) < threshold))
    list_index = list_index.reshape(list_index.shape[1])

    list_time = np.array([j for j in list_index for i in range(K)
                          if j + i < normalized_autocorrelation.shape[0] and abs(normalized_autocorrelation[j + i]) < threshold])

    bandwidth = np.min(list_time)

    return 2 * bandwidth


def calculate_p_opt(trajectory, truncate_length):
    autocorrelation = calculate_autocorrelation(trajectory, truncate_length)
    bandwidth = find_bandwidth(autocorrelation)

    print('\t-- Calculating the optimal probability p_opt with bandwidth =', bandwidth)
    list_G = np.array([window_function(lag, bandwidth) * lag *
                       autocorrelation[lag] for lag in range(bandwidth)])
    G = 2 * np.sum(list_G)

    list_D = np.array([window_function(lag, bandwidth) *
                       autocorrelation[lag] for lag in range(1, bandwidth)])
    D = 2 * (autocorrelation[0] + 2 * np.sum(list_D))**2.0

    p_opt = np.power((2.0 * G**2.0 / D) *
                     autocorrelation.shape[0], -1.0 / 3.0)

    print('\t\tp_opt = ', p_opt)
    if p_opt > 1.0:
        return 1.0
    else:
        return p_opt


def window_function(lag, bandwidth):
    if lag <= 0.5 * bandwidth:
        return 1.0
    elif lag < bandwidth:
        return 2.0 * (1.0 - lag / bandwidth)
    else:
        return 0


def mean_suscep_kurtosis(list_bsamples):
    number_bsamples = list_bsamples.shape[0]
    mean = np.mean(list_bsamples, axis=1)
    bmean = mean.reshape(number_bsamples, 1)
    temp = (list_bsamples - bmean) ** 2.0
    qr_mean = np.mean(temp ** 2.0, axis=1)
    sq_mean = np.mean(temp, axis=1)

    return mean, sq_mean - mean ** 2.0, qr_mean / sq_mean ** 2.0


def percentile_conf_interval(list_observable, alpha):
    """
    The bootstrap percentile method for estimating 100*alpha % confidence interval

    Args:
        list_observable: A list of bootstrap samples
        alpha: The value for the confidence

    Return:
        The lower and upper limits of the estimated 100*alpha % confidence interval
    """

    list_observable.sort()
    gap = 0.5 * (1.0 - alpha)
    low_index = (int)(number_bsamples * gap)
    up_index = (int)(number_bsamples * (1.0 - gap))
    return list_observable[low_index], list_observable[up_index]


def lazy_BCa(list_observable, alpha):
    """
    The bias-corrected and accelerated (BC_a) method with no acceleration for estimating the confidence interval, 
    see [T. J. Diccio and B. Efron (1996)]

    Args:
        list_observable: A list of bootstrap samples
        alpha: The value for the confidence

    Return:
        The lower and upper limits of the estimated 100*alpha % confidence interval
    """
    list_observable.sort()
    mean = np.mean(list_observable)
    hatalpha0 = np.count_nonzero(
        list_observable <= mean) / len(list_observable)
    hatz0 = norm.ppf(hatalpha0)

    gap = 0.5 * (1.0 - alpha)
    acceleration = 0
    l_hatalpha = norm.cdf(hatz0 + (hatz0 + norm.ppf(gap)) /
                          (1.0 - acceleration * (hatz0 + norm.ppf(gap))))
    u_hatalpha = norm.cdf(hatz0 + (hatz0 + norm.ppf(1.0 - gap)) /
                          (1.0 - acceleration * (hatz0 + norm.ppf(1.0 - gap))))
    low_index = (int)(number_bsamples * l_hatalpha)
    up_index = (int)(number_bsamples * u_hatalpha)
    return list_observable[low_index], list_observable[up_index]


def lazy_Bootstrap_t(list_observable, alpha):
    """
    The bootstrap-t method for estimating the confidence interval, with an assumption the standard error 
    does not depend on bootstrap samples, see [T. J. Diccio and B. Efron (1996)]

    Args:
        list_observable: A list of bootstrap samples
        alpha: The value for the confidence

    Return:
        The lower and upper limits of the estimated 100*alpha % confidence interval
    """
    list_observable.sort()
    sigma = np.sqrt(np.var(list_observable) / (number_bsamples - 1))
    mean = np.mean(list_observable)
    bmean = np.full((number_bsamples), mean)
    list_t = (list_observable - bmean) / sigma
    list_t.sort()
    gap = 0.5 * (1.0 - alpha)
    low_index = (int)(number_bsamples * gap)
    up_index = (int)(number_bsamples * (1.0 - gap))
    low_t = list_t[low_index]
    up_t = list_t[up_index]

    return mean - sigma * low_t, mean - sigma * up_t


def print_item(truncate_length, p_opt, list_observable, alpha):

    low, up = percentile_conf_interval(list_observable, alpha)
    if args.btmethod:
        low, up = lazy_Bootstrap_t(list_observable, alpha)
    if args.bcamethod:
        low, up = lazy_BCa(list_observable, alpha)

    string = str(alpha * 100) + '%_confidence_interval'

    item = ['Length', truncate_length,
            '\tProb', p_opt,
            '\tMean', np.mean(list_observable),
            '\tErr', np.sqrt(number_bsamples *
                             np.var(list_observable) / (number_bsamples - 1)),
            '\t', string, low, up]

    return item


def plot_coeff_error_vs_n(list_output_observable, name, filename):
    plt.figure(name)
    xdata = [item[1] for item in list_output_observable]
    gap = 1.0 - 0.5 * (1.0 - alpha)
    ysdata = [item[7] * norm.ppf(gap) for item in list_output_observable]
    yddata = [(0.5 * (item[11] - item[10])) for item in list_output_observable]
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(xdata, ysdata, 'ko-', fillstyle='none',
             markersize=5, label=r'' + str(norm.ppf(gap)) + '$\sigma_n$')
    plt.plot(xdata, yddata, 'rs-', fillstyle='none',
             markersize=5, label=r'$\Delta_n$')
    plt.legend(frameon=True, labelspacing=0.2)
    plt.title(r'' + name, fontsize=20)
    plt.xlabel(r'Timeseries length $n$', fontsize=20)
    output_name = name.replace(" ", "_") + '_' + filename + '.pdf'
    plt.savefig(output_name, bbox_inches='tight')


def output_data(list_output, name, filename):
    file = name + '_' + filename + '.dat'
    print('\t', '\033[31m', file, '\033[0m')
    np.savetxt(file, list_output, fmt="%s")


@njit
def build_pseudo_timeseries_parallel(prob, trajectory, iteration):
    """
    Parallelized resampling method for the stationary bootstrap

    Args:
        prob: The parameter p for the stationary bootstrap method
        trajectory: Input time series data
        iteration: The (minimum) number of sampling from a geometric distribution needed to fill the pseudo time series

    Returns:
        A single resampled time series
    """

    list_length = np.random.geometric(prob, size=iteration)
    while np.sum(list_length) < trajectory.shape[0]:
        list_length = np.append(
            list_length, np.random.geometric(prob, size=1))

    list_beginning = np.random.randint(
        low=0, high=trajectory.shape[0], size=list_length.shape[0]
    )

    # Find the maximum index to be accessed (which can be larger than the trajectory length)
    max_index = np.max(list_beginning + list_length)

    # Find the minimum number to repeat the trajectory
    number_repeat = int(np.ceil(max_index / trajectory.shape[0]))

    # Make a periodic trajectory by replicating the original one
    periodic_trajectory = trajectory
    for i in range(number_repeat):
        periodic_trajectory = np.concatenate((periodic_trajectory, trajectory))

    pseudo_time_series = np.zeros(np.sum(list_length))
    first_index = 0
    for i in range(list_beginning.shape[0]):
        pseudo_time_series[first_index: first_index + list_length[i]] = periodic_trajectory[list_beginning[i]: list_beginning[i] +
                                                                                            list_length[i]]
        first_index += list_length[i]
    return pseudo_time_series[0:trajectory.shape[0]]


@njit(parallel=args.parallel)
def stationary_bootstrap_parallel(prob, trajectory, number_bsamples, truncate_length):
    """
    Parallelized stationary bootstrap.
    The arguments are same with the single core version

    Args:
        prob: The parameter p for the stationary bootstrap method
        trajectory: The original time series data
        number_bsamples: Number of bootstrap samples
        truncate_length: Upper bound for the length of the time series data

    Returns:
        resampled time series data
    """
    print('\t-- Buidling', number_bsamples, 'pseudo timeseries with p =', prob)
    iteration = (int)(1.1 * truncate_length * prob)

    truncate_trajectory = trajectory[: truncate_length]

    bsamples_array = np.zeros((number_bsamples, truncate_length))
    for b_index in prange(number_bsamples):
        bsamples_array[b_index, :] = build_pseudo_timeseries_parallel(
            prob, truncate_trajectory, iteration
        )
    return bsamples_array


def main():
    print('\033[31m-- Stationary bootstrap analysis starts --\033[0m')
    if args.parallel:
        print('\tParallel pseudo timeseries building')

    trajectory = read_timeseries()
    print('\tSimple mean:', np.mean(trajectory))

    list_truncate_length = [
        (int)(np.power(trajectory.shape[0], i / 20.0)) for i in range(5, 21)]

    list_output_mean = []
    list_output_suscep = []
    list_output_kurtosis = []
    for truncate_length in list_truncate_length:
        print('-- Truncate the timeseries at', truncate_length)
        p_opt = calculate_p_opt(trajectory, truncate_length)
        list_bsamples = stationary_bootstrap_parallel(
            p_opt, trajectory, number_bsamples, truncate_length)

        list_mean, list_suscep, list_kurtosis = mean_suscep_kurtosis(
            list_bsamples)

        list_output_mean.append(print_item(
            truncate_length, p_opt, list_mean, alpha))

        list_output_suscep.append(print_item(
            truncate_length, p_opt, list_suscep, alpha))

        list_output_kurtosis.append(print_item(
            truncate_length, p_opt, list_kurtosis, alpha))

    print('\033[31m', 'Output files:', '\033[0m')

    filename = os.path.splitext(os.path.basename(args.arg1))[0]

    #  Output data
    output_data(list_output_mean, 'mean', filename)
    output_data(list_output_suscep, 'susceptibility', filename)
    output_data(list_output_kurtosis, 'binder', filename)

    # Plot the standard error and the confidence interval
    plot_coeff_error_vs_n(list_output_mean, 'mean', filename)
    plot_coeff_error_vs_n(list_output_suscep,
                          'susceptibility', filename)
    plot_coeff_error_vs_n(list_output_kurtosis,
                          'Binder parameter', filename)


if __name__ == "__main__":
    main()
