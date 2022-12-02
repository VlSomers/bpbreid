import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from torchreid.utils import AverageMeter, Logger
from torchreid.utils.engine_state import EngineState


def plot_body_parts_pairs_distance_distribution(body_part_pairwise_dist, q_pids, g_pids, tag):
    m = body_part_pairwise_dist.shape[0]
    cols = round(math.sqrt(m))
    rows = cols
    while rows * cols < m:
        rows += 1
    fig = plt.figure(figsize=(rows*5, cols*3))
    ssmd_meter = AverageMeter()
    for i in range(0, m):
        ax = fig.add_subplot(rows, cols, i+1)
        pos_p_mean, pos_p_std, neg_p_mean, neg_p_std, ssmd = compute_distance_distribution(
            ax, body_part_pairwise_dist[i], q_pids, g_pids, "Bp {} pairs distance distribution".format(i))
        ssmd_meter.update(ssmd)
    fig.tight_layout()
    Logger.current_logger().add_figure("{} body part pairs distance distribution".format(tag), fig, EngineState.current_engine_state().epoch)
    return ssmd_meter.avg


def plot_pairs_distance_distribution(distmat, q_pids, g_pids, tag):
    fig, ax = plt.subplots()
    result = compute_distance_distribution(ax, distmat, q_pids, g_pids, "{} pairs distance distribution".format(tag))
    Logger.current_logger().add_figure("{} pairs distance distribution".format(tag), fig, EngineState.current_engine_state().epoch)
    return result


def compute_distance_distribution(ax, distmat, q_pids, g_pids, title):
    pos_p = distmat[np.expand_dims(q_pids, axis=1) == np.expand_dims(g_pids, axis=0)]
    neg_p = distmat[np.expand_dims(q_pids, axis=1) != np.expand_dims(g_pids, axis=0)]

    pos_p_mean, pos_p_std, neg_p_mean, neg_p_std, ssmd = compute_ssmd(neg_p, pos_p)

    # plot_distributions(ax, neg_p, pos_p, pos_p_mean, pos_p_std, neg_p_mean, neg_p_std)
    # ax.set_title(title + " - SSMD = {:.4f} ".format(ssmd))

    return pos_p_mean, pos_p_std, neg_p_mean, neg_p_std, ssmd


def compute_ssmd(neg_p, pos_p):
    pos_p_mean = np.mean(pos_p)
    pos_p_std = np.std(pos_p)
    neg_p_mean = np.mean(neg_p)
    neg_p_std = np.std(neg_p)
    ssmd = abs(pos_p_mean - neg_p_mean) / (pos_p_std ** 2 + neg_p_std ** 2)

    return pos_p_mean, pos_p_std, neg_p_mean, neg_p_std, ssmd


def plot_distributions(ax, neg_p, pos_p, pos_p_mean, pos_p_std, neg_p_mean, neg_p_std):
    bins = 100
    ax.hist(pos_p, weights=np.ones_like(pos_p)/len(pos_p), bins=bins, label='{} positive pairs : $\mu={:10.3f}$, $\sigma={:10.3f}$'.format(len(pos_p), pos_p_mean, pos_p_std), density=False, alpha=0.4, color='green')
    ax.hist(neg_p, weights=np.ones_like(neg_p)/len(neg_p), bins=bins, label='{} negative pairs : $\mu={:10.3f}$, $\sigma={:10.3f}$'.format(len(neg_p), neg_p_mean, neg_p_std), density=False, alpha=0.4, color='red')
    ax.axvline(x=pos_p_mean, linestyle='--', color='darkgreen')
    ax.axvline(x=neg_p_mean, linestyle='--', color='darkred')
    ax.set_xlabel("pairs distance")
    ax.set_ylabel("pairs count")
    ax.legend()
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
