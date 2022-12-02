import numpy as np
from scipy import signal


def gkern(kernlen=21, std=None):
    """Returns a 2D Gaussian kernel array."""
    if std is None:
        std = kernlen / 4
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def build_gaussian_heatmaps(kp_xyc, w, h, gaussian=None):
    gaussian_heatmaps = np.zeros((len(kp_xyc), h, w))
    for i, kp in enumerate(kp_xyc):
        # do not use invisible keypoints
        if kp[2] == 0:
            continue

        kpx, kpy = kp[:2].astype(int)

        if gaussian is None:
            g_scale = 6
            g_radius = int(w / g_scale)
            gaussian = gkern(g_radius * 2 + 1)
        else:
            g_radius = gaussian.shape[0] // 2

        rt, rb = min(g_radius, kpy), min(g_radius, h - 1 - kpy)
        rl, rr = min(g_radius, kpx), min(g_radius, w - 1 - kpx)

        gaussian_heatmaps[i, kpy - rt:kpy + rb + 1, kpx - rl:kpx + rr + 1] = gaussian[
                                                                             g_radius - rt:g_radius + rb + 1,
                                                                             g_radius - rl:g_radius + rr + 1]
    return gaussian_heatmaps
