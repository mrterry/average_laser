import numpy as np
from itertools import chain

import pylab as plt

from pdd_pointing import get_pointing, pdd_beam_pattern


def sample_beam_pattern(pattern_xyi, nrays):
    X, Y, I = pattern_xyi
    I2 =  I[:-1, :-1].copy()
    I2 += I[:-1, 1:]
    I2 += I[1:, :-1]
    I2 += I[1:, 1:]

    ni, nj = I2.shape
    I2 = I2.flatten()

    cum_prob = I2.cumsum()
    cum_prob = np.insert(cum_prob, 0, 0.)
    cum_prob /= cum_prob[-1]

    samps = np.random.random(nrays)
    k = np.digitize(samps, cum_prob) - 1

    i, j = divmod(k, nj)
    dx = X[0, 1:] - X[0, :-1]
    dy = Y[1:, 0] - Y[:-1, 0]

    focus = np.zeros((nrays, 3))
    focus[:, 0] = X[i,j] + dx[j]*np.random.random(nrays)
    focus[:, 1] = Y[i,j] + dy[i]*np.random.random(nrays)
    return focus


def get_pq(lens, focus, far_pattern, nrays):
    """
    Get ray launch locations (p) and directions (q)
    far_pattern:
            the far-field intensity patter (at focal spot)
    """
    lens_hw = 40./2
    lens = np.array(lens)
    focus = np.array(focus)

    p0 = np.zeros((nrays, 3))
    p0[:, 0] = 2*lens_hw*np.random.random(nrays) - lens_hw
    p0[:, 1] = 2*lens_hw*np.random.random(nrays) - lens_hw
    p0[:, 2] = mag(lens)
    p0 = rotate(p0, lens)

    p1 = sample_beam_pattern(far_pattern, nrays)
    p1[:, 2] = mag(focus)
    p1 = rotate(p1, focus)

    q = p1 - p0
    q /= mag(q)[:, None]
    return p0, q


def mag(a):
    """Magnitude of vector (L2 norm)"""
    return np.sqrt((a*a).sum(-1))


def rotate(points, center):
    r = mag(points)
    theta = np.arccos(points[:, 2]/r)
    phi = np.arctan2(points[:, 1], points[:, 0])

    rr = mag(center)
    dt = np.arccos(center[2]/rr),
    dp = np.arctan2(center[1], center[0])

    theta += dt
    phi += dp

    sint, cost = np.sin(theta), np.cos(theta)
    sinp, cosp = np.sin(phi), np.cos(phi)

    ans = np.zeros_like(points)
    ans[:, 0] = r*sint*cosp
    ans[:, 1] = r*sint*sinp
    ans[:, 2] = r*cosp
    return ans
def pprint_points_rz(p, **kwargs):
    r = mag(p[:, :2])
    z = p[:, 2]
    plt.plot(z, r, 'o', **kwargs)


def pprint_lines(p0, p1):
    r0 = mag(p0[:, :2])
    r1 = mag(p1[:, :2])
    for z0, rr0, z1, rr1 in zip(p0[:, 2], r0, p1[:, 2], r1):
        plt.plot([z0, z1], [rr0, rr1])


def cost(p, q, r):
    """
    Given initial location (p), direction (q) and shell radius (r)
    return the cosine of the angle of incidence
    """
    p_mag = mag(p)
    #pprint_points_rz(p, label='launch')

    s = (p*q).sum(1)
    mu = s / p_mag
    disc = mu**2 - 1 + r**2/p_mag**2
    mask = disc >= 0
    print 'discarded %i / %i' % (sum(-mask), len(p))
    #pprint_points_rz(p[mask], label='filtered')

#    l_miss = -p[-mask, 0]/q[-mask, 0]
#    p_miss = p[-mask] + l_miss[:, None]*q[-mask]
#    pprint_lines(p[-mask], p_miss)
    s = s[mask]
    p_mag = p_mag[mask]
    disc = disc[mask]

    l1 = -s + p_mag * np.sqrt(disc)
    l2 = -s - p_mag * np.sqrt(disc)
    l = np.minimum(l1, l2)

    #p1 = p[mask] + l[:, None]*q[mask]
    #pprint_points_rz(p1, label='impact')

    ans = np.empty(len(p))
    ans[mask] = -(s+l)/r
    ans[-mask] = -5
    return ans


def xyz(r, t, p):
    """ Convert r-theta-phi coordinates to x-y-z """
    st, ct = np.sin(t), np.cos(t)
    sp, cp = np.sin(p), np.cos(p)
    return np.array((r*st*cp, r*st*sp, r*ct))


def iter_step_sizes(total, stride):
    assert total > 0
    assert stride > 0
    n_whole, left = divmod(total, stride)
    for i in range(n_whole):
        yield stride
    if left:
        yield left


class IncrementalHist(object):
    def __init__(self, bins, buff_size):
        self.bins = bins
        self.buffer = np.zeros(buff_size)
        self.hist = np.zeros(len(bins)-1)
        self.loc = 0
        self.end = len(self.buffer)

    def add(self, data):
        n = len(data)
        if self.loc + n >= self.end:
            self.flush()

        self.buffer[self.loc:self.loc+n] = data[:]
        self.loc += n

    def flush(self):
        if self.loc > 0:
            h, b = np.histogram(self.buffer[:self.loc], bins=self.bins)
            self.hist += h
            self.loc = 0


def get_hist(r0, R, nrays, path, stride=10000):
    with open(path) as f:
        pointings, rings_top, rings_bot, spots = get_pointing(f, r0)


    cos_theta_bins = np.cos(np.linspace(0, np.pi/2, 100))[::-1]
    cos_theta_bins[0] = 0.
    cos_theta_bins[-1] = 1.
    print cos_theta_bins
    theta_hist = np.zeros(len(cos_theta_bins)-1)

    for spot, top_beam_ids, bot_beam_ids in zip(spots, rings_top, rings_bot):
        XYI = get_beam_pattern(r0, spot)
        for beam in chain(pointings[top_beam_ids], pointings[bot_beam_ids]):
            lens = xyz(R, beam['theta'], beam['phi'])
            focus = xyz(r0, beam['theta_rp'], beam['phi_rp'])

            for step in iter_step_sizes(nrays, stride):
                p, q = get_pq(lens, focus, XYI, nrays)
                cos_t = cost(p, q, r0)
                b, hist = np.histogram(cos_t, bins=cos_theta_bins)
                theta_hist += theta_hist
    return cos_theta_bins, theta_hist


r0 = 1585.e-4
R = 500.
nrays = 200000
cos_theta_bins, theta_hist = get_hist(r0, R, nrays, 'NIFPortConfig.dat')
import pylab as plt
plt.plot(cos_theta_bins[1:], theta_hist)
plt.show()
