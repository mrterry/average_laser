import numpy as np
from itertools import chain

import pylab as plt

import draco


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


def get_pq(lens_rtp, lens_hw, focus_rtp, far_pattern, nrays):
    """
    Get ray launch locations (p) and directions (q)
    far_pattern:
            the far-field intensity patter (at focal spot)
    """
    p0 = np.zeros((nrays, 3))
    p0[:, 0] = np.random.uniform(low=-lens_hw, high=lens_hw, size=nrays)
    p0[:, 1] = np.random.uniform(low=-lens_hw, high=lens_hw, size=nrays)
    p0[:, 2] = lens_rtp[0]
    theta, phi = lens_rtp[1:]
    p0 = rotate(p0, dtheta=theta, dphi=phi)

    p1 = sample_beam_pattern(far_pattern, nrays)
    p1[:, 2] = focus_rtp[0]
    theta_rp, phi_rp = focus_rtp[1:]
    p1 = rotate(p1, dtheta=theta_rp, dphi=phi_rp)

    q = p1 - p0
    q /= mag(q)[:, None]
    return p0, q


def mag(a):
    """Magnitude of vector (L2 norm)"""
    return np.sqrt((a*a).sum(-1))


def rotate(points, dtheta=0., dphi=0.):
    """Rotate z-axis by dtheta, then dphi"""

    # Rotation about y-axis
    Rtheta = np.zeros((3,3))
    ct, st = np.cos(dtheta), np.sin(dtheta)
    Rtheta[0, 0] = ct
    Rtheta[2, 0] = -st
    Rtheta[1, 1] = 1.
    Rtheta[0, 2] = st
    Rtheta[2, 2] = ct

    # rotation about z axix
    Rphi = np.zeros((3,3))
    cp, sp = np.cos(dphi), np.sin(dphi)
    Rphi[0, 0] = cp
    Rphi[1, 0] = sp
    Rphi[0, 1] = -sp
    Rphi[1, 1] = cp
    Rphi[2, 2] = 1.

    R = np.dot(Rphi, Rtheta)
    return np.dot(R, points.T).T


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
        pointings, rings_top, rings_bot, spots = draco.read_scheme(f, r0)

    lens_hw = 40./2

    nbins = 200
    bins = np.linspace(0, np.pi/2, nbins+1)
    bins = np.cos(bins)[::-1]
    theta_hist = IncrementalHist(bins, 10*nrays)

    deg2rad = np.pi/180.

    for spot, top_beam_ids, bot_beam_ids in zip(spots[:1], rings_top, rings_bot):
        print spot
        XYI = draco.beam_pattern(r0, spot)
        for beam in chain(pointings[top_beam_ids], pointings[bot_beam_ids]):
            lens_rtp = R, beam['theta']*deg2rad, beam['phi']*deg2rad
            focus_rtp = r0, beam['theta_rp']*deg2rad, beam['phi_rp']*deg2rad

            for step in iter_step_sizes(nrays, stride):
                p, q = get_pq(lens_rtp, lens_hw, focus_rtp, XYI, step)
                cos_t = cost(p, q, r0)
                theta_hist.add(cos_t)
    theta_hist.flush()
    return theta_hist.bins, theta_hist.hist


def run():
    r0 = 1585.e-4
    R = 500.
    nrays = 200000
    bins, hist = get_hist(r0, R, nrays, 'NIFPortConfig.dat')

    hist = hist/float(sum(hist))
    t = np.arccos(bins[1:])
    np.savez('ray_hist.npz', theta=t, hist=hist)

    plt.clf()
    plt.plot(t, hist, 'o-')
    plt.show()

run()
