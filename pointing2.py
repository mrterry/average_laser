import numpy as np
from itertools import chain


def get_n(nums, m, num_type):
    """strip off the first m numbers iin nums"""
    ans = [num_type(n) for n in nums[:m]]
    nums = nums[m:]
    return nums, ans


def parse(lines):
    """Parse provided pointing document

    Return pointings, rings, spots, interp
    pointings: len(pointings) = n_beams
        The wall coordinates & on-target focal spot coordinates for each beam
    rings: len(rings) = n_rings
        For each ring, a list of the component beams
    spots: len(spots) = n_rings
        For each ring, the focal spot paramters
    """
    lines = (line.strip() for line in lines)
    lines = (line for line in lines if line != '')
    lines = (line for line in lines if line[0] != '!')
    nums = ' '.join(lines).split()


    n_segs = int(nums[0])
    nums = nums[1:]

    ms = [int(m) for m in nums[:n_segs]]
    nums = nums[n_segs:]

    dtype = ('theta', 'phi', 'theta_rp', 'phi_rp')
    dtype = [(k, float) for k in dtype]
    pointings = np.zeros(ms[0], dtype=dtype)

    nums, pointings['theta'] = get_n(nums, ms[0], float)
    nums, pointings['phi'] = get_n(nums, ms[1], float)
    nums, pointings['theta_rp'] = get_n(nums, ms[2], float)
    nums, pointings['phi_rp'] = get_n(nums, ms[3], float)


    rings = []
    i = 5 + ms[n_segs-1] - 1
    for m in ms[4:i]:
        nums, r = get_n(nums, m, int)
        rings.append(np.array(r, dtype=int)-1)

    interp = 0

    n_rings  = ms[-1]

    dtype = ('sg1', 'radii', 'ellip1', 'amp1', 'amp2', 'sg2', 'off1', 'off2', 'ellip2')
    dtype = [(k, float) for k in dtype]
    spots = np.zeros(n_rings, dtype=dtype)
    if nums:
        nums, spots['sg1'] = get_n(nums, n_rings, float)
        nums, spots['radii'] = get_n(nums, n_rings, float)
        nums, spots['ellip1'] = get_n(nums, n_rings, float)
        spots['amp1'] = 1.
        nums, spots['amp2'] = get_n(nums, n_rings, float)
        nums, spots['sg2'] = get_n(nums, n_rings, float)
        nums, spots['off2'] = get_n(nums, n_rings, float)
        nums, spots['ellip2'] = get_n(nums, n_rings, float)

    return pointings, rings, spots, interp


def get_beam_pattern(r0, spot):
    x = 0.2 * np.linspace(-1, 1, 500)
    X, Y = np.meshgrid(x, x)

    a = -np.log(0.05)
    def R(xo, el):
        return np.sqrt(X**2 + ((Y-xo)*el)**2)

    R0 = R(spot['off1'], 1)
    R1 = R(0, 1)
    R2 = R(spot['off2']*r0, spot['ellip2'])

    I = np.zeros_like(R1)
    I += spot['amp1'] * np.exp(-a*(R1/r0)**spot['sg1'])
    I += spot['amp2'] * np.exp(-a*(R2/r0)**spot['sg2'])

    sg_apr = 10.
    r_apr = 1.15*r0
    I *= np.exp(-a * (R0/r_apr)**sg_apr)

    I /= I.max()
    return X, Y, I


def get_pointing(f, r0):
    with open('NIFPortConfig.dat') as f:
        lines = f.readlines()
    pointings, rings, spots, interp = parse(lines)

    # split out top & bottom rings
    rings_top = []
    rings_bot = []
    for r in rings:
        mask = pointings[r]['theta'] > 90.
        rings_top.append(r[-mask])
        rings_bot.append(r[mask])
        
    # infer off1 based on pointing information
    for i,r in enumerate(rings_top):
        p = pointings[r]
        t_rp = p['theta_rp'][0]
        t = p['theta'][0]
        off1 = 0.5*r0 * np.sin(np.pi/180.*(t_rp - t)) / np.sin(t)
        spots[i]['off1'] = off1
    return pointings, rings_top, rings_bot, spots


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
