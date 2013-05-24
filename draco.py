import numpy as np


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


def read_scheme(f, r0):
    """
    Read the pointing scheme from an open file handle
    """
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


def beam_pattern(r0, spot):
    """
    Compute the intensity pattern for a spot pointed at a target with radius r0
    """
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
