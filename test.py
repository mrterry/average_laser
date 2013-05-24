from transport import rotate, mag
import numpy as np
import pylab as plt


def test_rotate():
    n = 200
    p0 = np.zeros((n, 3))
    p0[:, 1] = np.linspace(0, 1.1, n)
    p0[:, 2] = 2.

    deg = np.pi/180.
    p0 = rotate(p0, dtheta=90*deg)

    q = np.zeros((n, 3))
    q[:, 0] = -1

    p1 = p0 + 5*q

    p0 = rotate(p0, dphi=45.*deg)
    p1 = rotate(p1, dphi=45.*deg)

    q = p0 - p1
    q /= mag(q)[:, None]

    plt.subplot(111, aspect='equal')
    plt.plot(p0[:, 0], p0[:, 1], 'o')
    plt.plot(p1[:, 0], p1[:, 1], 'o')
    plt.show()


