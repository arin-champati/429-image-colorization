import numpy as np
from sklearn.neighbors import NearestNeighbors
from load_cifar import get_cifar10_data, get_cifar100_data
from skimage import color
from scipy.ndimage import gaussian_filter

# Partitions ab color space into bins of param size


def get_ab_bins(size=100):
    if size == 313:
        bins = np.load('Data/pts_in_hull.npy')
        print(np.load('Data/prior_probs.npy'))
        return bins
    a = [-110, 110]
    b = [-110, 110]
    side = int(size ** 0.5)
    stride = 220 / side
    a_range = np.arange(a[0], b[1], stride)
    b_range = np.arange(a[0], b[1], stride)
    bins = []
    for aa in a_range:
        for bb in b_range:
            bins.append([aa, bb])
    bins = np.array(bins)
    print(bins)
    return bins


def get_uv_bins(size=100):
    u = [-0.5, 0.5]
    v = [-0.5, 0.5]
    side = size ** 0.5
    stride = 1 / side
    u_range = np.arange(u[0], v[1], stride)
    v_range = np.arange(u[0], v[1], stride)
    bins = []
    for uu in u_range:
        for vv in v_range:
            bins.append([uu, vv])
    print(bins)
    return np.array(bins)


def get_gamut(size=100, num_neighbors=1, ab=False):
    bins = None
    if ab:
        bins = get_ab_bins(size)
    else:
        bins = get_uv_bins(size)
    gamut = NearestNeighbors(n_neighbors=num_neighbors)
    gamut.fit(bins)
    return gamut


def get_freq_uv(data, size=225):
    gamut = get_gamut(size=size, num_neighbors=1)
    luv = color.rgb2yuv(data)

    u = np.ravel(luv[:, :, :, 1])
    v = np.ravel(luv[:, :, :, 2])
    uv = np.vstack((u, v)).T
    _, idx = gamut.kneighbors(uv)
    idx = np.squeeze(idx)
    elements, freq = np.unique(idx, return_counts=True)
    freq_real = np.zeros(size)
    freq_real[elements] = freq
    return freq_real


def get_freq_ab(data, size=225):
    gamut = get_gamut(size=size, num_neighbors=1, ab=True)
    lab = color.rgb2lab(data)
    side = int(size ** 0.5)
    a = np.ravel(lab[:, :, :, 1])
    b = np.ravel(lab[:, :, :, 2])
    ab = np.vstack((a, b)).T
    _, idx = gamut.kneighbors(ab)
    idx = np.squeeze(idx)
    elements, freq = np.unique(idx, return_counts=True)
    freq_real = np.zeros(size)
    freq_real[elements] = freq
    return freq_real


# obtains empirical probability of ab color space with respect to bins,
# using the cifar_10 dataset


def get_cifar10_freq(size=100, ab=False):
    data = get_cifar10_data()
    if ab:
        return get_freq_ab(data, size)
    freq = get_freq_uv(data, size)
    return freq

# obtains empirical probability of ab color space with respect to bins,
# using the cifar_100 dataset


def get_cifar_100_freq(size=100, ab=False):
    data = get_cifar100_data()
    if ab:
        return get_freq_ab(data, size)
    freq = get_freq_uv(data, size)
    return freq

# merges probabilities obtained from cifar_10 and cifar_100


def get_joint_prob_smoothened(size=100, lamb=0.1, sigma=3, ab=False):
    freq10 = get_cifar10_freq(size, ab)
    side = int(size ** 0.5)
    probs = freq10 / size
    print("====================================================")
    probs = np.reshape(probs, (side, side))
    probs_prior = probs
    probs = probs_prior / np.sum(probs_prior)
    print(probs)

    # Mix in uniform distribution, reciprocate
    prior_mix = (1-lamb) * probs + lamb / size
    prior_mix = 1 / prior_mix

    # renormalize so that expected value is 1
    ev = np.sum(prior_mix * probs_prior)
    prior_mix /= ev
    prior_mix = prior_mix.flatten()
    return prior_mix

# mixes in a uniform distribution, and normalize the final probability distribution


def get_joint_prob(size=100, gamma=0.1, sigma=3, ab=False):
    freq10 = get_cifar10_freq(size, ab)
    side = int(size ** 0.5)
    prior_probs = freq10 / np.sum(freq10)
    prior_probs = prior_probs.reshape((side, side))
    print(prior_probs)
    print("================================================")
    prior_probs = gaussian_filter(prior_probs, sigma=sigma, mode='constant')
    print(prior_probs)
    prior_probs /= np.sum(prior_probs)
    prior_probs = prior_probs.flatten()
    print("================================================")
    print(np.sum(prior_probs))

    # convex combination of empirical prior and uniform distribution
    prior_mix = (1-gamma)*prior_probs + gamma/size

    # set prior factor
    prior_factor = 1/prior_mix
    out = prior_factor / np.sum(prior_factor * prior_probs)
    return out


def save_probs(size=100, ab=False):
    probs = get_joint_prob(size, ab=ab)
    print(probs, probs.shape, np.sum(probs))
    np.save(f'probs_{size}', probs)


if __name__ == '__main__':
    # save_probs(size=313)
    save_probs(size=81, ab=True)
