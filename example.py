from scipy import special
from copy import deepcopy
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, entropy

THRESHOLD = 1e-6
NUM_TERMS = 10

def clip_ratio(x, epsilon=0.2):
    return np.clip(x, a_min=1-epsilon, a_max=1+epsilon)

def q_logarithm(x, q):
    """deal with the base case in the power series where q=k
    """
    if q == 1:
        return np.ones_like(x) 
    return (x**(1-q) - 1) / (1-q)


def kl_tkl(num_terms, pi, mu):
    x = clip_ratio(mu/pi)
    if num_terms == 2:
        num_terms += 1
    return np.nansum([(-1)**q / q * (np.sum([(-1)**k*special.comb(q, k)*np.nansum(pi*((q-k)*q_logarithm(x, 1-q+k)+1)) for k in range(1, q+1)]) + np.nansum(pi*(q*q_logarithm(x, 1-q)+1))) for q in range(2, num_terms)])


def jeffrey_tkl(num_terms, pi, mu):
    x = clip_ratio(mu/pi)
    if num_terms == 2:
        num_terms += 1
    return np.nansum([(-1)**q / (q-1) * (np.sum([(-1)**k*special.comb(q, k)*np.nansum(pi*((q-k)*q_logarithm(x, 1-q+k)+1)) for k in range(1, q+1)]) + np.nansum(pi*(q*q_logarithm(x, 1-q)+1))) for q in range(2, num_terms)])


def jensen_shannon_tkl(num_terms, pi, mu):
    x = clip_ratio(mu/pi)
    return np.nansum([(-1)**q*(1 - 0.5**(q-2))/(q*(q-1)) * (np.sum([(-1)**k*special.comb(q, k)*np.nansum(pi*((q-k)*q_logarithm(x, 1-q+k)+1)) for k in range(1, q+1)]) + np.nansum(pi*(q*q_logarithm(x, 1-q)+1))) for q in range(2, num_terms+1)])


def gan_tkl(num_terms, pi, mu):
    x = clip_ratio(mu/pi)
    return np.nansum([(-1)**q*(1 - 0.5**(q-1))/(q*(q-1)) * (np.sum([(-1)**k*special.comb(q, k)*np.nansum(pi*((q-k)*q_logarithm(x, 1-q+k)+1)) for k in range(1, q+1)]) + np.nansum(pi*(q*q_logarithm(x, 1-q)+1))) for q in range(2, num_terms+1)])

def ratio_filtering(pk, qk):
    pk[pk<THRESHOLD] = np.nan
    qk[qk<THRESHOLD] = np.nan
    return pk, qk

if __name__ == '__main__':
    """
    The f divergence approximation is based on the assumption that the policy ratio is close to 1,
    which is the prerequisite for the Taylor expansion using Chi^n or TKL divergence.

    From a practical point of view, the power series of distribution ratio (pk/qk)**n can be divergent without clipping,
    and the approximation can converge only when the two distributions are similar.
    If the two distributions differ very much the approximation can be very bad!
    modify the following mean and std to see the effect of the approximation
    """
    
    mean1, std1 = np.random.rand() * 10, np.random.rand() * 5 + 5  # std1 in range [5, 10]
    mean2 = mean1 + (np.random.rand() - 0.5) * 10  # Ensure mean2 is within ±2.5 of mean1
    std2 = std1 + (np.random.rand() - 0.5) * 2  # Ensure std2 is within ±1 of std1

    x = np.linspace(-10, 20, 1000)
    gaussian1 = norm.pdf(x, mean1, std1)
    gaussian2 = norm.pdf(x, mean2, std2)

    gaussian1 /= np.sum(gaussian1)
    gaussian2 /= np.sum(gaussian2)

    exact_kl = entropy(gaussian1, gaussian2)
    exact_jeffrey = entropy(gaussian1, gaussian2) + entropy(gaussian2, gaussian1)
    exact_jensen_shannon = entropy(gaussian1, (gaussian1+gaussian2)) + entropy(gaussian2, (gaussian1+gaussian2)) 

    pk, qk = ratio_filtering(deepcopy(gaussian1), deepcopy(gaussian2))

    error_kl = [exact_kl, exact_kl]
    error_jeffrey = [exact_jeffrey, exact_jeffrey]
    error_jensen_shannon = [exact_jensen_shannon, exact_jensen_shannon]

    for num_term in range(2, NUM_TERMS):
        approx_kl_tkl = kl_tkl(num_term, pk, qk)
        approx_jeffrey_tkl = jeffrey_tkl(num_term, pk, qk)
        approx_gan_tkl = gan_tkl(num_term, pk, qk)
        error_kl.append(abs(approx_kl_tkl - exact_kl))
        error_jeffrey.append(abs(approx_jeffrey_tkl - exact_jeffrey))
        error_jensen_shannon.append(abs(approx_gan_tkl - exact_jensen_shannon))
    
    divs = {'kl': error_kl, 'jeffrey': error_jeffrey, 'jensen_shannon': error_jensen_shannon}
    keys = list(divs.keys())
    fig, ax = plt.subplots(len(keys)+1, 1, figsize=(12, 10))
    for i in range(len(ax)):
        if i == 0:
            ax[i].plot(x, gaussian1, label=f'Gaussian 1: mean={mean1:.2f}, std={std1:.2f}')
            ax[i].plot(x, gaussian2, label=f'Gaussian 2: mean={mean2:.2f}, std={std2:.2f}')
            ax[i].legend()
            continue
        ax[i].plot(divs[keys[i-1]], label=f'{keys[i-1]} Approx. Error')
        ax[i].set_xticks(list(range(NUM_TERMS)))
        ax[i].set_xticklabels(list(range(NUM_TERMS)))
        ax[i].set_ylabel('Error')
        ax[i].legend()
    ax[-1].set_xlabel('Number of Terms')
    fig.suptitle(f'Approximation By Tsallis KL Divergence')
    plt.show()