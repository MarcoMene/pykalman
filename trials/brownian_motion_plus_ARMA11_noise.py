import numpy as np
from bsp_data_science.minime.fitter import fit_minimizing
from scipy.linalg import expm
from numpy.random import multivariate_normal, normal, exponential
import matplotlib.pyplot as plt
from numpy import sqrt

from bsp_stochastic.plotting.utils_plots import show_plot

from pykalman import KalmanFilter
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

np.random.seed(666)

dt = 1 / (6 * 60 * 24 * 365)  # 10 seconds data

# MC truth parameters
x0 = 0
sigma = 0.07  # yearly volatility
sigma_eps = 0.00015  # 0 #  # size of flukes

# ARMA(1,1) flukes
phi = 0.3
theta = -0.2

print(f"MC truth: sigma {sigma} sigma_eps {sigma_eps}, (phi, theta) == {(phi, theta)}")

# Kalman variables initialization
x_0 = np.array([x0, 0, 0])
R = np.array([0])  # measurement noise null

# process noise
Q = np.array([[sigma * sigma * dt, 0, 0],
              [0, sigma_eps ** 2, sigma_eps ** 2],
              [0, sigma_eps ** 2, sigma_eps ** 2]
              ])

F = np.array([[1, 0, 0],
              [0, phi, theta],
              [0, 0, 0]
              ])

H = np.array([[1, 1, 0]])

# simulation
# T = 6 * 60 * 20  # time steps ~ 1 day
T = 1000  # trial

state_true = [x_0]
z = [np.array([x_0[0]])]  # measurements

times = np.cumsum([dt] * T)

print(f"MC truth: sigmas per time step: real {sigma * sqrt(dt)}, noise {sigma_eps}")

for i in range(1, len(times)):
    # diff_time = times[i] - times[i - 1]

    x_t = np.dot(F, state_true[i - 1]) + multivariate_normal([0, 0, 0], Q)
    state_true.append(x_t)

    z.append(np.dot(H, x_t))

state_true = np.array(state_true)
z = np.array(z)

# print(state_true)
# print(z)

x_true = state_true[:, 0]
z_ = z[:, 0]
f = state_true[:, 1]
f_v2 = z_ - x_true
epsilon = state_true[:, 2]

# kalman filter - pykalman version
# kf = KalmanFilter(transition_matrices=F,
#                   observation_matrices=H,
#                   initial_state_mean=x_0,
#                   initial_state_covariance=Q * 3,
#                   transition_covariance=Q,
#                   transition_offsets=np.zeros(3),
#                   observation_offsets=np.zeros(1),
#                   observation_covariance=R,
#                   em_vars=None  # [ 'transition_covariance'], # 'transition_matrices'
#                   )
#
measurements = z


# kf = kf.em(measurements, n_iter=10)

# kf.loglikelihood(z_)

# try to estimate parameters by maximizing likelihood

def negative_log_likelihood_kf(params, data):
    """
    :param params: phi, theta
    :param data:  measurements
    """
    phi, theta, sigma, sigma_eps = params

    if phi < -1 or theta < -1 or phi > 1 or theta > 1:
        return np.inf

    F_cur = np.array([[1, 0, 0],
                      [0, phi, theta],
                      [0, 0, 0]
                      ])

    Q_cur = np.array([[sigma * sigma * dt, 0, 0],
                      [0, sigma_eps ** 2, sigma_eps ** 2],
                      [0, sigma_eps ** 2, sigma_eps ** 2]
                      ])

    kf = KalmanFilter(transition_matrices=F_cur,
                      observation_matrices=H,
                      initial_state_mean=x_0,
                      initial_state_covariance=Q_cur * 3,
                      transition_covariance=Q_cur,
                      transition_offsets=np.zeros(3),
                      observation_offsets=np.zeros(1),
                      observation_covariance=R,
                      em_vars=None
                      )
    return -kf.loglikelihood(data)


# fitting parameters without EM algo
# fit_result = fit_minimizing(minimizing_function=negative_log_likelihood_kf, minimizing_function_data=[z], initial_params=[0.1,-0.1,sigma, sigma_eps]) #, method='Powell')
# print(fit_result)
# print(f"Fitted phi, theta, sigma, sigma_eps: {fit_result.x}")
##     fitted_params = fit_result.x


# re-define KF with pre-fitted parameters
phi_hat, theta_hat, sigma_hat, sigma_eps_hat = 1.51190377e-01, -5.59381360e-02, 6.64069705e-02, 1.47392465e-04
kf = KalmanFilter(transition_matrices=np.array([[1, 0, 0],
                                                [0, phi_hat, theta_hat],
                                                [0, 0, 0]
                                                ]),
                  observation_matrices=H,
                  initial_state_mean=x_0,
                  initial_state_covariance=Q * 3,
                  transition_covariance=np.array([[sigma_hat * sigma_hat * dt, 0, 0],
                                                  [0, sigma_eps_hat ** 2, sigma_eps_hat ** 2],
                                                  [0, sigma_eps_hat ** 2, sigma_eps_hat ** 2]
                                                  ]),
                  transition_offsets=np.zeros(3),
                  observation_offsets=np.zeros(1),
                  observation_covariance=R,
                  em_vars=None  # [ 'transition_covariance'], # 'transition_matrices'
                  )

print(f"true process covariance {Q}, estimated with EM: {kf.transition_covariance}")
print(f"true transition matrix {F}, estimated with EM: {kf.transition_matrices}")

(filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

x_filter = filtered_state_means[:, 0]
x_smooth = smoothed_state_means[:, 0]
f_filter = filtered_state_means[:, 1]
f_smooth = smoothed_state_means[:, 1]
eps_filter = filtered_state_means[:, 2]
eps_smooth = smoothed_state_means[:, 2]

plt.figure(0)
plt.title("Brownian motion plus noise")
plt.plot(times, x_true, label="true trajectory")
plt.plot(times, z_, "o", label="true noisy trajectory")
plt.plot(times, x_filter, "--", label="filter")
plt.plot(times, x_smooth, "--", label="smoother")
plt.xlabel("time (years)")
plt.legend()

plt.figure(1)
plt.title("Brownian motion plus noise, flukes")
plt.plot(times, f, label="true flukes")
plt.plot(times, f_filter, "--", label="filter")
plt.plot(times, f_smooth, "--", label="smoother")
plt.xlabel("time (years)")
plt.legend()

plt.figure(2)
plt.title("Brownian motion plus noise, flukes scatter")
plt.scatter(f, f_filter)
plt.xlabel("true flukes")
plt.ylabel("filtered flukes")
plt.legend()

# see autocorrelations:
plot_acf(np.diff(x_filter), lags=10, title=f"dx_filter ACF")
plot_pacf(np.diff(x_filter), lags=10, title=f"dx_filter PACF")
# plot_acf(np.diff(x_smooth), lags=10, title=f"dx_smooth ACF")
# plot_pacf(np.diff(x_smooth), lags=10, title=f"dx_smooth PACF")

plot_acf(f_filter, lags=10, title=f"f_filter ACF")
plot_pacf(f_filter, lags=10, title=f"f_filter PACF")
# plot_acf(f_smooth, lags=10, title=f"f_smooth ACF")
# plot_pacf(f_smooth, lags=10, title=f"f_smooth PACF")

plot_acf(eps_filter, lags=10, title=f"eps_filter ACF")
plot_pacf(eps_filter, lags=10, title=f"eps_filter PACF")
# plot_acf(eps_smooth, lags=10, title=f"eps_smooth ACF")
# plot_pacf(eps_smooth, lags=10, title=f"eps_smooth PACF")

# MC truth in what follows (for reference)

# visualizing flukes
# plt.figure(1)
# plt.title("Flukes")
# plt.plot(times, f)
# plt.xlabel("time (years)")
#
# plot_acf(np.diff(z_), lags=10, title=f"TRUE dz ACF")
# plot_pacf(np.diff(z_), lags=10, title=f"TRUE  dz PACF")
#
# plot_acf(np.diff(x_true), lags=10, title=f"TRUE dx ACF")
# plot_pacf(np.diff(x_true), lags=10, title=f"TRUE dx PACF")
#
# plot_acf(f, lags=10, title=f"TRUE flukes f = z - x ACF")
# plot_pacf(f, lags=10, title=f"TRUE flukes f = z - x PACF")
#
# plot_acf(epsilon, lags=10, title=f"TRUE epsilon ACF")
# plot_pacf(epsilon, lags=10, title=f"TRUE epsilon PACF")


show_plot()
