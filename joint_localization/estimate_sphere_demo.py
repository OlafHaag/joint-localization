import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d


#%% Ground truth values
mu_x = 8
mu_y = 13
mu_z = 20
mu_r = 5
sigma = 0.2


#%% random vectors
def sample_spherical(npoints, radius, sig=0.0, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    vec *= radius
    vec += np.random.normal(loc=0, scale=sig, size=npoints)
    return vec


#%% Create points on surface of unit sphere
xi, yi, zi = sample_spherical(100, mu_r, sigma)

# Move & scale points by ground truth
xi = np.abs(xi) + mu_x  # only one side of data for more realism.
yi = yi + mu_y
zi = zi + mu_z


#%% Function to optimize
def sum_squared(parameters, data):
    x_center = parameters[0]
    y_center = parameters[1]
    z_center = parameters[2]
    r_hat = parameters[3]
    # Extract axis components from point data
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    r = np.sqrt((x-x_center)**2 + (y-y_center)**2 + (z-z_center)**2)
    return np.sum((r-r_hat)**2)


#%%Run optimization
# Set initial values for parameters.
x0 = np.array([xi.mean(), yi.mean(), zi.mean(), np.ptp(xi)*0.5])
# Create data array that is closer to what we'll get from real world data.
points = np.array([xi, yi, zi]).T
solution = minimize(sum_squared, x0, args=points)

if solution.success:
    # Extract estimated parameters
    est_x = solution.x[0]
    est_y = solution.x[1]
    est_z = solution.x[2]
    est_r = solution.x[3]
    print("Sum of squared difference for radius:", solution.fun)
    print("number of iterations:", solution.nit)
    print("Estimated sphere parameters: x={}, y={}, z={}, radius={}".format(est_x, est_y, est_z, est_r))
else:
    print("ERROR: Optimization was not successful!")
    est_x = 0
    est_y = 0
    est_z = 0
    est_r = 0

#%% plot data
# Create wireframe data for ground truth visualization.
phi = np.linspace(0, np.pi, 10)
theta = np.linspace(0, 2 * np.pi, 20)
x_truth_wire = mu_r * np.outer(np.sin(theta), np.cos(phi)) + mu_x
y_truth_wire = mu_r * np.outer(np.sin(theta), np.sin(phi)) + mu_y
z_truth_wire = mu_r * np.outer(np.cos(theta), np.ones_like(phi)) + mu_z

# Create wireframe data for estimated sphere visualization.
phi = np.linspace(0, np.pi, 10)
theta = np.linspace(0, 2 * np.pi, 20)
x_est_wire = est_r * np.outer(np.sin(theta), np.cos(phi)) + est_x
y_est_wire = est_r * np.outer(np.sin(theta), np.sin(phi)) + est_y
z_est_wire = est_r * np.outer(np.cos(theta), np.ones_like(phi)) + est_z

fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d', 'aspect': 'equal'})
ax.plot_wireframe(x_truth_wire, y_truth_wire, z_truth_wire, color='k', alpha=0.3, rstride=1, cstride=1, label='ground truth')
ax.plot_wireframe(x_est_wire, y_est_wire, z_est_wire, color='lightblue', alpha=0.6, rstride=1, cstride=1, label='Estimate')
ax.scatter(xi, yi, zi, s=100, c='r', zorder=10, label='data')
plt.legend()
plt.show()
