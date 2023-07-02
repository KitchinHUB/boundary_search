from shapely.geometry import Polygon
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from pyDOE2 import *
from scipy.stats.qmc import Sobol
from sklearn import svm
import random


class sample:
    """
    Function takes in a starting center point, and certain hyper parameters.
    Outputs full sampling suite.
    """

    def __init__(self, center_point, polygon, boundmin=0, boundmax=1):
        """store inputs as global variables
        CENTER_POINT:  list of [x1, x2] values where the initial sampling
        should take place
        POLYGON: is a Polygon shape that defines the boundary apriori
        """
        self.center_points = np.array([center_point])
        self.polygon1 = polygon
        self.x, self.y = self.polygon1.exterior.xy
        self.cat = np.array([])
        self.X = np.array([[]])
        self.boundmax = boundmax
        self.boundmin = boundmin
        self.scale = 0.05 * (
            (np.max(self.x) - np.min(self.x)) * (np.max(self.y) - np.min(self.y))
        )

    def classify_curve_point(self, x1, x2):
        rect = mplPath.Path(np.array(np.array([self.x, self.y]).T))
        point = (x1, x2)
        if rect.contains_point(point):
            return 1
        else:
            return 0

    def add_noise(self, X_next, c, w=10, seed=42):
        """Add noise based on how far a point is from the actual boundary.
        Uses shapely geometry Point to compute the distance from the boundary
        to the point.
        """

        for i in range(len(X_next)):
            if c[i] == 1:
                sign = 1
            elif c[i] == 0:
                sign = -1

            # compute distance
            d = sign * self.polygon1.exterior.distance(Point(*X_next[i]))
            q = 1 / (1 + np.exp(-w * d))  # compute probability
            np.random.seed(seed)
            r = np.random.rand()
            thresh = 0.5 - np.abs(q - 0.5)  # compute threshold for swap
            if r < thresh:
                c[i] = 1 - c[i]
        return c

    def classify_X(self, X_next):
        """Classifies X_next points based on the calculation of whether it
        satisfies the desired output space"""
        cat_next = []
        for i in X_next:
            # See if points fall in desired output space
            c = self.classify_curve_point(*i)
            cat_next.append(c)

        return np.array(cat_next)

    def constraints(self, X_next):
        """Constraints for the next data points."""
        X_next[:, 0][X_next[:, 0] < self.boundmin] = self.boundmin
        X_next[:, 0][X_next[:, 0] > self.boundmax] = self.boundmax
        X_next[:, 1][X_next[:, 1] < self.boundmin] = self.boundmin
        X_next[:, 1][X_next[:, 1] > self.boundmax] = self.boundmax
        return X_next

    def bound_constraint(self, bound):
        """Takes bound and ensures that the bound that we compute is within the
        constraints.

        """
        b = bound[
            (bound[:, 0] > self.boundmin)
            & (bound[:, 0] < self.boundmax)
            & (bound[:, 1] > self.boundmin)
            & (bound[:, 1] < self.boundmax)
        ]
        return b

    def acquisition(self, bound, r=0.1, min_points=1, samples=8):
        """Acquisition takes X, the boundary, minimum number of points, and
        radius for which to look at points. It filters the boundary points by
        those that do not have many points sampled near by, and then filters
        again by taking a random sampling of 8 points within that subset.
        """

        b = bound[np.array(self.compute_density(self.X, bound, r=r)) < min_points]
        X_next = b
        if len(b) > samples:
            random.seed(42)
            ind_bound = np.sort(random.sample(list(range(0, len(b))), samples))
            X_next = b[ind_bound]
        return X_next

    def compute_density(self, X, bound, r=0.1):
        """Finds the number of points within a radius r.
        """
        density = []
        for i in bound:
            curvePoint = i
            point_bool_in_rad = np.sum((X - curvePoint) ** 2, axis=1) <= r**2
            density.append(np.sum(point_bool_in_rad))  # get total point densit
        return density

    def get_bound(self, X, cat):
        """Get bound uses a svm classifier to make a model with self.X, and
        self.cat. We then enumerate the space based on the current sample domain
        and then make predictions based on the trained model. With the
        predictions, a contour plot is made and the bound is extracted with a 2d
        array of vertices.

        """
        clf = svm.SVC(kernel="rbf", C=1000, probability=True)
        clf.fit(X, cat)

        # make predictions in the space
        grid_x, grid_y = np.mgrid[
            0 : X[:, 0].max() + 0.1 : 100j, 0 : X[:, 1].max() + 0.1 : 100j
        ]
        grid = np.stack([grid_x, grid_y], axis=-1)
        pt_test = clf.predict(grid.reshape(-1, 2)).reshape(*grid_x.shape)

        # plot results
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax = plt.gca()
        ax.scatter(X[:, 0], X[:, 1], c=cat, s=30, cmap=plt.cm.Paired)
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")

        # plot decision boundary and margins
        cs = ax.contour(grid_x, grid_y, pt_test, colors="k", levels=[0.5], alpha=0.5)
        plt.close(fig)

        bound = cs.allsegs[0][0]
        if len(cs.allsegs[0]) > 1:
            bound = np.concatenate((cs.allsegs[0]), axis=0)

        return bound

    def first_sample(self, ni=16, x1max=2, x2max=5, w=10, seed=42):
        """This function makes the first sample for the iteration. Given an
        initial center point, number of samples, and scaling factor for radius
        of samples, it will create the samples, classify them, fit a model,
        and get a decision boundary.

        outputs: X, cat, area
        """

        input_sampler = Sobol(d=2, seed=seed)

        # get random points around one edge of the solution
        self.X = input_sampler.random(n=ni)
        X1 = self.X.T[0] * x1max
        X2 = self.X.T[1] * x2max
        self.X = np.array([X1, X2]).T
        X_next = self.X

        self.cat = self.classify_X(X_next)
        self.cat = self.add_noise(X_next, self.cat, w=w)

        # ensure we actually get a curve
        while len(np.unique(self.cat)) < 2:
            self.X = input_sampler.random(n=ni)
            X1 = self.X.T[0] * x1max
            X2 = self.X.T[1] * x2max
            self.X = np.array([X1, X2]).T
            X_next = self.X

            self.cat = np.array([])
            self.cat = self.classify_X(X_next)
            self.cat = self.add_noise(X_next, self.cat, w=w)

        self.bound = self.get_bound(self.X, self.cat)
        self.bound = self.bound_constraint(self.bound)

        self.area = [0.001, Polygon(self.bound).area]
        return

    def iter_sample(
        self, w=10, min_points=1, r_acq=0.2, tol=0.0001, conv_trials=2, samples=8
    ):
        """This function builds off the first sample and runs a sequential
        sampling trial. Is it necessary to have initialized X, cat, and
        center_points. It runs a loop that will converge if two convergence
        criteria are met: The ratio of predicted areas should not change between
        n last areas, and the minimum number of points within a radius r has to
        be no less than min_den.

        W: weight factor that determines the likelihood of noise
        MIN_POINTS: The minimum number of points within a radius for acquisition.
        R_ACQ: Radius that dictates where the MIN_POINTS must fall withing.
        TOL: The area tolerance for convergence
        CONV_TRIALS: The number of trials that must match the area convergence
                     condition.

        """
        k = 0
        conv_ratio = np.repeat(100, conv_trials)
        X_next = [10]
        self.seqind = [len(self.X)]

        # while ((conv_ratio > tol).all()):
        while len(X_next) > 0:
            bound_subset = self.bound[
                np.array(self.compute_density(self.X, self.bound, r=r_acq)) < min_points
            ]

            X_next = self.acquisition(
                bound_subset, r=r_acq, min_points=min_points, samples=samples
            )

            # Update data
            self.X = np.append(self.X, X_next, axis=0)

            # classify X_next
            cat = self.classify_X(X_next)
            cat = self.add_noise(X_next, cat, w=w)
            self.cat = np.append(self.cat, cat)

            self.bound = self.get_bound(self.X, self.cat)
            self.bound = self.bound_constraint(self.bound)

            # compute area of the bound
            self.area = np.append(self.area, Polygon(self.bound).area)

            ratio = self.area[:-1] / self.area[1:]
            conv_ratio = np.abs(1 - ratio)[-conv_trials:]

            self.seqind.append(len(self.X))
            k += 1
        return

    def sample_simulation(
        self,
        ni=16,
        x1max=2,
        x2max=5,
        w=10,
        min_points=1,
        r_acq=0.2,
        tol=0.0001,
        conv_trials=2,
        samples=8, 
        seed=42
    ):
        """This function calls on the first sample function and the iter_sample
        functions to make the complete sampling simulation. It should return X,
        cat, bound, and the final area.

        N2: The number of sample points per iteration
        DIS2: The spread distribution of the sample points per iteration
        W: weight factor that determines the likelihood of noise
        MIN_DEN: The minimum number of points within a radius for convergence.
        TOL: The area tolerance for convergence
        CONV_TRIALS: The number of trials that must match the area convergence
                     condition.
        DOMAIN_STEP: The domain step size when the domain has not maxed yet.
        R: Radius for acquitision function to determine point and class density.
        """
        self.first_sample(ni=ni, x1max=x1max, x2max=x2max, w=w, seed = seed)
        self.iter_sample(
            w=w,
            min_points=min_points,
            r_acq=r_acq,
            tol=tol,
            conv_trials=conv_trials,
            samples=samples
        )

        return self.X, self.cat, self.bound, self.area[-1]

    def plot_final(self, X, cat):
        # fit initial model
        clf = svm.SVC(kernel="rbf", C=1000, probability=True)
        clf.fit(X, cat.ravel())

        # make predictions in the space
        grid_x, grid_y = np.mgrid[
            0 : X[:, 0].max() + 0.1 : 100j, 0 : X[:, 1].max() + 0.1 : 100j
        ]
        grid = np.stack([grid_x, grid_y], axis=-1)
        pt_test = clf.predict(grid.reshape(-1, 2)).reshape(*grid_x.shape)

        # plot results
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
        ax1 = plt.gca()
        ax1.scatter(X[:, 0], X[:, 1], c=cat, s=30, cmap=plt.cm.Paired)

        # plot decision boundary and margins
        ax1.contour(grid_x, grid_y, pt_test, colors="k", levels=[0.5], alpha=0.5)

        # plot support vectors
        ax1.scatter(
            clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=100,
            linewidth=1,
            facecolors="none",
        )

        ax1.set_title("Broad Initial Sampling")
        ax1.set_xlim([self.boundmin, self.boundmax])
        ax1.set_ylim([self.boundmin, self.boundmax])

        # plot real boundary
        x, y = self.polygon1.exterior.xy
        ax1.plot(x, y, color="k")

        # plot probability
        predict_prob = clf.predict_proba(grid.reshape(-1, 2))[:, 0].reshape(
            *grid_x.shape
        )
        ax1.contourf(
            grid_x,
            grid_y,
            predict_prob,
            colors=["grey"],
            levels=[0.025, 0.5, 0.975],
            alpha=0.2,
        )

        fig.legend(["True Boundary", "Data", "95% confidence"], loc="upper right")

        fig.tight_layout()
        return fig

    def bound_point_density(self, r=0.05):
        """Computes the number of points within a radius r from the boundary."""
        polybound = Polygon(self.bound)

        disPoint = []
        for i in self.X:
            dis = polybound.exterior.distance(Point(*i))
            disPoint.append(dis)
        return np.sum(np.array(disPoint) <= r) / len(self.X)

    def avg_dis_bound(self):
        """This computes the average distance of points to the bound.
        """

        disPoint = []
        for i in self.X:
            dis = self.polygon1.exterior.distance(Point(*i))
            disPoint.append(dis)
        disPoint = np.array(disPoint)
        return np.mean(disPoint)

    def area95_ratio(self, x1max, x2max):
        """Computes the ratio of areas of the 95% confidence interval and the
        area of the domain space.
        """
        clf = svm.SVC(kernel="rbf", C=10000, probability=True)
        clf.fit(self.X, self.cat.ravel())

        grid_x, grid_y = np.mgrid[0:x1max:100j, 0:x2max:100j]
        grid = np.stack([grid_x, grid_y], axis=-1)

        fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
        ax1 = plt.gca()
        predict_prob = clf.predict_proba(grid.reshape(-1, 2))[:, 0].reshape(
            *grid_x.shape
        )
        ci = ax1.contourf(
            grid_x,
            grid_y,
            predict_prob,
            colors=["grey"],
            levels=[0.025, 0.975],
            alpha=0.2,
        )

        bound = ci.allsegs[0][0]
        if len(ci.allsegs[0]) > 1:
            bound = np.concatenate((ci.allsegs[0]), axis=0)

        poly95 = Polygon(bound)

        area95 = poly95.area
        return area95
