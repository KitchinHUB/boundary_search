from shapely.geometry import Polygon
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points
from scipy.optimize import minimize
from pyDOE2 import *
from scipy.stats.qmc import Sobol
from sklearn import svm
import matplotlib.tri as mtri
import os
import sys


class sample:
    '''
    Function takes in a starting center point, and certain hyper parameters.
    Outputs full sampling suite.
    '''

    def __init__(self, center_point, polygon, boundmin=0, boundmax=1):
        '''store inputs as global variables
        CENTER_POINT:  list of [x1, x2] values where the initial sampling
        should take place
        POLYGON: is a Polygon shape that defines the boundary apriori
        '''
        self.center = center_point
        self.polygon1 = polygon
        self.x, self.y = self.polygon1.exterior.xy
        self.cat = np.array([])
        self.X = np.array([[]])
        self.boundmax = boundmax
        self.boundmin = boundmin
        self.scale = 0.05*((np.max(self.x)-np.min(self.x)) *
                           (np.max(self.y)-np.min(self.y)))
        
    def classify_curve_point(self, x1, x2):
        rect = mplPath.Path(np.array(np.array([self.x, self.y]).T))
        point = (x1, x2)
        if rect.contains_point(point):
            return 1
        else:
            return 0

    def add_noise(self, X_next, c, w=10, seed=42):
        ''' add noise based on how far a point is from the actual boundary.
        Uses shapely geometry Point to compute the distance from the boundary
        to the point
        '''
        for i in range(len(X_next)):
            if c[i] == 1:
                sign = 1
            elif c[i] == 0:
                sign = -1

            # compute distance
            d = sign*self.polygon1.exterior.distance(Point(*X_next[i]))
            q = 1/(1+np.exp(-w*d))   # compute proability
            np.random.seed(seed)
            r = np.random.rand()
            thresh = (0.5 - np.abs(q-0.5))   # compute threshold for swap
            if r < thresh:
                c[i] = 1-c[i]
        return c

    def classify_X(self, X_next):
        '''classifies X_next points based on the calculation of whether it
        satisfies the desired output space '''
        cat_next = []
        for i in X_next:
            # See if points fall in desired output space
            c = self.classify_curve_point(*i)
            cat_next.append(c)
        cat = np.append(self.cat, np.array(cat_next))
        return cat

    def constraints(self, X_next):
        '''constraints for the next data points
        '''
        X_next[:, 0][X_next[:, 0] < self.boundmin] = self.boundmin
        X_next[:, 0][X_next[:, 0] > self.boundmax] = self.boundmax
        X_next[:, 1][X_next[:, 1] < self.boundmin] = self.boundmin
        X_next[:, 1][X_next[:, 1] > self.boundmax] = self.boundmax
        return X_next

    def bound_constraint(self, bound):
        '''bound_constraint takes bound and ensures that the bound
        that we compute is within the constraints.
        '''
        b = bound[(bound[:, 0] > self.boundmin) & (bound[:, 0] < self.boundmax) &
                  (bound[:, 1] > self.boundmin) & (bound[:, 1] < self.boundmax)]
        return b

    def acquisition(self, center, bound, step=0.2):
        poly = Polygon(bound)
        p1, p2 = nearest_points(poly, Point(*center))
        boundind = np.argmin(np.sum(np.abs(bound - [*p1.coords[0]]), axis=1))

        if boundind == 0:
            boundind = 2

        #get distance from the bound ind to the other points on the boundary
        disbound = np.sum((bound[:boundind] - bound[boundind])**2,
                          axis=1)**0.5

        p1 = bound[boundind]
        p2 = bound[:boundind][0]

        if np.sum(disbound > self.scale) != 0:
            p2 = bound[:boundind][disbound >= self.scale][-1]

        vector = (p2-p1)+0.001
        dx2dx1 = (vector[1]+0.0001)/(vector[0]+0.001)
        m_vec = np.array([1, dx2dx1])

        # The new x data point is a fixed distance away from the first point in the direction of the slope vector
        u_vec = m_vec/np.linalg.norm(m_vec)*np.sign(vector[0])
        c1 = p1 + step*u_vec
        c1 = self.constraints(np.array([c1]))[0]
        
        if (c1 == self.center[-1]).all():
            c1 = c1 + 2*step*u_vec
        if (c1 == self.center[-2]).all():
            c1 = c1 + 4*step*u_vec

        c1 = self.constraints(np.array([c1]))[0]
        return c1, u_vec

    def get_bound(self, X, cat):
        clf = svm.SVC(kernel='rbf', C=1000, probability=True)
        clf.fit(X, cat)
        X1 = X.T[0]
        X2 = X.T[1]

        # make predictions in the space
        grid_x, grid_y = np.mgrid[X1.min()-self.scale:X1.max()+self.scale:100j,
                                  X2.min()-self.scale:X2.max()+self.scale:100j]
        grid = np.stack([grid_x, grid_y], axis=-1)
        pt_test = clf.predict(grid.reshape(-1, 2)).reshape(*grid_x.shape)

        # plot results
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax = plt.gca()
        ax.scatter(X1, X2, c=cat, s=30, cmap=plt.cm.Paired)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')

        # plot decision boundary and margins
        cs = ax.contour(grid_x, grid_y, pt_test, colors='k',
                        levels=[0.5], alpha=0.5)
        plt.close(fig)

        bound = cs.allsegs[0][0]
        if len(cs.allsegs[0]) > 1:
            bound = np.concatenate((cs.allsegs[0]), axis=0)

        return bound

    def first_sample(self, n=8, dis=0.2, seed=42):
        ''' This function makes the first sample for the iteration. Given an
        initial center point, number of samples, and scaling factor for radius
        of samples, it will create the samples, calssify them, fit a model,
        and get a decision boundary.

        outputs: X, cat, area

        N: number of initial samples
        DIS: radius of sample distribution
        '''
        input_sampler = Sobol(d=2, seed=seed)
        # get random points around one edge of the solution
        self.X = dis*(input_sampler.random(n=n)-0.5) + self.center[0]
        #self.X = self.constraints(self.X)
        self.cat = self.classify_X(self.X)

        # ensure we actually get a curve
        while len(np.unique(self.cat)) < 2:
            self.cat = np.array([])
            input_sampler = Sobol(d=2, seed=seed)
            self.X = dis*(input_sampler.random(n=n)-0.5) + self.center[0]
            #self.X = self.constraints(self.X)
            self.cat = self.classify_X(self.X)
           
        return

    def iter_sample(self, ran_sam=True, conv_num=2, step=None, n=4, dis=None,
                    atol=0.01, centol=None, seed=42):
        '''iter_sample is the iterative process to find the boundary. It takes a
        number of hyperparameters, and outputs X, cat, bound, and area.
        
        RAM_SAM: checks to see if random sampling is true or folse. If false,
                 random sampling will not be used in each iteration. 
        CONV_NUM: number of iterations where area must not change
        STEP: Step size for the next sampling
        N: number of samples to test at each step
        DIS: radius of distribution for batch points
        ATOL: Tolerance for change in area ratio
        CENTOL: closeness for first and last center point
        '''
        j = 0
        self.area = np.array([])
        conv_ratio = np.array(np.repeat(10, conv_num))
        startend = False
        input_sampler = Sobol(d=2, seed=seed)
        
        if step is None:
            step = self.scale*5
        if centol is None:
            centol = self.scale
        if dis is None:
            dis = self.scale*2

        dis1 = dis
        if not ran_sam:
            n = 2

        self.center = np.array([self.center[-1], self.center[-1]])

        while (((conv_ratio > atol).all()|(startend > centol))&(len(self.X)<2000)):
        #for j in range(50):
            self.bound = self.get_bound(self.X, self.cat)
            #self.bound = self.bound_constraint(self.bound)
            poly = Polygon(self.bound)
            self.area = np.append(self.area, poly.area)

            
            c1, uvec = self.acquisition(self.center[-1], self.bound, step)
            
            if ((len(np.unique(self.cat[-n:])) > 1)&(dis1>dis)):
                dis1 *= 0.5
            elif (len(np.unique(self.cat[-n:])) == 1):
                if dis1 < (dis*4):
                    dis1 *= 2
                c1, uvec = self.acquisition(self.center[-2], self.bound, step)
            self.center = np.append(self.center, [c1], axis=0)

            if ran_sam:
                input_sampler = Sobol(d=2, seed=seed)
                X_next = dis1*(input_sampler.random(n=n)-0.5) + c1
            else:
                X_next = c1+dis1*np.array([[-uvec[1], uvec[0]],
                                           [uvec[1], -uvec[0]]])

            self.cat = self.classify_X(X_next)
            self.X = np.append(self.X, X_next, axis = 0)
            
            ratio = self.area[:-1]/self.area[1:]
            conv_ratio = np.abs(1-ratio)[-conv_num:]
            startend = np.sum((self.center[0] - self.center[-1])**2)**0.5
            j += 1

        return

    def sample_simulation(self, n1=8, dis1=0.2, ran_sam=True, conv_num=2, step=None, n2=4,
                          dis2=None, atol=0.01, centol=None, seed=42):
        ''' This function calls on the first sample function and the iter_sample
        functions to make the complete sampling simulation. It should return X,
        cat, bound, and the final area.

        N1: number of initial samples
        DIS1: radius of sample distribution
        CONV_NUM: number of iterations where area must not change
        STEP: Step size for the next sampling
        N2: number of samples to test at each step
        DIS2: radius of distribution for batch points
        ATOL: Tolerance for change in area ratio
        CENTOL: closeness for first and last center point
        '''
        self.first_sample(n=n1, dis=dis1, seed=seed)
        self.iter_sample(conv_num=conv_num, step=step, ran_sam=ran_sam,
                         n=n2, dis=dis2, atol=atol, centol=centol, seed=seed)
        self.plot_final(self.X, self.cat)
        return self.X, self.cat, self.bound, self.area[-1]

    def plot_final(self, X, cat):
        '''
        plot_final takes X and cat and plots the decision boundary, sampled points,
        their classification, and the 95% uncertainty interval.

        outputs: figure
        
        X: 2D array of data points with each row being a different data point
        cat: classification of each sampled point
        '''
        # fit initial model
        clf = svm.SVC(kernel='rbf', C=10000, probability=True)
        clf.fit(X, cat.ravel())
        
        # make predictions in the space 
        grid_x, grid_y = np.mgrid[0:X[:, 0].max()+0.1:100j,
                                  0:X[:, 1].max()+0.1:100j]
        grid = np.stack([grid_x, grid_y], axis=-1)
        pt_test = clf.predict(grid.reshape(-1, 2)).reshape(*grid_x.shape)
        
        # plot results
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
        ax1 = plt.gca()
        ax1.scatter(X[:, 0], X[:, 1], c=cat, s=30, cmap=plt.cm.Paired)
        
        # plot decision boundary and margins
        ax1.contour(grid_x, grid_y, pt_test,
                    colors='k', levels=[0.5], alpha=0.5)
        
        # plot support vectors
        ax1.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                    s=100, linewidth=1, facecolors='none')

        ax1.set_title('EOS Density Sampling')
        #ax1.set_xlim([self.boundmin, self.boundmax])
        #ax1.set_ylim([self.boundmin, self.boundmax])
        
        # plot real boundary
        x, y = self.polygon1.exterior.xy
        ax1.plot(x, y, color='k')
        ax1.scatter(self.center.T[0], self.center.T[1], color='green', s=120,
                    alpha = np.arange(0, len(self.center))/len(self.center))

        # plot probability
        predict_prob = clf.predict_proba(grid.reshape(-1, 2))[:, 0].reshape(*grid_x.shape)
        ax1.contourf(grid_x, grid_y, predict_prob, colors=['grey'],
                     levels=[0.025, 0.5, 0.975], alpha=0.2)

        fig.legend(['True Boundary', 'Data', '95% confidence'],
                   loc='upper right')
        
        fig.tight_layout()
        #plt.savefig(f'{k}-iteration.png')
        return fig


    def bound_point_density(self, r=0.05):
        '''computes the number of points within a radius r from the boundary'''
        polybound = Polygon(self.bound)

        disPoint = []
        for i in self.X:
            dis = polybound.exterior.distance(Point(*i))
            disPoint.append(dis)
        return np.sum(np.array(disPoint) <= r)/len(self.X)

    def avg_dis_bound(self):
        '''this computes the average distance of points to the bound whenever
        you run it
        '''
        polybound = Polygon(self.bound)

        disPoint = []
        for i in self.X:
            dis = self.polygon1.exterior.distance(Point(*i))
            disPoint.append(dis)
        disPoint = np.array(disPoint)
        return np.mean(disPoint)

    def area95_ratio(self, x1max, x2max):
        '''computes the ratio of areas of the 95% confidence iterval and the
        area of the domain space
        '''
        clf = svm.SVC(kernel='rbf', C=10000, probability=True)
        clf.fit(self.X, self.cat.ravel())

        grid_x, grid_y = np.mgrid[0:x1max:100j, 0:x2max:100j]
        grid = np.stack([grid_x, grid_y], axis=-1)
        pt_test = clf.predict(grid.reshape(-1, 2)).reshape(*grid_x.shape)

        fig, ax1 = plt.subplots(1, 1, figsize = (6, 5))
        ax1 = plt.gca()
        predict_prob = clf.predict_proba(grid.reshape(-1, 2))[:, 0].reshape(*grid_x.shape)
        ci = ax1.contourf(grid_x, grid_y, predict_prob, colors=['grey'],
                          levels=[0.025, 0.975], alpha=0.2)

        plt.close(fig)
        bound = ci.allsegs[0][0]
        if len(ci.allsegs[0]) > 1:
            bound = np.concatenate((ci.allsegs[0]), axis=0)
            
        poly95 = Polygon(bound)

        area95 = poly95.area/(self.polygon1.length*self.scale)
        return area95
