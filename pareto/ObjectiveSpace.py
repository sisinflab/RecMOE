import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pymoo.indicators.hv import HV

np.random.seed(7)

class ObjectivesSpace:
    def __init__(self, df, functions):
        self.functions = functions
        self.df = df[df.columns.intersection(self._constr_obj())]
        self.points = self._get_points()

    def _constr_obj(self):
        objectives = list(self.functions.keys())
        objectives.insert(0, 'model')
        return objectives

    def _get_points(self):
        pts = self.df.to_numpy()
        factors = np.array(list(map(lambda x: 1 if x == 'max' else -1, list(self.functions.values()))))
        pts[:, 1:] = pts[:, 1:] * factors
        # sort points by decreasing sum of coordinates: the point having the greatest sum will be non dominated
        pts = pts[pts[:, 1:].sum(1).argsort()[::-1]]
        # initialize a boolean mask for non dominated and dominated points (in order to be contrastive)
        non_dominated = np.ones(pts.shape[0], dtype=bool)
        dominated = np.zeros(pts.shape[0], dtype=bool)
        for i in range(pts.shape[0]):
            # process each point in turn
            n = pts.shape[0]
            # definition of Pareto optimality: for each point in the iteration, we find all points non dominated by
            # that point.
            mask1 = (pts[i + 1:, 1:] >= pts[i, 1:])
            mask2 = np.logical_not(pts[i + 1:, 1:] <= pts[i, 1:])
            non_dominated[i + 1:n] = (np.logical_and(mask1, mask2)).any(1)
            # A point could dominate another point, but it could also be dominated by a previous one in the iteration.
            # The following row take care of this situation by "keeping in memory" all dominated points in previous
            # iterations.
            dominated[i + 1:n] = np.logical_or(np.logical_not(non_dominated[i + 1:n]), dominated[i + 1:n])
        pts[:, 1:] = pts[:, 1:] * factors
        return pts[(np.logical_not(dominated))], pts[dominated]

    def get_nondominated(self):
        return pd.DataFrame(self.points[0], columns=self._constr_obj())

    def get_dominated(self):
        return pd.DataFrame(self.points[1], columns=self._constr_obj())

    def plot(self, not_dominated, dominated, r):
        not_dominated = not_dominated.values
        dominated = dominated.values
        fig = plt.figure()
        if not_dominated.shape[1] == 3:
            ax = fig.add_subplot()
            ax.scatter(dominated[:, 1], dominated[:, 2], color='red')
            ax.scatter(not_dominated[:, 1], not_dominated[:, 2], color='blue')
            ax.scatter(r[0], r[1], color='green')
            plt.show()
        elif not_dominated.shape[1] == 4:
            ax = fig.add_subplot(projection='3d')
            ax.scatter(dominated[:, 1], dominated[:, 2], dominated[:, 3], color='red')
            ax.scatter(not_dominated[:, 1], not_dominated[:, 2], not_dominated[:, 3], color='blue')
            ax.scatter(r[0], r[1], r[2], color='green')
            ax.set_xlim3d(not_dominated[:, 1].min(), not_dominated[:, 1].max())
            ax.set_ylim3d(not_dominated[:, 2].min(), not_dominated[:, 2].max())
            ax.set_zlim3d(not_dominated[:, 3].min(), not_dominated[:, 3].max())
            plt.show()
        else:
            print("Cannot print >3-dimensional objective funtion space")

    """
        @For: Spread
        @Output: Measures the range of a solution set
        @Tips: Higher the value, better extensity 
    """
    def maximum_spread(self):
        n_objs = self.points[0].shape[1]
        ms = 0
        for j in range(1, n_objs):
            ms += (max(self.points[0][:, j]) - min(self.points[0][:, j]))**2
        return np.sqrt(ms)

    """
        @For: Uniformity
        @Output: Measures the variation of the distance between solutions in a set.
        @Tips: lower the value, better the uniformity 
        @From: https://github.com/Valdecy/pyMultiobjective/blob/main/pyMultiobjective/util/indicators.py
    """
    def spacing(self):
        sol = np.copy(self.points[0][:, 1:])
        dm = np.zeros(sol.shape[0])
        for i in range(0, sol.shape[0]):
            try:
                dm[i] = min([np.linalg.norm(sol[i] - sol[j]) for j in range(0, sol.shape[0]) if i != j])
            except ValueError:
                return 0
        d_mean = np.mean(dm)
        spacing = np.sqrt(np.sum((dm - d_mean) ** 2) / sol.shape[0])
        return spacing

    """
        @For: Cardinality
        @Output: Considers the proportion of non-dominated solutions
        @Tips: Smaller value is preferable
    """
    def error_ratio(self):
        return self.points[0].shape[0] / (self.points[0].shape[0] + self.points[1].shape[0])

    def _get_pareto(self, sets):
        pts = sets.to_numpy()
        factors = np.array(list(map(lambda x: 1 if x == 'max' else -1, list(self.functions.values()))))
        pts[:, 1:] = pts[:, 1:] * factors
        # sort points by decreasing sum of coordinates: the point having the greatest sum will be non dominated
        pts = pts[pts[:, 1:].sum(1).argsort()[::-1]]
        # initialize a boolean mask for non dominated and dominated points (in order to be contrastive)
        non_dominated = np.ones(pts.shape[0], dtype=bool)
        dominated = np.zeros(pts.shape[0], dtype=bool)
        for i in range(pts.shape[0]):
            # process each point in turn
            n = pts.shape[0]
            # definition of Pareto optimality: for each point in the iteration, we find all points non dominated by
            # that point.
            mask1 = (pts[i + 1:, 1:] >= pts[i, 1:])
            mask2 = np.logical_not(pts[i + 1:, 1:] <= pts[i, 1:])
            non_dominated[i + 1:n] = (np.logical_and(mask1, mask2)).any(1)
            # A point could dominate another point, but it could also be dominated by a previous one in the iteration.
            # The following row take care of this situation by "keeping in memory" all dominated points in previous
            # iterations.
            dominated[i + 1:n] = np.logical_or(np.logical_not(non_dominated[i + 1:n]), dominated[i + 1:n])
        pts[:, 1:] = pts[:, 1:] * factors
        return pts[(np.logical_not(dominated))]

    """
            @For: Convergence
            @Type: Dominance-based QI
            @Output: Measures the relative quality between two sets on covergence and cardinality
    """
    def c_indicator(self, set_a):
        set_b = self.get_nondominated()
        sets = pd.concat([set_a, set_b], axis=0)
        not_dom = pd.DataFrame(self._get_pareto(sets), columns=self._constr_obj())
        c_ind = pd.merge(not_dom, set_b, how='inner', on=['model']).shape[0] / set_b.shape[0]
        return c_ind

    """
        @For: All Quality Aspects
        @Type: Volume-based
        @Output: A set that achieves the maximum HV value for a given problem will contain all Pareto optimal solutions.
    """
    def hypervolumes(self, r):
        factors = np.array(list(map(lambda x: -1 if x == 'max' else 1, list(self.functions.values()))))
        hv_pts = np.copy(self.points[0])
        hv_pts[:, 1:] = hv_pts[:, 1:] * factors
        r = r * factors
        not_dominated = pd.DataFrame(hv_pts, columns=self._constr_obj())
        x = not_dominated[list(self.functions.keys())]
        ind = HV(ref_point=r)
        return ind(np.array(x.values))