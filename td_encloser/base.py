import abc
import time
import functools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BaseTDENCLOSER(abc.ABC):
    """ Base class for TD-ENCLOSER. """

    def _calculate_density_at_location(self, location):
        """ Method to return the density at a location. """
        return self.spline(location['x'], location['y'])

    def __init__(
            self,
            xx,
            yy,
            spline,
            plot=False,
            n_merge=4,
            delta_outer=1.6,
            delta_saddle=4.0,
            delta_peak=4.8,
            file='group_finder',
            target=True,
            mono=0):

        assert plot in [False, 'quick', 'verbose'], \
            'plot is False, quick or verbose'

        '''
        Routine to find groups using the hopping method.
        We want to know which galaxies are in the same group as the target
        galaxy. group_no = 1 if galaxy belongs to the same group as the target
        galaxy.

        :param plot: If true, plots what's going on
        :return:
        '''

        self.xx = xx
        self.yy = yy

        self.start = time.time()

        self.spline = spline
        self.plot = plot
        self.delta_outer = delta_outer
        self.delta_saddle = delta_saddle
        self.delta_peak = delta_peak
        self.levels = np.array([delta_outer, delta_saddle])
        self.n_merge = int(n_merge)
        self.target = target
        self.file = file

        self.pause = True if plot == 'verbose' else False
        self.mono = mono
        self.array = np.zeros(1)

    @property
    @functools.lru_cache()
    def df_gxys(self):
        """ Method to return a DataFrame representation of the galaxy
        catalogue. """
        df_gxys = pd.DataFrame({'x': self.xx, 'y': self.yy})

        # Calculate density at each galaxy
        df_gxys['density'] = (
            df_gxys
            .apply(self._calculate_density_at_location, axis=1)
            .apply(lambda x: x[0][0]))

        df_gxys = (
            df_gxys
            .assign(
                # add density rank
                density_rank=lambda x: x['density'].rank(ascending=False),
                group_no=0,
                group_peak=False)
            .sort_values(['density_rank'])
            .reset_index(drop=True))

        return df_gxys

    @property
    @functools.lru_cache()
    def ind_target(self):
        """ Method to return the index of the target galaxy. """
        # need to check this
        return (
            self.df_gxys
            .loc[lambda x: x['density_rank'] == x['density_rank'].max()]
            .index)

    @property
    @functools.lru_cache()
    def df_grid(self):
        """ Method to return a DataFrame of a grid for contours. """
        df_grid = pd.DataFrame({
            'x': np.linspace(
                np.round(np.min(self.df_gxys['x'])).astype(int),
                np.round(np.max(self.df_gxys['x'])).astype(int),
                501),
            'y': np.linspace(
                np.round(np.min(self.df_gxys['y'])).astype(int),
                np.round(np.max(self.df_gxys['y'])).astype(int),
                501)
        })

        df_grid['density'] = df_grid.apply(
            self._calculate_density_at_location,
            axis=1)

        return df_grid

    @property
    @functools.lru_cache()
    def df_contours(self):
        """ Method to return a DataFrame of density. """
        # rotate grid for contours
        yyy, xxx = np.meshgrid(self.df_grid['x'], self.df_grid['y'])

        breakpoint()

        return pd.DataFrame({
            'x': xxx.ravel(),
            'y': yyy.ravel(),
            'f': self.df_grid['density'].ravel()})
