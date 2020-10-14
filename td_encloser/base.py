import abc
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BaseTDENCLOSER(abc.ABC):

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

        assert plot in [False, 'quick', 'verbose'], 'plot is False, quick or verbose'

        '''
        Routine to find groups using the hopping method.
        We want to know which galaxies are in the same group as the target galaxy.
        group_no = 1 if galaxy belongs to the same group as the target galaxy

        :param plot: If true, plots what's going on
        :return:
        '''

        self.start = time.time()

        # rotate galaxy coordinates for interpolation....
        xx_, yy_ = yy.copy(), xx.copy()
        # Calculate density at each galaxy
        ff = np.array([spline(y, x) for (y, x) in zip(yy_, xx_)])[:, 0][:, 0]

        # Sort by density in descending order
        inds = np.argsort(ff)[::-1]
        self.inds_undo = np.argsort(inds)
        self.gxys = pd.DataFrame({
            'xx': xx[inds],
            'yy': yy[inds],
            'ff': ff[inds],
            'group_no': 0,
            'group_peak': 0})
        self.ind_0 = np.where(inds == 0)[0][0]       # indice of the target galaxy

        xxx = np.linspace(
            np.round(np.min(self.gxys['xx'])).astype(int),
            np.round(np.max(self.gxys['xx'])).astype(int),
            501)
        yyy = np.linspace(
            np.round(np.min(self.gxys['yy'])).astype(int),
            np.round(np.max(self.gxys['yy'])).astype(int),
            501)
        fff = spline(yyy, xxx)
        xxx, yyy = np.meshgrid(xxx, yyy)
        xxx, yyy = yyy.copy(), xxx.copy()         # rotate grid for contours

        self.contours = pd.DataFrame({
            'x': xxx.ravel(),
            'y': yyy.ravel(),
            'f': fff.ravel()})

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
