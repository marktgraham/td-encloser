import abc
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BaseSecondPass(abc.ABC):

    def run_second_pass(self):
        # For groups above delta saddle, select galaxies between delta outer and delta saddle
        w = (
            (self.gxys.ff >= self.delta_outer) &
            (self.gxys.ff < self.delta_saddle) &
            np.isin(self.gxys.group_no, self.gxys.group_no[
                (self.gxys.ff >= self.delta_peak) &
                (self.gxys.group_peak == 1)]))

        if np.sum(w):  # If such galaxies exist
            print('Attempting to add %u remaining galaxies to existing groups...' % np.sum(w))
            # Second pass: see if they can join existing groups
            for row in self.gxys[w].itertuples():

                dist = np.sqrt(
                    (row.xx - self.gxys['xx']) ** 2 +
                    (row.yy - self.gxys['yy']) ** 2)
                inds = np.argsort(dist)  # Sort by distance
                ww = self.gxys.group_no.iloc[inds][:self.n_merge] != 0

                # If at least one of the nearest n_merge - 1 neighbours is in a different group...
                if np.sum(ww):
                    near_max = self.gxys.ff[inds][:self.n_merge].values.argmax()
                    if self.plot == 'verbose':
                        self.title = 'Second Pass: Chopping Border Galaxies...'
                        if (np.abs(row.xx) <= 2) & (np.abs(row.yy) <= 2):
                            self.plot_groups(
                                x1=row.xx,
                                y1=row.yy,
                                x2=(
                                    self
                                    .gxys
                                    .xx
                                    .iloc[inds][:self.n_merge]
                                    .iloc[near_max]),
                                y2=(
                                    self
                                    .gxys
                                    .yy
                                    .iloc[inds][:self.n_merge]
                                    .iloc[near_max]))

                    ff_mid = np.array(
                        [self.spline(
                            row.xx + (
                                self.gxys['xx'].iloc[inds][:self.n_merge].iloc[near_max] -
                                row.xx) * p,
                            row.yy + (
                                self.gxys['yy'].iloc[inds][:self.n_merge].iloc[near_max] -
                                row.yy) * p)[0][0] for p in np.linspace(0.0, 1.0, 11)])

                    self.gxys.at[row[0], 'group_no'] = 0 if \
                        np.mean(
                            [
                                np.min(ff_mid),
                                np.max(
                                    self
                                    .gxys
                                    .ff
                                    .iloc[inds][:self.n_merge])]) < self.delta_saddle else self.gxys.at[row[0], 'group_no']

            if self.target:
                if self.gxys.group_no.iloc[self.ind_0] == 1:
                    print('Target galaxy has been added back in')
                elif self.gxys.group_no.iloc[self.ind_0] == 0:
                    print('Target galaxy is still not in a group')

        if self.plot:
            self.title = 'Second Pass'
            self.plot_groups(legend=False)
            plt.gca().set_yticklabels([])

