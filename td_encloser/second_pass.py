import abc
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BaseSecondPass(abc.ABC):

    def run_second_pass(self):
        # For groups above delta saddle, select galaxies between delta outer and delta saddle
        w = (
            (self.df_gxys['density'] >= self.delta_outer) &
            (self.df_gxys['density'] < self.delta_saddle) &
            np.isin(self.df_gxys['group_no'], self.df_gxys.loc[
                (self.df_gxys['density'] >= self.delta_peak) &
                (self.df_gxys['group_peak'] == 1), 'group_no']))

        if np.sum(w):  # If such galaxies exist
            print('Attempting to add %u remaining galaxies to existing groups...' % np.sum(w))
            # Second pass: see if they can join existing groups
            for i, row in self.df_gxys[w].iterrows():
                dist = np.sqrt(
                    (row['x'] - self.df_gxys['x']) ** 2 +
                    (row['y'] - self.df_gxys['y']) ** 2)
                inds = np.argsort(dist)  # Sort by distance
                ww = self.df_gxys['group_no'].iloc[inds][:self.n_merge] != 0

                # If at least one of the nearest n_merge - 1 neighbours is in a different group...
                if np.sum(ww):
                    near_max = self.df_gxys.loc[inds, 'density'][:self.n_merge].values.argmax()
                    if self.plot == 'verbose':
                        self.title = 'Second Pass: Chopping Border Galaxies...'
                        if (np.abs(row['x']) <= 2) & (np.abs(row['y']) <= 2):
                            self.plot_groups(
                                x1=row['x'],
                                y1=row['y'],
                                x2=(
                                    self.df_gxys['x']
                                    .iloc[inds][:self.n_merge]
                                    .iloc[near_max]),
                                y2=(
                                    self.df_gxys['y']
                                    .iloc[inds][:self.n_merge]
                                    .iloc[near_max]))

                    ff_mid = np.array(
                        [self.spline(
                            row['x'] + (
                                self.df_gxys['x'].iloc[inds][:self.n_merge].iloc[near_max] -
                                row['x']) * p,
                            row['y'] + (
                                self.df_gxys['y'].iloc[inds][:self.n_merge].iloc[near_max] -
                                row['y']) * p)[0][0] for p in np.linspace(0.0, 1.0, 11)])

                    self.df_gxys.loc[i, 'group_no'] = 0 if \
                        np.mean(
                            [
                                np.min(ff_mid),
                                np.max(
                                    self.df_gxys['density']
                                    .iloc[inds][:self.n_merge])]) < self.delta_saddle else self.df_gxys.loc[i, 'group_no']

            if self.target:
                if self.df_gxys['group_no'].iloc[self.ind_0] == 1:
                    print('Target galaxy has been added back in')
                elif self.df_gxys['group_no'].iloc[self.ind_0] == 0:
                    print('Target galaxy is still not in a group')

        if self.plot:
            self.title = 'Second Pass'
            self.plot_groups(legend=False)
            plt.gca().set_yticklabels([])
