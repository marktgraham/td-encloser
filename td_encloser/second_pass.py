import abc
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BaseSecondPass(abc.ABC):

    def run_second_pass(self):
        # For groups above delta saddle, select galaxies between delta outer
        # and delta saddle
        between_delta_outer_delta_saddle = (
            (self.df_gxys['density'] >= self.delta_outer) &
            (self.df_gxys['density'] < self.delta_saddle) &
            np.isin(self.df_gxys['group_no'], self.df_gxys.loc[
                (self.df_gxys['density'] >= self.delta_peak) &
                (self.df_gxys['group_peak'] == 1), 'group_no']))

        if between_delta_outer_delta_saddle.sum():  # If such galaxies exist
            print(
                f'Attempting to add {between_delta_outer_delta_saddle.sum()} '
                'remaining galaxies to existing groups...')
            # Second pass: see if they can join existing groups
            for i, row in (
                    self.df_gxys
                    .loc[between_delta_outer_delta_saddle]
                    .iterrows()):
                dist = np.sqrt(
                    (row['x'] - self.df_gxys['x']) ** 2 +
                    (row['y'] - self.df_gxys['y']) ** 2)
                inds = np.argsort(dist)  # Sort by distance

                neighbours_in_different_group = (
                    self.df_gxys
                    ['group_no']
                    .iloc[inds]
                    [:self.n_merge] != 0)

                # If at least one of the nearest n_merge - 1 neighbours is in
                # a different group...
                if neighbours_in_different_group.sum():
                    near_max = (
                        self.df_gxys
                        .loc[inds, 'density']
                        [:self.n_merge]
                        .values.argmax())

                    if self.plot == 'verbose':
                        self.title = 'Second Pass: Chopping Border Galaxies...'
                        if (row['x'].abs() <= 2) & (row['y'].abs() <= 2):
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

                    dist_x = (
                        self.df_gxys['x']
                        .iloc[inds]
                        [:self.n_merge]
                        .iloc[near_max]) - row['x']

                    dist_y = (
                        self.df_gxys['y']
                        .iloc[inds]
                        [:self.n_merge]
                        .iloc[near_max]) - row['y']

                    ff_mid = np.array([
                        self.spline(
                            row['x'] + dist_x * p,
                            row['y'] + dist_y * p)[0][0]
                        for p in np.linspace(0.0, 1.0, 11)])

                    ff_density_neighbour_max = np.max(
                        self.df_gxys['density'].iloc[inds][:self.n_merge])

                    if (
                            np.mean([np.min(ff_mid), ff_density_neighbour_max])
                            < self.delta_saddle):
                        self.df_gxys.loc[i, 'group_no'] = 0

            if self.target:
                if self.df_gxys['group_no'].iloc[self.ind_0] == 1:
                    print('Target galaxy has been added back in')
                elif self.df_gxys['group_no'].iloc[self.ind_0] == 0:
                    print('Target galaxy is still not in a group')

        if self.plot:
            self.title = 'Second Pass'
            self.plot_groups(legend=False)
            plt.gca().set_yticklabels([])
