import abc
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BaseSecondPass(abc.ABC):
    """ Class holding methods to run the seconddd pass. """

    def run_second_pass(self):
        """ Method to run the second pass. """
        # For groups above delta saddle, select galaxies between delta outer
        # and delta saddle
        df_between_delta_outer_delta_saddle = (
            self.df_gxys
            .loc[lambda x: x['density'] >= self.delta_outer]
            .loc[lambda x: x['density'] < self.delta_saddle]
            .loc[lambda x: x['group_no'].isin(
                self.df_gxys.loc[
                    (self.df_gxys['density'] >= self.delta_peak) &
                    (self.df_gxys['group_peak'] == 1), 'group_no'])])

        if not df_between_delta_outer_delta_saddle.empty:  # If such galaxies exist
            print(
                f'Attempting to add {len(df_between_delta_outer_delta_saddle)} '
                'remaining galaxies to existing groups...')
            # Second pass: see if they can join existing groups
            for i, galaxy in df_between_delta_outer_delta_saddle.iterrows():
                dist = np.sqrt(
                    (galaxy['x'] - self.df_gxys['x']) ** 2 +
                    (galaxy['y'] - self.df_gxys['y']) ** 2)
                inds = np.argsort(dist)  # Sort by distance

                df_neighbour_groups = self.df_gxys.iloc[inds][:self.n_merge]

                # If at least one of the nearest n_merge - 1 neighbours is in
                # a different group...
                if not df_neighbour_groups.loc[lambda x: x['group_no'] != 0].empty:
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
                        .iloc[near_max]) - galaxy['x']

                    dist_y = (
                        self.df_gxys['y']
                        .iloc[inds]
                        [:self.n_merge]
                        .iloc[near_max]) - galaxy['y']

                    ff_mid = np.array([
                        self.spline(
                            galaxy['x'] + dist_x * p,
                            galaxy['y'] + dist_y * p)[0][0]
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
