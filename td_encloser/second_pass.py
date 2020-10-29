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
                df_sorted_by_distance = (
                    self.df_gxys
                    .copy()
                    .assign(
                        dist=lambda x: np.sqrt(
                            (galaxy['x'] - x['x']) ** 2 +
                            (galaxy['y'] - x['y']) ** 2))
                    .sort_values('dist'))

                df_neighbour_groups = df_sorted_by_distance[:self.n_merge]

                # If at least one of the nearest n_merge - 1 neighbours is in
                # a different group...
                if not df_neighbour_groups.loc[lambda x: x['group_no'] != 0].empty:

                    df_near_max = (
                        df_neighbour_groups
                        .loc[lambda x: x['density'] == x['density'].max()])

                    if self.plot == 'verbose':
                        self.title = 'Second Pass: Chopping Border Galaxies...'
                        if (galaxy['x'].abs() <= 2) & (galaxy['y'].abs() <= 2):
                            self.plot_groups(
                                x1=galaxy['x'],
                                y1=galaxy['y'],
                                x2=df_near_max['x'],
                                y2=df_near_max['y]'])

                    dist_from_galaxy_to_max = {
                        'x': df_near_max['x'] - galaxy['x'],
                        'y': df_near_max['y'] - galaxy['y']}

                    ff_mid = np.array([
                        self.spline(
                            galaxy['x'] + dist_from_galaxy_to_max['x'] * p,
                            galaxy['y'] + dist_from_galaxy_to_max['y'] * p)[0][0]
                        for p in np.linspace(0.0, 1.0, 11)])

                    if (
                            np.mean([np.min(ff_mid), df_near_max['density']])
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
