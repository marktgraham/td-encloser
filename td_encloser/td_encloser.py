import abc
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .base import BaseTDENCLOSER
from .first_pass import BaseFirstPass
from .second_pass import BaseSecondPass
from .plotting import BasePlotting


class TDENCLOSER(
        BaseTDENCLOSER,
        BaseFirstPass,
        BaseSecondPass,
        BasePlotting):

    def _assign_group_mems(self):
        # add column with group members
        self.df_gxys['group_mem'] = np.select(
            [self.df_gxys['group_no'] != 0, self.df_gxys['group_no'] == 0],
            [
                self.df_gxys
                .groupby(['group_no'])
                ['group_peak']
                .transform('count')
                .values, 1])

        # set all galaxies with group_mem = 1 to group 0
        self.df_gxys['group_no'] = np.select(
            [self.df_gxys['group_mem'] == 1, self.df_gxys['group_mem'] != 1],
            [0, self.df_gxys['group_no']])

        self.df_gxys['group_peak'] = np.select(
            [self.df_gxys['group_no'] == 0, self.df_gxys['group_mem'] != 0],
            [False, self.df_gxys['group_peak']])

    def _assign_final_group_no(self):
        group_mapping = (
            self.df_gxys
            .loc[lambda x: x['group_peak'] is True]
            .sort_values(
                ['group_mem', 'density_rank'],
                ascending=[False, True])
            .reset_index(drop=True)
            .reset_index()
            .assign(group_no_new=lambda x: x['index'] + 1)
            .set_index('group_no')
            ['group_no_new']
            .to_dict())

        group_mapping[0] = 0

        self.df_gxys['group_no'] = self.df_gxys['group_no'].map(group_mapping)

    def find_groups(self):

        if self.plot:
            plt.figure(figsize=(12, 4))
            plt.subplot(131)
            plt.ylabel('y (Mpc)')

        target_density = self.df_gxys['density'].iloc[self.ind_target].item()

        # If target galaxy is above delta_outer...
        if (
                ((target_density >= self.delta_outer) & self.target) |
                (not self.target)):
            self.df_gxys.loc[0, 'group_no'] = 1
            self.df_gxys.loc[0, 'group_peak'] = True
            print('Starting first pass...')
            w = self.df_gxys.index.values > 0
            self.run_first_pass(selection=w, min_group_no=0)
            print('Completed in %.1f seconds' % (time.time() - self.start))

        else:
            self.df_gxys.loc[self.ind_target, 'group_no'] = 1
            self.grps = pd.DataFrame(
                {
                    'x': 0,
                    'y': 0,
                    'f': self.df_gxys['density'].iloc[self.ind_target],
                    'group_no': 1},
                index=[0])

        if (
                (self.df_gxys['group_no'].iloc[self.ind_target].item() > 1) &
                (self.target)):
            w = self.df_gxys['group_no'] == \
                self.df_gxys['group_no'].iloc[self.ind_target]
            ww = self.df_gxys['group_no'] == 1
            self.df_gxys.loc[ww, 'group_no'] = \
                self.df_gxys['group_no'].iloc[self.ind_target]
            self.df_gxys.loc[w, 'group_no'] = 1

        if self.plot:
            self.title = 'First Pass'
            self.plot_groups(legend=False)

            plt.tight_layout()
            if self.plot == 'verbose':
                if input('Continue? ') != '~':
                    pass
            plt.subplot(132)

        if self.target:
            if self.df_gxys['group_no'].iloc[self.ind_target] == 1:
                print('Target galaxy is in a group')
            elif self.df_gxys['group_no'].iloc[self.ind_target] == 0:
                print('Target galaxy is isolated')

        # Second pass: Break up Group 1

        if np.sum(self.df_gxys['group_no'] == 1) != 1:
            # For groups where the peak is greater than delta_saddle,
            # select galaxies below delta_saddle (and above delta outer)
            self.run_second_pass()

        if self.plot:
            if self.plot == 'verbose':
                if input('Continue? ') != '~':
                    pass
            plt.subplot(133)

        # Select galaxies between delta_outer and delta_saddle
        w_saddle = (
            (self.df_gxys['group_no'] == 0) &
            (self.df_gxys['density'] >= self.delta_outer))

        if np.sum(w_saddle):  # If such galaxies exist
            print(
                f'Attempting to form new groups from {w_saddle} '
                'remaining galaxies...')

            max_group_no = np.max(self.df_gxys['group_no'])
            print('Starting third pass...')
            self.title = \
                'Third Pass: Forming New Groups with Remaining Galaxies...'
            self.run_first_pass(
                selection=w_saddle,
                min_group_no=max_group_no + 1,
                cap=False)

            if self.target:
                if self.df_gxys['group_no'][self.ind_target] == 1:
                    print('Target galaxy is in a new group')
                elif self.df_gxys['group_no'][self.ind_target] == 0:
                    print('Target galaxy is still isolated')
        else:
            max_group_no = 0

        if self.target:
            if self.df_gxys['group_no'][self.ind_target] != 1:
                w = self.df_gxys['group_no'] == 1
                ww = self.df_gxys['group_no'] == \
                    self.df_gxys['group_no'].iloc[self.ind_target]
                self.df_gxys.loc[w, 'group_no'] = \
                    self.df_gxys['group_no'].iloc[self.ind_target]
                self.df_gxys.loc[ww, 'group_no'] = 1

        assert np.sum(self.df_gxys['group_no'] == 1) > 0, 'Problem!'
        if self.target:
            assert self.df_gxys['group_no'].iloc[self.ind_target] == 1, \
                'Problem!'
        assert np.sum(
            (self.df_gxys['density'] >= self.delta_outer) &
            (self.df_gxys['group_no'] == 0)) == 0, 'Problem!'

        if self.plot:
            self.title = 'Third Pass'
            self.plot_groups(
                legend=False,
                alpha_group = max_group_no + 1)
            plt.gca().set_yticklabels([])
            plt.subplots_adjust(
                left=0.05, right=0.98, top=0.92, bottom=0.12, wspace=0)

            if self.plot == 'verbose':
                if input('Continue? ') != '~':
                    pass

            plt.savefig(self.file + '.pdf')

        num_groups = len(
            self.df_gxys
            .loc[lambda x: x['group_no'] > 0, 'group_no']
            .drop_duplicates())

        print(
            f'Found {num_groups} groups in '
            f'{(time.time() - self.start):.1f} seconds')

        self._assign_group_mems()

        self._assign_final_group_no()

        return (
            self.df_gxys
            .sort_values(['group_no', 'density_rank'], ascending=[True, True])
            .reset_index(drop=True))
