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

    def find_groups(self):

        if self.plot:
            plt.figure(figsize=(12, 4))
            plt.subplot(131)
            plt.ylabel('y (Mpc)')

        # If target galaxy is above delta_outer...
        if ((
                self.gxys.ff.iloc[self.ind_0] >= self.delta_outer)
                & self.target) | (not self.target):
            self.gxys.at[0, 'group_no'], self.gxys.at[0, 'group_peak'] = 1, 1
            print('Starting first pass...')
            w = self.gxys.index.values > 0
            self.run_first_pass(selection=w, min_group_no=0)
            print('Completed in %.1f seconds' % (time.time() - self.start))

        else:
            self.gxys.at[self.ind_0, 'group_no'] = 1
            self.grps = pd.DataFrame(
                {
                    'x': 0,
                    'y': 0,
                    'f': self.gxys.ff.iloc[self.ind_0],
                    'group_no': 1},
                index=[0])

        if (self.gxys.group_no.iloc[self.ind_0] > 1) & (self.target):
            w = self.gxys.group_no == self.gxys.group_no.iloc[self.ind_0]
            ww = self.gxys.group_no == 1
            self.gxys.at[ww, 'group_no'] = self.gxys.group_no.iloc[self.ind_0]
            self.gxys.at[w, 'group_no'] = 1

        if self.plot:
            self.title = 'First Pass'
            self.plot_groups(legend=False)

            plt.tight_layout()
            if self.plot == 'verbose':
                if input('Continue? ') != '~':
                    pass
            plt.subplot(132)

        if self.target:
            if self.gxys.group_no.iloc[self.ind_0] == 1:
                print('Target galaxy is in a group')
            elif self.gxys.group_no.iloc[self.ind_0] == 0:
                print('Target galaxy is isolated')

        # Second pass: Break up Group 1

        if np.sum(self.gxys.group_no == 1) != 1:
            # For groups where the peak is greater than delta_saddle,
            # select galaxies below delta_saddle (and above delta outer)
            self.run_second_pass()

        if self.plot:
            if self.plot == 'verbose':
                if input('Continue? ') != '~':
                    pass
            plt.subplot(133)

        # Select galaxies between delta_outer and delta_saddle
        w_saddle = (self.gxys.group_no == 0) & (self.gxys.ff >= self.delta_outer)

        if np.sum(w_saddle):  # If such galaxies exist
            print('Attempting to form new groups from %u remaining galaxies...' % np.sum(w_saddle))

            max_group_no = np.max(self.gxys.group_no)
            print('Starting third pass...')
            self.title = 'Third Pass: Forming New Groups with Remaining Galaxies...'
            self.run_first_pass(
                selection=w_saddle,
                min_group_no=max_group_no + 1,
                cap=False)

            if self.target:
                if self.gxys.group_no[self.ind_0] == 1:
                    print('Target galaxy is in a new group')
                elif self.gxys.group_no[self.ind_0] == 0:
                    print('Target galaxy is still isolated')
        else:
            max_group_no = 0

        if self.target:
            if self.gxys.group_no[self.ind_0] != 1:
                w = self.gxys.group_no == 1
                ww = self.gxys.group_no == self.gxys.group_no[self.ind_0]
                self.gxys.at[w, 'group_no'] = self.gxys.group_no[self.ind_0]
                self.gxys.at[ww, 'group_no'] = 1

        assert np.sum(self.gxys.group_no == 1) > 0, 'Problem!'
        if self.target:
            assert self.gxys.group_no.iloc[self.ind_0] == 1, 'Problem!'
        assert np.sum(
            (self.gxys.ff >= self.delta_outer) &
            (self.gxys.group_no == 0)) == 0, 'Problem!'

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

        print('Found %u groups in %.1f seconds' % (
            len(np.unique(self.gxys.group_no)), time.time() - self.start))

        self.gxys = self.gxys.iloc[self.inds_undo].reset_index(drop=True)

        return self.gxys