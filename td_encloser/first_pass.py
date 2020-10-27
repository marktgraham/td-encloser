import abc
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BaseFirstPass(abc.ABC):
    """ Class holding methods to run the first pass. """

    def _create_new_group(self, i, row, min_group_no):
        """ Method to create a new group. """
        self.df_gxys.loc[i, 'group_no'] = self.df_gxys['group_no'].max() + 1
        self.df_gxys.loc[i, 'group_peak'] = True

        if self.plot == 'verbose':
            if ((row['x'].abs() <= 2) & (row['y'].abs() <= 2)) | (not cap):
                self.title = (
                    "First Pass: "
                    f"{len(self.df_gxys['group_no'].drop_duplicates())} "
                    "Groups Found")
                self.plot_groups(
                    x1=row['x'],
                    y1=row['y'],
                    alpha_group=min_group_no)

    def _assign_to_existing_group(self, i, row, min_group_no, cap):
        """ Method to assign a galaxy to an existing group. """
        # select galaxies which are at a group peak
        group_peaks = (
            (self.df_gxys['group_no'] > min_group_no) &
            (self.df_gxys['group_peak']))

        # construct DataFrame
        df_group_peaks = (
            self.df_gxys
            .loc[group_peaks]
            .assign(
                # calculate distance from each peak to the galaxy
                dist_to_gxy=lambda group_peak: np.sqrt(
                    (row['x'] - group_peak['x']) ** 2 +
                    (row['y'] - group_peak['y']) ** 2),
                # create dummy variable
                contour_group=0)
            # sort peaks by distance from galaxy in order of nearest to
            # furthest
            .sort_values('dist_to_gxy')
            .reset_index(drop=True))

        # calculate number of samples along which the density is taken along
        # the connecting line between a peak and the galaxy
        df_group_peaks['samples'] = df_group_peaks.apply(
            lambda group_peak: int(max(
                np.ceil((
                    600 * group_peak['dist_to_gxy'] *
                    (group_peak['dist_to_gxy'] + 1) ** -2)),
                2.0)), axis=1)

        # calculate the spacing between density samples
        df_group_peaks['spacing'] = (
            df_group_peaks['dist_to_gxy'] /
            df_group_peaks['samples'])

        # Only check nearest 10 groups
        for j, group_peak in df_group_peaks[:10].iterrows():

            ff_mid = np.array([
                self.spline(
                    row['x'] + (group_peak['x'] - row['x']) * p,
                    row['y'] + (group_peak['y'] - row['y']) * p)[0][0]
                for p in np.linspace(0.0, 1.0, group_peak['samples'])]).round(5)

            ff_lower_mid = ff_mid <= np.round(group_peak['density'], 5)

            # TODO: This check is now throwing up an error
            #     assert np.sum(w) > 1, 'Not enough samples'

            if not cap:
                ff_lower_mid = np.ones_like(ff_lower_mid).astype(bool)

            if (
                    len(ff_mid[ff_lower_mid]) * group_peak['spacing'] >
                    0.1 + group_peak['spacing']):
                grad_check = max(int(float('{:.0f}'.format(
                    0.1 / group_peak['spacing']))), 1)

                ff_mid_diff = (
                    ff_mid[ff_lower_mid][grad_check:] -
                    ff_mid[ff_lower_mid][:-grad_check])

            else:
                ff_mid_diff = np.array([
                    ff_mid[ff_lower_mid][-1] -
                    ff_mid[ff_lower_mid][0]])

            if (
                    (
                        cap &
                        (np.min(ff_mid_diff) >= self.mono) &
                        (np.min(ff_mid) >= self.delta_outer) &
                        (np.max(ff_mid) < 2 * group_peak['density'])) |
                    (
                        (not cap) &
                        (np.min(ff_mid_diff) >= self.mono / 10) &
                        (np.min(ff_mid) >= self.delta_outer))):

                df_group_peaks.loc[j, 'contour_group'] = 1
                break

        if (
                (np.max(df_group_peaks['contour_group']) < 0) |
                (np.max(df_group_peaks['contour_group']) > 1)):
            print('debug')

        if not df_group_peaks.loc[lambda x: x['contour_group'] == 1].empty:
            # One connecting group found
            self.df_gxys.loc[i, 'group_no'] = \
                df_group_peaks.loc[
                    lambda x: x['contour_group'] == 1, 'group_no'].item()
        else:
            # No connecting groups found
            self._create_new_group(i, row, min_group_no)

        if self.plot == 'verbose':
            if (
                    (np.abs(row['x']) <= 2) &
                    (np.abs(row['y']) <= 2)) | (cap is False):
                self.title = (
                    'First Pass: '
                    f"{len(self.df_gxys['group_no'].drop_duplicates())} "
                    'Groups Found')
                self.plot_groups(
                    x1=row['x'],
                    y1=row['y'],
                    x2=group_peak['x'],
                    y2=group_peak['y'],
                    alpha_group=min_group_no)

    def run_first_pass(self, selection, min_group_no, cap=True):
        """ Method to run the first pass. """
        # For each galaxy...
        for i, row in (
                self.df_gxys
                .loc[selection]
                .loc[lambda x: x['density'] >= self.delta_outer]
                .iterrows()):
            # If no groups above min_group_no exist
            if not len(self.df_gxys[self.df_gxys['group_no'] > min_group_no]):
                self._create_new_group(i, row, min_group_no)

            else:
                self._assign_to_existing_group(i, row, min_group_no, cap)
