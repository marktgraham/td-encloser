import abc
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BaseFirstPass(abc.ABC):

    def _create_new_group(self, i, row, min_group_no):
        self.df_gxys.loc[i, 'group_no'] = np.max(self.df_gxys['group_no']) + 1
        self.df_gxys.loc[i, 'group_peak'] = True

        if self.plot == 'verbose':
            if ((np.abs(row['x']) <= 2) & (np.abs(row['y']) <= 2)) | (not cap):
                self.title = 'First Pass: %u Groups Found' % len(np.unique(self.df_gxys['group_no']))
                self.plot_groups(
                    x1=row['x'],
                    y1=row['y'],
                    alpha_group=min_group_no)

    def run_first_pass(self, selection, min_group_no, cap=True):
        w = (self.df_gxys.loc[selection, 'density'] >= self.delta_outer)

        # For each galaxy...
        for i, row in self.df_gxys.loc[selection].loc[w].iterrows():
            # If no groups above min_group_no exist
            if not len(self.df_gxys[self.df_gxys['group_no'] > min_group_no]):
                self._create_new_group(i, row, min_group_no)

            else:
                ww = (
                    (self.df_gxys['group_no'] > min_group_no) &
                    (self.df_gxys['group_peak'] == True))
                dist = np.sqrt(
                    (row['x'] - self.df_gxys.loc[ww, 'x']) ** 2 +
                    (row['y'] - self.df_gxys.loc[ww, 'y']) ** 2).values
                inds = np.argsort(dist)
                contour_group = np.zeros_like(dist)

                for j, row2 in (
                        self.df_gxys[ww].iloc[inds].reset_index(drop=True).iterrows()):
                    if j < 10:       # Only check nearest 10 groups
                        samples = int(max(
                            np.ceil(
                                (600 * dist[inds][j] * (dist[inds][j] + 1)** -2)),
                                2.0))
                        spacing = dist[inds][j] / samples

                        ff_mid = np.array([self.spline(
                            row['x'] + (row2['x'] - row['x']) * p,
                            row['y'] + (row2['y'] - row['y']) * p)[0][0] for p in np.linspace(0.0, 1.0, samples)]).round(5)

                        w = ff_mid <= np.round(row2['density'], 5)

                        # TODO: This check is now throwing up an error
                        #     assert np.sum(w) > 1, 'Not enough samples'

                        wwww = np.copy(w) if cap else (
                            np.ones_like(w)
                            .astype(bool))

                        if len(ff_mid[wwww]) * spacing > 0.1 + spacing:
                            grad_check = max(int(float('{:.0f}'.format(
                                0.1 / spacing))), 1)

                            ff_mid_diff = ff_mid[wwww][grad_check:] - \
                                          ff_mid[wwww][:-grad_check]

                        else:
                            ff_mid_diff = np.array(
                                [ff_mid[wwww][-1] - ff_mid[wwww][0]])

                        if (
                                (
                                    cap &
                                    (np.min(ff_mid_diff) >= self.mono) &
                                    (np.min(ff_mid) >= self.delta_outer) &
                                    (np.max(ff_mid) < 2 * row2['density'])) |
                                (
                                    (not cap) &
                                    (np.min(ff_mid_diff) >= self.mono / 10) &
                                    (np.min(ff_mid) >= self.delta_outer))):

                            contour_group[j] = 1
                            break

                if (np.max(contour_group) < 0) | (np.max(contour_group) > 1):
                    print('debug')

                # No connecting groups found
                if np.max(contour_group) == 0:
                    self._create_new_group(i, row, min_group_no)
                # One connecting group found
                elif np.max(contour_group) == 1:
                    www = contour_group == 1
                    self.df_gxys.loc[i, 'group_no'] = (
                        self.df_gxys.loc[ww, 'group_no'].iloc[inds]
                        .values[np.min(np.where(www)[0])])

                if self.plot == 'verbose':
                    if (
                            (np.abs(row['x']) <= 2) &
                            (np.abs(row['y']) <= 2)) | (cap == False):
                        self.title = 'First Pass: %u Groups Found' % len(
                            np.unique(self.df_gxys['group_no']))
                        self.plot_groups(
                            x1=row['x'],
                            y1=row['y'],
                            x2=row2['x'],
                            y2=row2['y'],
                            alpha_group=min_group_no)
