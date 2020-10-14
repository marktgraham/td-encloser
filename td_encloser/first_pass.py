import abc
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BaseFirstPass(abc.ABC):

    def _create_new_group(self, row, min_group_no):
        self.gxys.at[row[0], 'group_no'] = np.max(self.gxys.group_no) + 1
        self.gxys.at[row[0], 'group_peak'] = 1

        if self.plot == 'verbose':
            if (
                    (np.abs(row.xx) <= 2) &
                    (np.abs(row.yy) <= 2)) | (not cap):
                self.title = 'First Pass: %u Groups Found' % len(np.unique(self.gxys.group_no))
                self.plot_groups(
                    x1=row.xx,
                    y1=row.yy,
                    alpha_group=min_group_no)

    def run_first_pass(self, selection, min_group_no, cap=True):
        w = (self.gxys[selection].ff >= self.delta_outer)

        # For each galaxy...
        for row in self.gxys[selection][w].itertuples():
            # If no groups above min_group_no exist
            if not len(self.gxys[self.gxys.group_no > min_group_no]):
                self._create_new_group(row, min_group_no)

            else:
                ww = (
                    (self.gxys.group_no > min_group_no) &
                    (self.gxys.group_peak == 1))
                dist = np.sqrt((row.xx - self.gxys[ww].xx) ** 2 +
                               (row.yy - self.gxys[ww].yy) ** 2).values
                inds = np.argsort(dist)
                contour_group = np.zeros_like(dist)

                for row2 in (
                        self.gxys[ww].iloc[inds].reset_index(drop=True).itertuples()):
                    if row2[0] < 10:       # Only check nearest 10 groups
                        samples = int(max(
                            np.ceil(
                                (600 * dist[inds][row2[0]] * (dist[inds][row2[0]] + 1)** -2)),
                                2.0))
                        spacing = dist[inds][row2[0]] / samples

                        ff_mid = np.array([self.spline(
                            row.xx + (row2.xx - row.xx) * p,
                            row.yy + (row2.yy - row.yy) * p)[0][0] for p in np.linspace(0.0, 1.0, samples)]).round(5)

                        w = ff_mid <= np.round(row2.ff, 5)

                        assert np.sum(w) > 1, 'Not enough samples'

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
                                    (np.max(ff_mid) < 2 * row2.ff)) |
                                (
                                    (not cap) &
                                    (np.min(ff_mid_diff) >= self.mono / 10) &
                                    (np.min(ff_mid) >= self.delta_outer))):

                            contour_group[row2[0]] = 1
                            break

                if (np.max(contour_group) < 0) | (np.max(contour_group) > 1):
                    print('debug')

                # No connecting groups found
                if np.max(contour_group) == 0:
                    self._create_new_group(row, min_group_no)
                # One connecting group found
                elif np.max(contour_group) == 1:
                    www = contour_group == 1
                    self.gxys.at[row[0], 'group_no'] = (
                        self.gxys.group_no[ww].iloc[inds]
                        .values[np.min(np.where(www)[0])])

                if self.plot == 'verbose':
                    if (
                            (np.abs(row.xx) <= 2) &
                            (np.abs(row.yy) <= 2)) | (cap == False):
                        self.title = 'First Pass: %u Groups Found' % len(
                            np.unique(self.gxys.group_no))
                        self.plot_groups(
                            x1=row.xx,
                            y1=row.yy,
                            x2=row2.xx,
                            y2=row2.yy,
                            alpha_group=min_group_no)
