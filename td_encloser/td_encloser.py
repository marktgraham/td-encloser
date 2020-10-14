import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class TDENCLOSER(object):

    def __init__(
            self,
            xx,
            yy,
            spline,
            plot=False,
            n_merge=4,
            delta_outer=1.6,
            delta_saddle=4.0,
            delta_peak=4.8,
            file='group_finder',
            target=True,
            mono=0):

        assert plot in [False, 'quick', 'verbose'], 'plot is False, quick or verbose'

        '''
        Routine to find groups using the hopping method.
        We want to know which galaxies are in the same group as the target galaxy.
        group_no = 1 if galaxy belongs to the same group as the target galaxy

        :param plot: If true, plots what's going on
        :return:
        '''

        start = time.time()

        # rotate galaxy coordinates for interpolation....
        xx_, yy_ = yy.copy(), xx.copy()
        # Calculate density at each galaxy
        ff = np.array([spline(y, x) for (y, x) in zip(yy_, xx_)])[:, 0][:, 0]

        # Sort by density in descending order
        inds = np.argsort(ff)[::-1]
        inds_undo = np.argsort(inds)
        self.gxys = pd.DataFrame({
            'xx': xx[inds],
            'yy': yy[inds],
            'ff': ff[inds],
            'group_no': 0,
            'group_peak': 0})
        self.ind_0 = np.where(inds == 0)[0][0]       # indice of the target galaxy

        xxx = np.linspace(
            np.round(np.min(self.gxys['xx'])).astype(int),
            np.round(np.max(self.gxys['xx'])).astype(int),
            501)
        yyy = np.linspace(
            np.round(np.min(self.gxys['yy'])).astype(int),
            np.round(np.max(self.gxys['yy'])).astype(int),
            501)
        fff = spline(yyy, xxx)
        xxx, yyy = np.meshgrid(xxx, yyy)
        xxx, yyy = yyy.copy(), xxx.copy()         # rotate grid for contours

        self.contours = pd.DataFrame({
            'x': xxx.ravel(),
            'y': yyy.ravel(),
            'f': fff.ravel()})

        self.spline = spline
        self.plot = plot
        self.delta_outer = delta_outer
        self.delta_saddle = delta_saddle
        self.delta_peak = delta_peak
        self.levels = np.array([delta_outer, delta_saddle])
        self.n_merge = int(n_merge)
        self.target = target

        self.pause = True if plot == 'verbose' else False
        # mono = 0
        self.mono = mono
        self.array = np.zeros(1)

        if plot:
            plt.figure(figsize=(12, 4))
            plt.subplot(131)
            plt.ylabel('y (Mpc)')

        # If target galaxy is above delta_outer...
        if ((
                self.gxys.ff.iloc[self.ind_0] >= self.delta_outer)
                & target) | (not target):
            self.gxys.at[0, 'group_no'], self.gxys.at[0, 'group_peak'] = 1, 1
            print('Starting first pass...')
            w = self.gxys.index.values > 0
            self.first_pass(selection=w, min_group_no=0, cap=True)
            print('Completed in %.1f seconds' % (time.time() - start))

        else:
            self.gxys.at[self.ind_0, 'group_no'] = 1
            self.grps = pd.DataFrame(
                {
                    'x': 0,
                    'y': 0,
                    'f': self.gxys.ff.iloc[self.ind_0],
                    'group_no': 1},
                index=[0])

        if (self.gxys.group_no.iloc[self.ind_0] > 1) & (target):
            w = self.gxys.group_no == self.gxys.group_no.iloc[self.ind_0]
            ww = self.gxys.group_no == 1
            self.gxys.at[ww, 'group_no'] = self.gxys.group_no.iloc[self.ind_0]
            self.gxys.at[w, 'group_no'] = 1

        if plot:
            self.title = 'First Pass'
            self.plot_groups(legend=False)

            plt.tight_layout()
            if self.plot == 'verbose':
                if input('Continue? ') != '~':
                    pass
            plt.subplot(132)

        if target:
            if self.gxys.group_no.iloc[self.ind_0] == 1:
                print('Target galaxy is in a group')
            elif self.gxys.group_no.iloc[self.ind_0] == 0:
                print('Target galaxy is isolated')

        # Second pass: Break up Group 1

        if np.sum(self.gxys.group_no == 1) != 1:
            # For groups where the peak is greater than delta_saddle,
            # select galaxies below delta_saddle (and above delta outer)
            self.second_pass()

        if self.plot:
            if self.plot == 'verbose':
                if input('Continue? ') != '~':
                    pass
            plt.subplot(133)

        # Select galaxies between delta_outer and delta_saddle
        w_saddle = (self.gxys.group_no == 0) & (self.gxys.ff >= delta_outer)

        if np.sum(w_saddle):  # If such galaxies exist
            print('Attempting to form new groups from %u remaining galaxies...' % np.sum(w_saddle))

            max_group_no = np.max(self.gxys.group_no)
            print('Starting third pass...')
            self.title = 'Third Pass: Forming New Groups with Remaining Galaxies...'
            self.first_pass(
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

            plt.savefig(file + '.pdf')

        print('Found %u groups in %.1f seconds' % (
            len(np.unique(self.gxys.group_no)), time.time()-start))

        self.gxys = self.gxys.iloc[inds_undo].reset_index(drop=True)

    def first_pass(self, selection, min_group_no, cap=True):
        w = (self.gxys[selection].ff >= self.delta_outer)

        # For each galaxy...
        for row in self.gxys[selection][w].itertuples():
            # If no groups above min_group_no exist
            if not len(self.gxys[self.gxys.group_no > min_group_no]):
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

            else:
                ww = (
                    (self.gxys.group_no > min_group_no) &
                    (self.gxys.group_peak == 1))
                dist = np.sqrt((row.xx - self.gxys[ww].xx) ** 2 +
                               (row.yy - self.gxys[ww].yy) ** 2).values
                inds = np.argsort(dist)
                contour_group = np.zeros_like(dist)

                for row2 in (
                        self
                        .gxys[ww]
                        .iloc[inds]
                        .reset_index(drop=True)
                        .itertuples()):
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
                    self.gxys.at[row[0], 'group_no'] = np.max(self.gxys.group_no) + 1
                    self.gxys.at[row[0], 'group_peak'] = 1

                # One connecting group found
                elif np.max(contour_group) == 1:
                    www = contour_group == 1
                    self.gxys.at[row[0], 'group_no'] = (
                        self
                        .gxys
                        .group_no[ww]
                        .iloc[inds]
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

    def second_pass(self):
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

    def plot_groups(
            self,
            lim=4,
            center=(0, 0),
            x1='',
            y1='',
            x2='',
            y2='',
            linecolor='k',
            alpha_group=1,
            legend=False,
            pause=False):
        ax = plt.gca()
        ax.clear()

        shape = (
            np.sqrt(len(self.contours)).astype(int),
            np.sqrt(len(self.contours)).astype(int))

        plt.contour(
            -self.contours.x.values.reshape(shape),
            self.contours.y.values.reshape(shape),
            self.contours.f.values.reshape(shape),
            colors='grey',
            levels=np.arange(2, int(np.max(self.contours.f) + 1), 1),
            linewidths=1,
            zorder=1)

        plt.contour(
            -self.contours.x.values.reshape(shape),
            self.contours.y.values.reshape(shape),
            self.contours.f.values.reshape(shape),
            colors='k',
            levels=self.levels,
            linewidths=2,
            zorder=1)

        if self.target:
            plt.plot([-lim * 0.05, -lim * 0.025], [0, 0], color='k')
            plt.plot([lim * 0.05, lim * 0.025], [0, 0], color='k')
            plt.plot([0, 0], [-lim * 0.05, -lim * 0.025], color='k')
            plt.plot([0, 0], [lim * 0.05, lim * 0.025], color='k')

        plt.scatter(
            -self.gxys['xx'][self.gxys.group_peak == 1],
            self.gxys['yy'][self.gxys.group_peak == 1],
            edgecolor='k',
            facecolor='none',
            linewidth=2,
            s=32,
            zorder=5)

        inds = np.argsort(
            np.sqrt(
                self.gxys['xx'][self.gxys.group_peak == 1] ** 2 +
                self.gxys['yy'][self.gxys.group_peak == 1] ** 2))

        marker = 'o'

        alpha = np.ones_like(self.gxys['xx'])
        alpha[
            (self.gxys.group_no > 1) &
            (self.gxys.group_no < alpha_group)] = 0.25

        plt.scatter(
            -self.gxys['xx'][self.gxys.group_no == 0],
            self.gxys['yy'][self.gxys.group_no == 0],
            c='C0',
            s=30,
            zorder=2,
            marker=marker,
            alpha=alpha[self.gxys.group_no == 0][0])
        plt.scatter(
            -self.gxys['xx'][self.gxys.group_no == 1],
            self.gxys['yy'][self.gxys.group_no == 1],
            c='C1',
            s=30,
            zorder=2,
            marker=marker,
            alpha=alpha[self.gxys.group_no == 1][0])

        marker_ = np.tile(np.array(['o', 's', 'D', '^', 'x']), 2000)

        for i, n in enumerate(
                np.unique(
                    self.gxys.group_no[self.gxys.group_no > 1])):
            w = self.gxys.group_no == n

            color = 'C%u' % ((i % 7) + 2)
            marker = marker_[np.floor((i + 2) / 10).astype(int)]

            plt.scatter(
                -self.gxys['xx'][w],
                self.gxys['yy'][w],
                c=color,
                s=30,
                zorder=2,
                marker=marker,
                label='Group %u: %u' % (n, np.sum(w)),
                alpha=alpha[w][0])

        if (x1 != '') & (y1 != '') & (x2 != '') & (y2 != ''):
            plt.plot(
                [-x1, -x2],
                [y1, y2],
                linestyle='--',
                color=linecolor,
                zorder=3)
        if (x1 != '') & (y1 != ''):
            plt.scatter(
                -x1,
                y1,
                marker='o',
                edgecolor='r',
                facecolor='none',
                zorder=4,
                s=80)
        if (x2 != '') & (y2 != ''):
            plt.scatter(-x2, y2, marker='x', color='r', zorder=4, s=80)

        plt.title(self.title, zorder=6)

        median = np.argsort(self.gxys['xx'])[len(self.gxys['xx']) // 2]

        if center == (0, 0):
            if not self.target:
                plt.xlim(
                    self.gxys['xx'][median] - lim,
                    self.gxys['xx'][median] + lim)
                plt.ylim(
                    self.gxys['yy'][median] - lim,
                    self.gxys['yy'][median] + lim)

            else:
                plt.xlim(-lim, lim)
                plt.ylim(-lim, lim)

        else:
            plt.xlim(center[0] - lim, center[0] + lim)
            plt.ylim(center[1] - lim, center[1] + lim)

        plt.gca().set_aspect('equal', 'box')

        plt.xlabel('x (Mpc)')

        if legend:
            plt.legend(loc='lower right', ncol=4)

        if self.pause:
            plt.pause(0.001)
