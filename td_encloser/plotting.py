import abc
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BasePlotting(abc.ABC):
    """ Class holding methods to produce pretty plots. """

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
        """ Method to visualise the groups. """
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
            -self.df_gxys.loc[self.gxys.group_peak == 1, 'x'],
            self.df_gxys.loc[self.gxys.group_peak == 1, 'y'],
            edgecolor='k',
            facecolor='none',
            linewidth=2,
            s=32,
            zorder=5)

        inds = np.argsort(
            np.sqrt(
                self.df_gxys.loc[self.gxys.group_peak == 1, 'x'] ** 2 +
                self.df_gxys.loc[self.gxys.group_peak == 1, 'y'] ** 2))

        marker = 'o'

        alpha = np.ones_like(self.df_gxys['x'])
        alpha[
            (self.df_gxys['group_no'] > 1) &
            (self.df_gxys['group_no'] < alpha_group)] = 0.25

        for group_no in [0, 1]:
            plt.scatter(
                -self.df_gxys.loc[lambda x: x['group_no'] == group_no, 'x'],
                self.df_gxys.loc[lambda x: x['group_no'] == group_no, 'y'],
                c=f'C{group_no}',
                s=30,
                zorder=2,
                marker=marker,
                alpha=alpha[self.df_gxys['group_no'] == group_no][0])

        marker_ = np.tile(np.array(['o', 's', 'D', '^', 'x']), 2000)

        for i, group_no in enumerate(
                self.df_gxys.loc[lambda x: x['group_no'] > 1, 'group_no']):
            group = self.df_gxys['group_no'] == group_no

            color = f'C{(i % 7) + 2}'
            marker = marker_[np.floor((i + 2) / 10).astype(int)]

            plt.scatter(
                -self.df_gxys.loc[group, 'x'],
                self.df_gxys.loc[group, 'y'],
                c=color,
                s=30,
                zorder=2,
                marker=marker,
                label=f'Group {group_no}: {group.sum()}',
                alpha=alpha[group][0])

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

        median = np.argsort(self.df_gxys['x'])[len(self.df_gxys['x']) // 2]

        if center == (0, 0):
            if not self.target:
                plt.xlim(
                    self.df_gxys['x'][median] - lim,
                    self.df_gxys['x'][median] + lim)
                plt.ylim(
                    self.df_gxys['y'][median] - lim,
                    self.df_gxys['y'][median] + lim)

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
