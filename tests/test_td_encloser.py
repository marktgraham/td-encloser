import pandas
import numpy
import matplotlib.pyplot as plt

from scipy.interpolate import RectBivariateSpline

from td_encloser import TDENCLOSER, gaussian_kde


class TestTDENCLOSER:

    def _generate_kde_field(self, x, y, box_size):
        xmin, xmax, ymin, ymax = -box_size, box_size, -box_size, box_size

        grid_size = box_size * 15 + 1

        x = numpy.ravel(x)
        y = numpy.ravel(y)
        values = numpy.vstack([x, y])

        # Peform the kernel density estimate
        xx, yy = numpy.mgrid[xmin:xmax:grid_size * 1j, ymin:ymax:grid_size * 1j].round(5)

        positions = numpy.vstack([xx.ravel(), yy.ravel()])

        kernel = gaussian_kde(values, gauss_sig=0.3)

        density = kernel(positions)                   # Density at each pixel

        density = density.T                   # Density at each pixel
        f = numpy.reshape(density, xx.shape)

        return xx, yy, f

    def test_gaussian(self):
        """ Method to test the density field. """

        df = pandas.DataFrame({'x': [0, 0.3], 'y': [0, 0]})

        arr_x, arr_y, ff = self._generate_kde_field(df['x'], df['y'], 10)

        spline = RectBivariateSpline(numpy.unique(arr_x), numpy.unique(arr_y), ff)

        density_zero = spline(0, 0)[0][0]

        numpy.testing.assert_almost_equal(density_zero, 1.60653, decimal=5)

    def test_td_encloser(self):
        """ Method to test TD-ENCLOSER. """

        df = pandas.read_csv('../data/Processed/Mock_galaxy_catalogue_test.csv')

        arr_x, arr_y, ff = self._generate_kde_field(df['x'], df['y'], 50)

        spline = RectBivariateSpline(numpy.unique(arr_x), numpy.unique(arr_y), ff)

        td_encloser = TDENCLOSER(df['x'].values, df['y'].values, spline, plot=False, mono=-0.01, target=False)

        results = td_encloser.gxys

        results_select = results.loc[lambda x: ((x['group_no'] <= 4) & (x['group_no'] > 0))]

        group_members = results_select['group_no'].value_counts()

        assert group_members.loc[1] == 36
        assert group_members.loc[2] == 30
        assert group_members.loc[3] == 33
        assert group_members.loc[4] == 31

        group_peaks = (
            results_select
            .loc[lambda x: x['group_peak'] == 1]
            .reset_index(drop=True)
            .sort_values('group_no'))

        box_size = 3

        fig, axes = plt.subplots(figsize=(10, 10), nrows=2, ncols=2)

        for i, group in group_peaks.iterrows():
            ax = numpy.ravel(axes)[i]
            ax.set_xlim(group['xx'] - box_size, group['xx'] + box_size)
            ax.set_ylim(group['yy'] - box_size, group['yy'] + box_size)

            results_box = (
                results
                .loc[lambda x: abs(x['xx'] - group['xx']) <= 3]
                .loc[lambda x: abs(x['yy'] - group['yy']) <= 3])

            for i, group_ in results_box['group_no'].drop_duplicates().reset_index(drop=True).iteritems():
                for color, edgecolor, s in zip([None, f'C{i}'], ['k', None], [50, 40]):
                    ax.scatter(
                        results_box.loc[lambda x: (x['group_no'] == group_), 'xx'],
                        results_box.loc[lambda x: (x['group_no'] == group_), 'yy'],
                        color=color,
                        s=s)

                ax.set_title(f"{group_members.loc[group['group_no']]} members", fontsize=16)

            ax.contour(
                arr_x, arr_y, ff,
                colors=('0', '0.25', '0.75'),
                levels=numpy.array([1, 2.5, 4]) * 1.6,
                linewidths=(2, 1, 1),
                zorder=1)

        plt.tight_layout()
        plt.pause(20)
