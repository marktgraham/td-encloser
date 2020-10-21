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
        xx, yy = numpy.mgrid[
            xmin:xmax:grid_size * 1j,
            ymin:ymax:grid_size * 1j].round(5)

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

        spline = RectBivariateSpline(
            numpy.unique(arr_x), numpy.unique(arr_y), ff)

        density_zero = spline(0, 0)[0][0]

        numpy.testing.assert_almost_equal(density_zero, 1.60653, decimal=5)

    def test_td_encloser_on_simulated_catalogue(self):
        numpy.random.seed(42)
        num = 40
        rad = 0.4

        df = pandas.DataFrame({
            'x': numpy.hstack((
                numpy.random.uniform(-rad, rad, num) - 1.5,
                numpy.random.uniform(-rad, rad, int(num * 0.3)),
                numpy.random.uniform(-5, 5, 150))),
            'y': numpy.hstack((
                numpy.random.uniform(-rad, rad, num) + 0.2,
                numpy.random.uniform(-rad, rad, int(num * 0.3)),
                numpy.random.uniform(-5, 5, 150)))})

        arr_x, arr_y, ff = self._generate_kde_field(df['x'], df['y'], 10)

        spline = RectBivariateSpline(
            numpy.unique(arr_x), numpy.unique(arr_y), ff)

        td_encloser = TDENCLOSER(
            df['x'].values, df['y'].values, spline,

            plot=False, mono=-0.01, target=False)

        results = td_encloser.find_groups()

        self._check_group_members(
            results, group1=50, group2=14, group3=6, group4=5)

        self.plot_results(results, arr_x, arr_y, ff)

    def _test_td_encloser_on_mock_catalogue(self):
        """ Method to test TD-ENCLOSER. """

        df = pandas.read_csv('data/processed/Mock_galaxy_catalogue_test.csv')

        arr_x, arr_y, ff = self._generate_kde_field(df['x'], df['y'], 50)

        spline = RectBivariateSpline(
            numpy.unique(arr_x), numpy.unique(arr_y), ff)

        td_encloser = TDENCLOSER(
            df['x'].values, df['y'].values, spline,
            plot=False, mono=-0.01, target=False)

        results = td_encloser.find_groups()

        self._check_group_members(
            results=results, group1=36, group2=30, group3=33, group4=31)

        self.plot_results(results)

    def _check_group_members(self, results, **kwargs):

        for group in kwargs:
            group_no = int(group[5:])
            group_mem_test = kwargs.get(group)
            try:
                assert results.loc[
                    lambda x: x['group_no'] == group_no, 'group_mem'].iloc[0] \
                        == group_mem_test
            except AssertionError:
                print(results.loc[lambda x: (x['group_peak'] == 1) & (x['group_no'] == group_no)])

    def plot_results(self, df, arr_x, arr_y, ff):

        group_members = df['group_no'].value_counts()

        df_top_4 = df.loc[lambda x: ((x['group_no'] <= 4) & (x['group_no'] > 0))]

        group_peaks = (
            df
            .loc[lambda x: x['group_peak'] == 1]
            .reset_index(drop=True)
            .sort_values('group_no'))

        box_size = 3

        fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=2)

        for i, group in group_peaks[:4].iterrows():
            ax = numpy.ravel(axes)[i]
            ax.set_xlim(group['x'] - box_size, group['x'] + box_size)
            ax.set_ylim(group['y'] - box_size, group['y'] + box_size)

            df_box = (
                df
                .loc[lambda x: abs(x['x'] - group['x']) <= 3]
                .loc[lambda x: abs(x['y'] - group['y']) <= 3])

            for i, group_ in df_box['group_no'].drop_duplicates().reset_index(drop=True).iteritems():
                for color, edgecolor, s in zip([None, f'C{i}'], ['k', None], [50, 40]):
                    ax.scatter(
                        df_box.loc[lambda x: (x['group_no'] == group_), 'x'],
                        df_box.loc[lambda x: (x['group_no'] == group_), 'y'],
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
