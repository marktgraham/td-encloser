import abc
import pandas
import numpy
import matplotlib.pyplot as plt

from scipy.interpolate import RectBivariateSpline

from td_encloser import TDENCLOSER, gaussian_kde


class BaseTestTDENCLOSER(abc.ABC):

    @abc.abstractmethod
    def setup(self):
        raise NotImplementedError()

    @staticmethod
    def _generate_kde_field(x, y, box_size):
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

    def _check_group_members(self, results, **kwargs):

        for group in kwargs:
            group_no = int(group[5:])
            group_mem_test = kwargs.get(group)
            try:
                assert results.loc[
                    lambda x: x['group_no'] == group_no, 'group_mem'].iloc[0] \
                        == group_mem_test
            except AssertionError:
                print(
                    results.loc[lambda x: (
                        (x['group_peak'] == 1) &
                        (x['group_no'] == group_no))])

    def _plot_results(self, df, arr_x, arr_y, ff):

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

            for i, group_ in (
                    df_box['group_no']
                    .drop_duplicates()
                    .reset_index(drop=True)
                    .iteritems()):

                kwargs_plot = [
                    {'color': None, 'edgecolor': 'k', 's': 70},
                    {'color': f'C{i}', 'edgecolor': None, 's': 40}]

                for kwargs in kwargs_plot:
                    ax.scatter(
                        df_box.loc[lambda x: (x['group_no'] == group_), 'x'],
                        df_box.loc[lambda x: (x['group_no'] == group_), 'y'],
                        **kwargs)

                num_mems = group_peaks.loc[
                    lambda x: x['group_no'] == group['group_no'],
                    'group_mem'].item()

                ax.set_title(f"{num_mems} members", fontsize=16)

            ax.contour(
                arr_x, arr_y, ff,
                colors=('0', '0.25', '0.75'),
                levels=numpy.array([1, 2.5, 4]) * 1.6,
                linewidths=(2, 1, 1),
                zorder=1)

        plt.tight_layout()
        plt.pause(20)

    def test_td_encloser(self, **kwargs):

        arr_x, arr_y, ff = self._generate_kde_field(
            self.df['x'], self.df['y'], self.box_size)

        spline = RectBivariateSpline(
            numpy.unique(arr_x), numpy.unique(arr_y), ff)

        td_encloser = TDENCLOSER(
            self.df['x'].values, self.df['y'].values, spline,
            plot=False, mono=-0.01, target=False)

        results = td_encloser.find_groups()

        self._check_group_members(
            results, **kwargs)

        self._plot_results(results, arr_x, arr_y, ff)


class TestGaussian():
    """ Class to test the Gaussian KDE. """

    def setup(self):

        self.df = pandas.DataFrame({'x': [0, 0.3], 'y': [0, 0]})

    def _generate_kde_field(self, **kwargs):
        return BaseTestTDENCLOSER._generate_kde_field(**kwargs)

    def test_gaussian(self):
        """ Method to test the density field. """

        arr_x, arr_y, ff = self._generate_kde_field(
            x=self.df['x'], y=self.df['y'], box_size=10)

        spline = RectBivariateSpline(
            numpy.unique(arr_x), numpy.unique(arr_y), ff)

        density_zero = spline(0, 0)[0][0]

        numpy.testing.assert_almost_equal(density_zero, 1.60653, decimal=5)


class TestOnSimulatedCatalogue(BaseTestTDENCLOSER):
    """ Class to test TD-ENCLOSER on a simulated catalogue. """

    def setup(self):

        numpy.random.seed(42)
        num = 40
        rad = 0.4

        self.df = pandas.DataFrame({
            'x': numpy.hstack((
                numpy.random.uniform(-rad, rad, num) - 1.5,
                numpy.random.uniform(-rad, rad, int(num * 0.3)),
                numpy.random.uniform(-5, 5, 150))),
            'y': numpy.hstack((
                numpy.random.uniform(-rad, rad, num) + 0.2,
                numpy.random.uniform(-rad, rad, int(num * 0.3)),
                numpy.random.uniform(-5, 5, 150)))})

        # box_size radius in Mpc
        self.box_size = 10

    def test_td_encloser(self):
        super().test_td_encloser(
            group1=50, group2=14, group3=6, group4=5)


class TestOnMockCatalogue(BaseTestTDENCLOSER):
    """ Class to test TD-ENCLOSER on a mock catalogue. """

    def setup(self):

        self.df = pandas.read_csv(
            'data/processed/Mock_galaxy_catalogue_test.csv')

        self.box_size = 50

    def test_td_encloser(self):
        super().test_td_encloser(
            group1=36, group2=33, group3=31, group4=30)
