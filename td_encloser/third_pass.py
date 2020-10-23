import abc


class BaseThirdPass(abc.ABC):
    """ Class holding methods to run the third pass. """

    def run_third_pass(self, selection, min_group_no):
        raise NotImplementedError()
