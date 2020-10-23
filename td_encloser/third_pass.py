import abc
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BaseThirdPass(abc.ABC):

    def run_third_pass(self, selection, min_group_no):
        raise NotImplementedError()
