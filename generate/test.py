from generate.sct import Sct
from generate.decor import Decor
from generate.utils import visualise_data
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    gen = Decor(160, 3)
    start, end = gen.generate()
    # gen.save()
    # start, end = gen.load_data("../data_sim/OTDCR_1.txt")

    # plt.hist(np.r_[start, end], bins=100)
    # plt.show()

    # visualise_data(np.r_[start, end])

    # visualise_data(
    #     sphere_uniform(1000, (0, 360), (0, 180)))
    # cube_uniform(1000, (0, 360), (0, 180), (0, 90)))
