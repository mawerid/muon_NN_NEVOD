from generate.sct import Sct
from generate.decor import Decor
from generate.sphere import Sphere
from generate.utils import visualise_data
import numpy as np

if __name__ == "__main__":
    gen = Sphere(1600, 3, theta_range=(0, 100 / 180 * np.pi))
    start, end = gen.generate()
    gen.save()
    # start, end = gen.load_data("../data_sim/OTDCR_1.txt")

    # plt.hist(np.r_[start, end], bins=100)
    # plt.show()

    # visualise_data(np.r_[start, end])

    # visualise_data(
    #     sphere_uniform(1000, (0, 360), (0, 180)))
    # cube_uniform(1000, (0, 360), (0, 180), (0, 90)))
