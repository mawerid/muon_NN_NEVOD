from generate.sct import Sct
from generate.decor import Decor
from generate.sphere import Sphere
from generate.utils import visualise_data, sphere_uniform
import numpy as np

if __name__ == "__main__":
    # sphere_uniform(1000000, (0, 180), (0, 360))

    print("Generating data...")
    gen = Decor(16000, 2)
    start, end = gen.generate()
    gen.save()
    print("DECOR data generated.")

    gen = Sct(160000, 1)
    start, end = gen.generate()
    gen.save()
    print("SCT data generated.")

    gen = Sphere(324000, 3, theta_range=(0, 100 / 180 * np.pi))
    start, end = gen.generate()
    gen.save()
    print("Sphere data generated.")

    # gen = Generator(16000, 2)
    # start, end = gen.load_data("../data_sim/OTDCR_1.txt")

    # plt.hist(np.r_[start, end], bins=100)
    # plt.show()

    # visualise_data(np.r_[start, end])

    # visualise_data(
    #     sphere_uniform(1000, (0, 360), (0, 180)))
    # cube_uniform(1000, (0, 360), (0, 180), (0, 90)))
