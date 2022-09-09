import numpy as np
import c_utils


def uniform_fuck(chromosone_A, chromosone_B):
    if len(chromosone_A) != len(chromosone_B):
        return ValueError(
            "you wouldn't ask a great dane to fuck a chihuaua, so why are you trying to mate different length chromosones?!"
        )
    child_chromosone = c_utils.uniform_fuck(chromosone_A, chromosone_B)
    return child_chromosone
