import numpy as np

def ttest(obesity1, ratio1, obesity2, ratio2, targetratio):
    population1 = int(obesity1/ratio1)
    population2 = int(obesity2/ratio2)
    se1 = ratio1 * (1-ratio1)
    se2 = ratio2 * (1-ratio2)
    return (ratio1 - ratio2 - targetratio) / np.sqrt(se1/population1 + se2/population2)

def degreefreedom(obesity1, ratio1, obesity2, ratio2):
    population1 = int(obesity1/ratio1)
    population2 = int(obesity2/ratio2)
    se1 = ratio1 * (1-ratio1)
    se2 = ratio2 * (1-ratio2)
    return ((se1/population1+se2/population2) ** 2) / (((se1/population1) ** 2) / (population1-1) + ((se2/population2) ** 2) / (population2-1))