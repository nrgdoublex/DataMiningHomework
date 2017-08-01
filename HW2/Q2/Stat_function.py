import numpy as np

def t_test(number1, ratio1, number2, ratio2, targetratio):
    population1 = int(number1/ratio1)
    population2 = int(number2/ratio2)
    se1 = ratio1 * (1-ratio1)
    se2 = ratio2 * (1-ratio2)
    return (ratio1 - ratio2 - targetratio) / np.sqrt(se1/population1 + se2/population2)

def degree_freedom(obesity1, ratio1, obesity2, ratio2):
    population1 = int(obesity1/ratio1)
    population2 = int(obesity2/ratio2)
    se1 = ratio1 * (1-ratio1)
    se2 = ratio2 * (1-ratio2)
    return ((se1/population1+se2/population2) ** 2) / (((se1/population1) ** 2) / (population1-1) + ((se2/population2) ** 2) / (population2-1))


if __name__ == '__main__':
    print t_test(10000, 0.1, 1000, 0.12, 0)
    print degree_freedom(10000, 0.1, 1000, 0.12)