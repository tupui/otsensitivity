import pytest
import numpy as np
import openturns as ot


@pytest.fixture(scope='session')
def ishigami():
    # Create the model and input distribution
    formula = ['sin(X1) + 7 * sin(X2)^2 + 0.1 * X3^4 * sin(X1)']
    input_names = ['X1', 'X2', 'X3']
    dimension = 3
    corners = [[-np.pi] * dimension, [np.pi] * dimension]
    model = ot.SymbolicFunction(input_names, formula)
    distribution = ot.ComposedDistribution([ot.Uniform(corners[0][i], corners[1][i])
                                            for i in range(dimension)])

    # Create X/Y data
    ot.RandomGenerator.SetSeed(0)
    size = 1000
    sample = ot.LowDiscrepancyExperiment(ot.SobolSequence(),
                                         distribution, size).generate()
    data = model(sample)

    return model, sample, data
