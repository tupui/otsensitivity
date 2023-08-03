# Copyright (C) Michaël Baudin (2023)
# -*- coding: utf-8 -*-
"""
Let Y=g(X) be the scalar output of 
the model g with vector input X with dimension nx. 
Let a < b be two real numbers. 
We consider the event {a < Y < b}. 
We want to compute the sensitivity of that event with respect to each input Xi. 

This script computes the conditional distribution of the input Xi given that the output Y is
in the interval [a, b], for i=1,...,nx. 
Compare that conditional distribution with 
the unconditional distribution of Xi: 
if there is no difference, then the input Xi is not influential for that event.
"""

import openturns as ot
import openturns.viewer as otv
from matplotlib import pylab as plt
import numpy as np


# %%
def computeConditionnedSample(
    sample,
    lowerBound,
    upperBound,
    columnIndex,
):
    """
    Return points from the -th component of the sample.
    Selects the values according to the alpha-level quantile of
    the criteriaComponent-th component of the sample.
    """
    sample = ot.Sample(sample)  # Copy the object
    sortedSampleCriteria = sample[:, columnIndex]
    condition = (np.array(sortedSampleCriteria.asPoint()) >= lowerBound) & (
        np.array(sortedSampleCriteria.asPoint()) < upperBound
    )
    indices = np.where(condition)[0]
    rowIndices = [int(j) for j in indices]
    conditionnedSortedSample = sample[rowIndices]
    return conditionnedSortedSample


def joint_input_output_sample(inputSample, outputSample):
    inputDimension = inputSample.getDimension()
    sampleSize = inputSample.getSize()
    outputDimension = outputSample.getDimension()
    # Joint the X and Y samples into a single one, so that the
    # sort can be done simultaneously on inputs and outputs
    jointXYSample = ot.Sample(sampleSize, inputDimension + outputDimension)
    jointXYSample[:, :inputDimension] = inputSample
    jointXYSample[:, inputDimension : inputDimension + outputDimension] = outputSample
    jointDescription = ot.Description(inputDimension + outputDimension)
    inputDescription = inputSample.getDescription()
    jointDescription[:inputDimension] = inputDescription
    jointDescription[
        inputDimension : inputDimension + outputDimension
    ] = outputSample.getDescription()
    jointXYSample.setDescription(jointDescription)
    return jointXYSample


def filter_sample(inputSample, outputSample, indexOutput, lowerValue, upperValue):
    """
    Lit les données et filtre sur la valeur de la variable de sortie.

    Parameters
    ----------
    filename : TYPE
        Le nom du fichier.
    quantile_level : float, in [0, 1]
        The quantile level of the output.
    fast : TYPE, optional
        Si vrai, utilise un sous-échantillon. The default is False.
    verbose : TYPE, optional
        Si vrai, affiche des messages intermédiaires. The default is False.

    Returns
    -------
    sample : ot.Sample
        L'échantillon.
    quantile_value : TYPE
        DESCRIPTION.

    """
    # 1. Joint X and Y samples
    jointXYSample = joint_input_output_sample(inputSample, outputSample)
    # 3. Condition
    inputDimension = inputSample.getDimension()
    jointXYIndex = inputDimension + indexOutput
    print("jointXYIndex = ", jointXYIndex)
    conditionedXYSample = computeConditionnedSample(
        jointXYSample,
        lowerValue,
        upperValue,
        jointXYIndex,
    )
    # 4. Split into X and Y
    outputDimension = outputSample.getDimension()
    conditionedInputSample = conditionedXYSample[:, 0:inputDimension]
    conditionedOutputSample = conditionedXYSample[
        :, inputDimension : inputDimension + outputDimension
    ]

    return conditionedInputSample, conditionedOutputSample


def plot_event(
    inputSample,
    outputSample,
    indexOutput,
    lowerValue,
    upperValue,
    inputDistribution,
    verbose=False,
):
    dimension_input = inputSample.getDimension()
    sample_size = inputSample.getSize()
    if verbose:
        print("Sample size = ", sample_size)
        print("+ Min et max")
        print("    Input maximum : ", inputSample.getMax())
        print("    Input minimum : ", inputSample.getMin())
        print("    Output maximum : ", outputSample.getMax())
        print("    Output minimum : ", outputSample.getMin())
        print("lower bound = ", lowerValue)
        print("upper bound = ", upperValue)
        print(inputSample[:10])
        print(outputSample[:10])

    conditionedInputSample, conditionedOutputSample = filter_sample(
        inputSample, outputSample, indexOutput, lowerValue, upperValue
    )
    conditionedSampleSize = conditionedInputSample.getSize()

    if verbose:
        print("Conditioned Sample size = ", conditionedSampleSize)
        print("+ Min et max")
        print("    Input maximum : ", conditionedInputSample.getMax())
        print("    Input minimum : ", conditionedInputSample.getMin())
        print("    Output maximum : ", conditionedOutputSample.getMax())
        print("    Output minimum : ", conditionedOutputSample.getMin())
        print(conditionedInputSample[:10])
        print(conditionedOutputSample[:10])

    if verbose:
        print("+ Visualise la distribution des entrées")
    number_of_graphs = 2
    inputDescription = inputSample.getDescription()
    grid = ot.GridLayout(1, dimension_input)
    for i in range(dimension_input):
        # Plot conditional distribution
        marginal_input_distribution = ot.KernelSmoothing().build(
            conditionedInputSample[:, i]
        )
        graph = marginal_input_distribution.drawPDF()
        graph.setLegends([""])
        graph.setXTitle(inputDescription[i])
        if i > 0:
            graph.setYTitle("")
        # Plot unconditional distribution
        marginal_distribution = inputDistribution.getMarginal(i)
        curve = marginal_distribution.drawPDF()
        curve.setLegends([""])
        graph.add(curve)
        graph.setColors(ot.Drawable().BuildDefaultPalette(number_of_graphs))
        grid.setGraph(0, i, graph)
    grid.setTitle(
        f"Unconditioned n={sample_size}, " f"conditioned n = {conditionedSampleSize}"
    )
    return grid
