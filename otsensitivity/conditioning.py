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

References
----------
- Estimating Global Sensitivity Measures: Torturing the Data Until They Confess.
  Elmar Plischke. Institut für Endlagerforschung. TU Clausthal
  St. Étienne, MASCOT-NUM, April 10, 2015.
"""

import openturns as ot
import numpy as np


def filterSample(
    sample,
    lowerBound,
    upperBound,
    columnIndex,
):
    """
    Filter out the rows in the sample based on bounds on the output.

    Return a sample with a reduced sample size.
    Each row of the filtered sample is such that the specified column
    is in a given interval defined by its bounds:

    a <= yj < b

    where yj is the columnIndex-th column of the sample,
    a is the lower bound and b is the upper bound.

    Parameters
    ----------
    sample: ot.Sample(size, dimension)
        The sample.
    lowerBound: float
        The lower bound.
    upperBound: float
        The upper bound.
    columnIndex: int
        The index of a column of the sample.
        Must be if the range 0, ..., dimension.

    Return
    ------
    conditionedSample: ot.Sample(conditionedSize, dimension
        The filtered sample.
    """
    dimension = sample.getDimension()
    if columnIndex < 0:
        raise ValueError(f"Negative column index {columnIndex}")
    if columnIndex >= dimension:
        raise ValueError(
            f"Column index {columnIndex} larger than dimension {dimension}."
        )
    if upperBound < lowerBound:
        raise ValueError(
            f"The lower bound {lowerBound} is greater "
            f"than the upper bound {upperBound}."
        )
    sample = ot.Sample(sample)  # Copy the object
    selectionSample = sample[:, columnIndex]
    selectionArray = np.array(selectionSample.asPoint())
    condition = (selectionArray >= lowerBound) & (selectionArray < upperBound)
    indices = np.where(condition)[0]
    rowIndices = [int(j) for j in indices]
    conditionnedSample = sample[rowIndices]
    return conditionnedSample


def joinInputOutputSample(inputSample, outputSample):
    """
    Make a single sample from an (X, Y) pair.

    Parameters
    ----------
    inputSample: ot.Sample(size, inputDimension)
        The input sample X.
    outputSample: ot.Sample(size, outputDimension)
        The output sample Y.

    Return
    ------
    jointXYSample: ot.Sample(size, dimension)
        The joint (X, Y) sample with dimension equal to inputDimension + outputDimension.
    """
    sampleSize = inputSample.getSize()
    if outputSample.getSize() != sampleSize:
        raise ValueError(
            f"The size of the input sample is {sampleSize} which "
            f"does not match the size of the output sample {outputSample.getSize()}."
        )
    jointXYSample = ot.Sample(inputSample)  # Make a copy
    jointXYSample.stack(outputSample)
    return jointXYSample


def filterInputOutputSample(
    inputSample, outputSample, outputIndex, lowerBound, upperBound
):
    """
    Filter out the rows in the input and output sample given bounds on the output.

    Return a pair of (input, output) sample with a reduced sample size.
    Each row of the filtered sample is such that the specified column
    of the output is in a given interval defined by its bounds:

    a <= yj < b

    where yj is the outputIndex-th column of the output sample,
    a is the lower bound and b is the upper bound.

    Parameters
    ----------
    inputSample: ot.Sample(size, inputDimension)
        The input sample X.
    outputSample: ot.Sample(size, outputDimension)
        The output sample Y.
    outputIndex : int
        The index of a column in the output sample.
        Must be in the set {0, ..., outputDimension - 1}.
    lowerBound : float
        The lower bound for filtering.
    upperBound : float
        The upper bound for filtering.

    Returns
    -------
    conditionedInputSample : ot.Sample(filteredSize, inputDimension)
        The filtered output sample.
    conditionedOutputSample: ot.Sample(filteredSize, outputDimension)
        The filtered output sample.

    """
    if upperBound < lowerBound:
        raise ValueError(
            f"The lower bound {lowerBound} is greater "
            f"than the upper bound {upperBound}."
        )
    sampleSize = inputSample.getSize()
    if outputSample.getSize() != sampleSize:
        raise ValueError(
            f"The size of the input sample is {sampleSize} which "
            f"does not match the size of the output sample {outputSample.getSize()}."
        )
    if outputIndex < 0 or outputIndex > outputSample.getDimension():
        raise ValueError(
            f"The output index {outputIndex} is not consistent with the "
            f"output dimension {outputSample.getDimension()}"
        )
    # 1. Join X and Y samples
    jointXYSample = joinInputOutputSample(inputSample, outputSample)
    # 2. Filter
    inputDimension = inputSample.getDimension()
    jointXYIndex = inputDimension + outputIndex
    conditionedXYSample = filterSample(
        jointXYSample,
        lowerBound,
        upperBound,
        jointXYIndex,
    )
    # 3. Split into X and Y
    outputDimension = outputSample.getDimension()
    conditionedInputSample = conditionedXYSample[:, 0:inputDimension]
    conditionedOutputSample = conditionedXYSample[
        :, inputDimension : inputDimension + outputDimension
    ]

    return conditionedInputSample, conditionedOutputSample


def plotConditionOutputBounds(
    inputSample,
    outputSample,
    outputIndex,
    lowerBound,
    upperBound,
    inputDistribution,
):
    """
    Plot the sensitivity of the output with respect to the input.

    Let Y=g(X) be the scalar output of
    the model g with vector input X with dimension nx.
    Let a < b be two real numbers.
    We consider the event {a <= Y < b}.
    We want to compute the sensitivity of that event with respect to each input Xi.

    This script computes the conditional distribution of the input Xi given that the output Y is
    in the interval [a, b], for i=1,...,nx.
    Compare that conditional distribution with
    the unconditional distribution of Xi:
    if there is no difference, then the input Xi is not influential for that event.

    Parameters
    ----------
    inputSample: ot.Sample(size, inputDimension)
        The input sample X.
    outputSample: ot.Sample(size, outputDimension)
        The output sample Y.
    outputIndex : int
        The index of a column in the output sample.
        Must be in the set {0, ..., outputDimension - 1}.
    lowerBound : float
        The lower bound for filtering.
    upperBound : float
        The upper bound for filtering.
    inputDistribution : ot.Distribution(inputDimension)
        The distribution of the input sample.

    Return
    ------
    grid: ot.GridLayout(1, 1 + inputDimension)
        The grid of sensitivity plots.
        The i-th plot presents the unconditional and conditional distribution of the
        i-th input to the outputIndex-th output.
        The last plot presents the unconditional and conditional distribution of the
        outputIndex-th output.
    """

    def plot_unconditional_and_conditional_distribution(
        unconditionalDistribution,
        conditionalDistribution,
        xTitle,
        marginalOutputDescription,
        lowerBound,
        upperBound,
    ):
        graph = ot.Graph("", xTitle, "PDF", True)
        # Plot unconditional distribution
        curve = unconditionalDistribution.drawPDF().getDrawable(0)
        curve.setLegend("Unconditional")
        curve.setLineStyle("dashed")
        graph.add(curve)
        # Plot conditional distribution
        curve = conditionalDistribution.drawPDF().getDrawable(0)
        curve.setLegend(
            f"{marginalOutputDescription} in [{lowerBound:.3e}, {upperBound:.3e}]"
        )
        graph.add(curve)
        #
        graph.setColors(ot.Drawable().BuildDefaultPalette(2))
        return graph

    dimension_input = inputSample.getDimension()
    sample_size = inputSample.getSize()
    # Filter the input, output sample
    conditionedInputSample, conditionedOutputSample = filterInputOutputSample(
        inputSample, outputSample, outputIndex, lowerBound, upperBound
    )
    conditionedSampleSize = conditionedInputSample.getSize()
    inputDescription = inputSample.getDescription()
    outputDescription = outputSample.getDescription()
    marginalOutputDescription = outputDescription[outputIndex]
    grid = ot.GridLayout(1, 1 + dimension_input)
    for i in range(dimension_input):
        # Plot unconditional distribution
        unconditionalDistribution = inputDistribution.getMarginal(i)
        conditionalDistribution = ot.KernelSmoothing().build(
            conditionedInputSample[:, i]
        )
        xTitle = "%s" % inputDescription[i]
        graph = plot_unconditional_and_conditional_distribution(
            unconditionalDistribution,
            conditionalDistribution,
            xTitle,
            marginalOutputDescription,
            lowerBound,
            upperBound,
        )
        if i > 0:
            graph.setYTitle("")
        grid.setGraph(0, i, graph)
    # Add the distribution of the output
    # Plot unconditional output distribution
    unconditionalDistribution = ot.KernelSmoothing().build(outputSample[:, outputIndex])
    conditionalDistribution = ot.KernelSmoothing().build(
        conditionedOutputSample[:, outputIndex]
    )
    outputDescription = outputSample.getDescription()
    xTitle = "%s" % outputDescription[outputIndex]
    graph = plot_unconditional_and_conditional_distribution(
        unconditionalDistribution,
        conditionalDistribution,
        xTitle,
        marginalOutputDescription,
        lowerBound,
        upperBound,
    )
    #
    graph.setYTitle("")
    graph.setLegendPosition("topright")
    grid.setGraph(0, dimension_input, graph)
    #
    grid.setTitle(
        f"Unconditioned n={sample_size}, " f"Conditioned n = {conditionedSampleSize}"
    )
    return grid


def plotConditionOutputQuantile(
    inputSample, outputSample, quantileLevel, inputDistribution
):
    """
    Plot sensitivity analysis from given quantile level.

    In this sensitivity plot, we consider the sensitivity of the inputs
    to the even that each output is exceeding a given threshold.
    This threshold is computed from the given quantile level.
    For example, if quantileLevel = 0.9, we are interested in the
    sensitivity of the inputs to the event {q(0.9) <= Y}.
    This amounts to repeated calls to plot_event_from_bounds() with
    bounds computed from the given quantile level.

    For all output marginal indices i from 0 to outputDimension - 1,
    we compute marginal output bounds as follows:
    - the minimum bound is the quantile of given level of the output marginal sample,
    - the maximum bound is the sample maximum of the output marginal sample.
    Then we call plot_event_from_bounds().
    Finally, we gather each marginal plot into a single grid of plots.

    Parameters
    ----------
    inputSample: ot.Sample(size, inputDimension)
        The input sample X.
    outputSample: ot.Sample(size, outputDimension)
        The output sample Y.
    quantileLevel: float, in [0, 1]
        The quantile level.
    inputDistribution : ot.Distribution(inputDimension)
        The distribution of the input sample.

    Return
    ------
    grid: ot.GridLayout(1, 1 + inputDimension)
        The grid of sensitivity plots.
        The i-th plot presents the unconditional and conditional distribution of the
        i-th input to the outputIndex-th output.
        The last plot presents the unconditional and conditional distribution of the
        outputIndex-th output.
    """
    outputDimension = outputSample.getDimension()
    inputDimension = inputSample.getDimension()
    grid = ot.GridLayout(outputDimension, 1 + inputDimension)
    outputDescription = outputSample.getDescription()
    for indexOutput in range(outputDimension):
        quantileLowerPoint = outputSample.computeQuantilePerComponent(quantileLevel)
        lowerValue = quantileLowerPoint[indexOutput]
        maxPoint = outputSample.getMax()
        upperValue = maxPoint[indexOutput]
        subGrid = plotConditionOutputBounds(
            inputSample,
            outputSample,
            indexOutput,
            lowerValue,
            upperValue,
            inputDistribution,
        )
        # Merge sub-graphs into the main one.
        for j in range(1 + inputDimension):
            graph = subGrid.getGraph(0, j)
            if j == inputDimension:
                graph.setLegends(
                    [
                        "Unconditional",
                        f"{outputDescription[indexOutput]} >= {lowerValue:.4e}",
                    ]
                )
            grid.setGraph(indexOutput, j, graph)

    lastSubGridTitle = subGrid.getTitle()
    grid.setTitle(f"Quantile at level {quantileLevel}, {lastSubGridTitle}")
    return grid


def computeOutputDistributionConditionalyOnInput(
    inputSample,
    outputSample,
    inputIndex,
    boundsList,
):
    """
    Condition distribution of the output conditionnaly of the input.

    Parameters
    ----------
    inputSample: ot.Sample(size, inputDimension)
        The input sample X.
    outputSample: ot.Sample(size, outputDimension)
        The output sample Y.
    inputIndex: int
        The index of an input.
    outputBoundList: list(numberOfBounds)
        The list of output bounds.

    Return
    ------
    outputDistributionList: list(numberOfCuts)
        The number of splits is equal to numberOfBounds - 1.
        Each distribution is the conditional distribution of the
        output given that the input is in the specified interval defined
        by its bounds.
    """
    if outputSample.getDimension() != 1:
        raise ValueError(
            f"Output dimension is equal to {outputSample.getDimension()}" "instead of 1"
        )
    inputDimension = inputSample.getDimension()
    if inputIndex < 0 or inputIndex >= inputDimension:
        raise ValueError(
            f"Unknown input marginal index {inputIndex}. "
            f"Must be in the [0, {inputDimension}] interval."
        )
    sampleSize = inputSample.getSize()
    if outputSample.getSize() != sampleSize:
        raise ValueError(
            f"The input sample has size {sampleSize} "
            f"but the output sample has size {outputSample.getSize()}."
        )
    # Joint the X and Y samples into a single one
    jointXYSample = joinInputOutputSample(inputSample, outputSample)
    outputMarginalIndex = inputDimension  # The output is the last column
    outputDistributionList = []
    numberOfCuts = len(boundsList) - 1
    for i in range(numberOfCuts):
        lowerBound = boundsList[i]
        upperBound = boundsList[1 + i]
        conditionnedSample = filterSample(
            jointXYSample,
            lowerBound,
            upperBound,
            inputIndex,
        )
        outputMarginalSample = conditionnedSample.getMarginal(outputMarginalIndex)
        kde = ot.KernelSmoothing().build(outputMarginalSample)
        outputDistributionList.append(kde)

    return outputDistributionList


def computeConditionInputQuantileDistributions(
    inputSample, outputSample, numberOfCuts=5
):
    """
    Condition on input with sequence of quantile levels and compute the conditional output distribution.

    Parameters
    ----------
    inputSample: ot.Sample(size, inputDimension)
        The input sample X.
    outputSample: ot.Sample(size, outputDimension)
        The output sample Y.
    numberOfCuts: int
        The number of cuts of the quantile levels.

    Return
    ------
    distributionOutputList: list(list(list(ot.Distribution())))
        This is a list of outputDimension lists.
        Each sub-list has inputDimension sub-sub-lists.
        Each sub-sub-list has numberOfCuts distributions.
        Each distribution is the distribution of the output conditionnaly
        that the input is in a given range defined by its quantiles.
    """
    sampleSize = inputSample.getSize()
    if outputSample.getSize() != sampleSize:
        raise ValueError(
            f"The size of the input sample is {sampleSize} which "
            f"does not match the size of the output sample {outputSample.getSize()}."
        )
    inputDimension = inputSample.getDimension()
    outputDimension = outputSample.getDimension()
    #
    alphaLevels = np.linspace(0.0, 1.0, 1 + numberOfCuts)
    distributionOutputList = []
    for outputIndex in range(outputDimension):
        outputMarginalSample = outputSample.getMarginal(outputIndex)
        # Compute the list of conditional distributions for all inputs
        distributionInputList = []
        for inputIndex in range(inputDimension):
            inputMarginalSample = inputSample.getMarginal(inputIndex)
            # Compute list of bounds
            boundsList = []
            for i in range(1 + numberOfCuts):
                quantilePoint = inputMarginalSample.computeQuantilePerComponent(
                    alphaLevels[i]
                )
                boundsList.append(quantilePoint[0])
            # Compute conditional distributions: condition on X, compute the distribution of Y | X
            jointXYSample = joinInputOutputSample(
                inputMarginalSample, outputMarginalSample
            )
            conditionalDistributionList = []
            numberOfCuts = len(boundsList) - 1
            for i in range(numberOfCuts):
                lowerBound = boundsList[i]
                upperBound = boundsList[1 + i]
                conditionnedSample = filterSample(
                    jointXYSample,
                    lowerBound,
                    upperBound,
                    0,
                )
                outputConditionedMarginalSample = conditionnedSample.getMarginal(1)
                kde = ot.KernelSmoothing().build(outputConditionedMarginalSample)
                conditionalDistributionList.append(kde)
            distributionInputList.append(conditionalDistributionList)
        distributionOutputList.append(distributionInputList)
    return distributionOutputList


def plotConditionInputAll(
    inputSample,
    outputSample,
    inputIndex,
    numberOfCuts=5,
):
    """
    Condition on input with sequence of quantile levels and see all plots of the conditional output.

    Parameters
    ----------
    inputSample: ot.Sample(size, inputDimension)
        The input sample X.
    outputSample: ot.Sample(size, outputDimension)
        The output sample Y.
    inputIndex: int
        The index of an input.
    numberOfCuts: int, greater than 1
        The number of cuts of the quantile levels.

    Return
    ------
    grid: ot.GridLayout(outputDimension, numberOfCuts)
        The number of splits defines partition of the [0, 1] interval into
        sub-intervals of equal lengths.
        Each sub-interval defines a interval of quantile  of the inputIndex-th input.
        Each plot represents the unconditional and conditional distribution
        of the ouput with respect to the inputIndex-th input.
        The conditional distribution of the output is defined as Y | Xi in [a, b]
        where a and b are computed from a list of quantiles of the input.
    """
    sampleSize = inputSample.getSize()
    if outputSample.getSize() != sampleSize:
        raise ValueError(
            f"The size of the input sample is {sampleSize} which "
            f"does not match the size of the output sample {outputSample.getSize()}."
        )
    inputDimension = inputSample.getDimension()
    if inputIndex < 0 or inputIndex > inputDimension:
        raise ValueError(
            f"Input marginal index {inputIndex} is not in " f"[0, {inputDimension}]"
        )
    if numberOfCuts < 1:
        raise ValueError(
            f"The number of splits must be larger than 1, but is equal to {numberOfCuts}."
        )
    outputDimension = outputSample.getDimension()
    inputDescription = inputSample.getDescription()
    outputDescription = outputSample.getDescription()
    marginalInputSample = inputSample.getMarginal(inputIndex)
    distributionOutputList = computeConditionInputQuantileDistributions(
        marginalInputSample, outputSample, numberOfCuts
    )
    grid = ot.GridLayout(outputDimension, numberOfCuts)
    grid.setTitle(
        f"Sensitivity of {inputDescription[inputIndex]}, " f"n = {sampleSize}"
    )
    for outputIndex in range(outputDimension):
        distributionInputList = distributionOutputList[outputIndex]
        conditionalDistributionList = distributionInputList[
            0
        ]  # There is only one input considered here
        outputMarginalSample = outputSample.getMarginal(outputIndex)
        # Unconditional output distribution
        outputDistribution = ot.KernelSmoothing().build(outputMarginalSample)
        unconditionalPDFPlot = outputDistribution.drawPDF()
        unconditionalPDFCurve = unconditionalPDFPlot.getDrawable(0)
        unconditionalPDFCurve.setLineStyle("dashed")
        unconditionalPDFCurve.setLegend("Unconditional")
        # Comput common bounding box
        unconditionalPDFBoundingBox = unconditionalPDFPlot.getBoundingBox()
        unconditionalLowerBound = unconditionalPDFBoundingBox.getLowerBound()
        unconditionalUpperBound = unconditionalPDFBoundingBox.getUpperBound()
        ymin = unconditionalUpperBound[0]
        ymax = unconditionalUpperBound[1]
        #
        # Compute list of bounds
        alphaLevels = np.linspace(0.0, 1.0, 1 + numberOfCuts)
        # Search for maximum PDF
        for i in range(numberOfCuts):
            conditionalDistribution = conditionalDistributionList[i]
            curve = conditionalDistribution.drawPDF()
            curveMax = curve.getBoundingBox().getUpperBound()[1]
            ymax = max(ymax, curveMax)
        # Set common interval
        commonInterval = ot.Interval(unconditionalLowerBound, [ymin, ymax])
        for i in range(numberOfCuts):
            alphaLevelMin = alphaLevels[i]
            alphaLevelMax = alphaLevels[i + 1]
            conditionalDistribution = conditionalDistributionList[i]
            graph = ot.Graph("", f"{outputDescription[outputIndex]}", "PDF", True)
            graph.add(unconditionalPDFCurve)
            curve = conditionalDistribution.drawPDF().getDrawable(0)
            curve.setLegend("Conditional")
            graph.add(curve)
            #
            if outputIndex == 0:
                graph.setTitle(
                    f"{inputDescription[inputIndex]} in [{alphaLevelMin:.2f}, {alphaLevelMax:.2f}]"
                )
            if i < numberOfCuts - 1:
                graph.setLegends([""])
            graph.setColors(ot.Drawable().BuildDefaultPalette(2))
            if i > 0:
                graph.setYTitle("")
            if i == numberOfCuts - 1:
                graph.setLegendPosition("topright")
            graph.setBoundingBox(commonInterval)
            grid.setGraph(outputIndex, i, graph)
    return grid


def createLighterPalette(baseColor, minimumValue, maximumValue, numberOfColors):
    """ "
    Create a palette based on a base color and bounds of the value channel.

    In the HSV color map, the value (or V) channel is the "brightness" of the color.
    This function creates a list of colors - or palette - by creating a
    linear scale between two bounds.

    Parameters
    ----------
    baseColor: [r, g, b]
        The base hexadecimal color in RGB color space where
        0 <= r <= 1, 0 <= g <= 1, 0 <= b <= 1.
    minimumValue: float, in [0, 1]
        The minimum value.
    maximumValue: float, in [0, 1]
        The maximum value.
    numberOfColors: int
        The number of colors in the palette.

    Return
    ------
    colorPalette: list(numberOfColors)
        The list of colors strings as hexadecimal codes.
    """
    if minimumValue > maximumValue:
        raise ValueError(
            f"The minimum value {minimumValue} is greater "
            f"than the maximum value {maximumValue}."
        )
    if numberOfColors < 1:
        raise ValueError(f"The number of colors {numberOfColors} is lower than 1.")
    r, g, b = baseColor
    h, s, v = ot.Drawable.ConvertFromRGBIntoHSV(r, g, b)
    valueArray = np.linspace(minimumValue, maximumValue, numberOfColors)
    colorPalette = []
    for i in range(numberOfColors):
        hexColor = ot.Drawable.ConvertFromHSV(h, s, valueArray[i])
        colorPalette.append(hexColor)
    return colorPalette


def createLighterPalette(baseColor, minimumValue, maximumValue, numberOfColors):
    """ "
    Create a palette based on a base color and bounds of the value channel.

    In the HSV color map, the value (or V) channel is the "brightness" of the color.
    This function creates a list of colors - or palette - by creating a
    linear scale between two bounds.

    Parameters
    ----------
    baseColor: [r, g, b]
        The base hexadecimal color in RGB color space where
        0 <= r <= 1, 0 <= g <= 1, 0 <= b <= 1.
    minimumValue: float, in [0, 1]
        The minimum value.
    maximumValue: float, in [0, 1]
        The maximum value.
    numberOfColors: int
        The number of colors in the palette.

    Return
    ------
    colorPalette: list(numberOfColors)
        The list of colors strings as hexadecimal codes.
    """
    if len(baseColor) != 3:
        raise ValueError(
            f"The number of colors in baseColor is {len(baseColor)} but it should be equal to 3."
        )
    if minimumValue > maximumValue:
        raise ValueError(
            f"The minimum value {minimumValue} is greater "
            f"than the maximum value {maximumValue}."
        )
    if numberOfColors < 1:
        raise ValueError(f"The number of colors {numberOfColors} is lower than 1.")
    r, g, b = baseColor
    h, s, v = ot.Drawable.ConvertFromRGBIntoHSV(r, g, b)
    valueArray = np.linspace(minimumValue, maximumValue, numberOfColors)
    colorPalette = []
    for i in range(numberOfColors):
        hexColor = ot.Drawable.ConvertFromHSV(h, s, valueArray[i])
        colorPalette.append(hexColor)
    return colorPalette


def plotConditionInputQuantileSequence(
    inputSample,
    outputSample,
    numberOfCuts=5,
    minimumColorValue=0.5,
    maximumColorValue=1.0,
):
    """
    Condition on input with sequence of quantile levels and see the conditional output.

    Parameters
    ----------
    inputSample: ot.Sample(size, inputDimension)
        The input sample X.
    outputSample: ot.Sample(size, outputDimension)
        The output sample Y.
    numberOfCuts: int, greater than 1
        The number of cuts of the quantile levels.

    Return
    ------
    grid: ot.GridLayout(outputDimension, inputDimension)
        The outputIndex-th, inputIndex-th plot presents all the
        conditional distributions of the output when the inputs has
        a conditional distribution in a given interval.
    """
    sampleSize = inputSample.getSize()
    if outputSample.getSize() != sampleSize:
        raise ValueError(
            f"The size of the input sample is {sampleSize} which "
            f"does not match the size of the output sample {outputSample.getSize()}."
        )
    if numberOfCuts < 1:
        raise ValueError(
            f"The number of splits must be larger than 1, but is equal to {numberOfCuts}."
        )
    if (
        minimumColorValue > maximumColorValue
        or minimumColorValue < 0.0
        or maximumColorValue > 1.0
    ):
        raise ValueError(
            f"The minimum color value is equal to {minimumColorValue} (must be >= 0.0) and "
            f"the maximum color value is equal to {maximumColorValue} (must be <= 1.0), which is inconsistent."
        )
    inputDimension = inputSample.getDimension()
    outputDimension = outputSample.getDimension()
    inputDescription = inputSample.getDescription()
    outputDescription = outputSample.getDescription()
    #
    # Join each plot into a single one.
    baseColorPalette = ot.Drawable().BuildDefaultPalette(2)
    unconditionalColor = baseColorPalette[0]
    baseConditionalColor = baseColorPalette[1]
    baseColor = ot.Drawable.ConvertToRGB(baseConditionalColor)
    conditionalColorPalette = createLighterPalette(
        baseColor, minimumColorValue, maximumColorValue, numberOfCuts
    )
    #
    alphaLevels = np.linspace(0.0, 1.0, 1 + numberOfCuts)
    distributionOutputList = computeConditionInputQuantileDistributions(
        inputSample, outputSample, numberOfCuts
    )
    grid = ot.GridLayout(outputDimension, inputDimension)
    grid.setTitle(f"n = {sampleSize}, number of quantiles : {numberOfCuts}")
    for outputIndex in range(outputDimension):
        distributionInputList = distributionOutputList[outputIndex]
        outputMarginalSample = outputSample.getMarginal(outputIndex)
        # Unconditional output distribution
        outputDistribution = ot.KernelSmoothing().build(outputMarginalSample)
        unconditionalPDFPlot = outputDistribution.drawPDF()
        unconditionalPDFCurve = unconditionalPDFPlot.getDrawable(0)
        unconditionalPDFCurve.setLineStyle("dashed")
        unconditionalPDFCurve.setColor(unconditionalColor)
        # Compute common bounding box
        unconditionalPDFCurveBoundingBox = unconditionalPDFPlot.getBoundingBox()
        unconditionalPDFLowerBound = unconditionalPDFCurveBoundingBox.getLowerBound()
        unconditionalPDFUpperBound = unconditionalPDFCurveBoundingBox.getUpperBound()
        ymin = unconditionalPDFUpperBound[0]
        ymax = unconditionalPDFUpperBound[1]
        for inputIndex in range(inputDimension):
            conditionalDistributionList = distributionInputList[inputIndex]
            for i in range(numberOfCuts):
                conditionalDistribution = conditionalDistributionList[i]
                curve = conditionalDistribution.drawPDF()
                curveMaximumDensity = curve.getBoundingBox().getUpperBound()[1]
                ymax = max(ymax, curveMaximumDensity)

        # Set common interval
        interval = ot.Interval(unconditionalPDFLowerBound, [ymin, ymax])
        for inputIndex in range(inputDimension):
            conditionalDistributionList = distributionInputList[inputIndex]
            graph = ot.Graph(
                inputDescription[inputIndex],
                outputDescription[outputIndex],
                "PDF",
                True,
            )
            if outputIndex > 0:
                graph.setTitle("")
            if inputIndex > 0:
                graph.setYTitle("")
            graph.add(unconditionalPDFCurve)
            for i in range(numberOfCuts):
                conditionalDistribution = conditionalDistributionList[i]
                curve = conditionalDistribution.drawPDF().getDrawable(0)
                curve.setLegend(f"[{alphaLevels[i]:.1f}, {alphaLevels[1 + i]:.1f}]")
                curve.setLineWidth(1.0)
                curve.setColor(conditionalColorPalette[i])
                graph.add(curve)
            graph.setBoundingBox(interval)
            if inputIndex < inputDimension - 1:
                graph.setLegends([""])
            else:
                graph.setLegendPosition("topright")
            grid.setGraph(outputIndex, inputIndex, graph)
    return grid


def computeConditionOutputQuantileDistributions(
    inputSample, outputSample, numberOfCuts=5
):
    """
    Condition on output with sequence of quantile levels and compute the conditional input distribution.

    See computeConditionInputQuantileDistributions() for the procedure when we condition
    on input.

    Parameters
    ----------
    inputSample: ot.Sample(size, inputDimension)
        The input sample X.
    outputSample: ot.Sample(size, outputDimension)
        The output sample Y.
    numberOfCuts: int
        The number of cuts of the quantile levels.

    Return
    ------
    distributionOutputList: list(list(list(ot.Distribution())))
        This is a list of outputDimension lists.
        Each sub-list has inputDimension sub-sub-lists.
        Each sub-sub-list has numberOfCuts distributions.
        Each distribution is the distribution of the output conditionnaly
        that the input is in a given range defined by its quantiles.
    """
    sampleSize = inputSample.getSize()
    if outputSample.getSize() != sampleSize:
        raise ValueError(
            f"The size of the input sample is {sampleSize} which "
            f"does not match the size of the output sample {outputSample.getSize()}."
        )
    inputDimension = inputSample.getDimension()
    outputDimension = outputSample.getDimension()
    #
    alphaLevels = np.linspace(0.0, 1.0, 1 + numberOfCuts)
    distributionInputList = []
    for inputIndex in range(inputDimension):
        inputMarginalSample = inputSample.getMarginal(inputIndex)
        # Compute the list of conditional distributions for all outputs
        distributionOutputList = []
        for outputIndex in range(outputDimension):
            outputMarginalSample = outputSample.getMarginal(outputIndex)
            # Compute list of bounds
            boundsList = []
            for i in range(1 + numberOfCuts):
                quantilePoint = outputMarginalSample.computeQuantilePerComponent(
                    alphaLevels[i]
                )
                boundsList.append(quantilePoint[0])
            # Compute conditional distributions: condition on Y, compute the distribution of X | Y
            jointXYSample = joinInputOutputSample(
                inputMarginalSample, outputMarginalSample
            )
            conditionalDistributionList = []
            numberOfCuts = len(boundsList) - 1
            for i in range(numberOfCuts):
                lowerBound = boundsList[i]
                upperBound = boundsList[1 + i]
                conditionnedSample = filterSample(
                    jointXYSample,
                    lowerBound,
                    upperBound,
                    1,
                )
                inputConditionedMarginalSample = conditionnedSample.getMarginal(0)
                kde = ot.KernelSmoothing().build(inputConditionedMarginalSample)
                conditionalDistributionList.append(kde)
            distributionOutputList.append(conditionalDistributionList)
        distributionInputList.append(distributionOutputList)
    return distributionInputList


def plotConditionOutputAll(
    inputSample,
    outputSample,
    outputIndex,
    numberOfCuts=5,
):
    """
    Condition on output with sequence of quantile levels and see all plots of the conditional input.

    See plotConditionInputAll() for the same procedure when
    we condition on input.

    Parameters
    ----------
    inputSample: ot.Sample(size, inputDimension)
        The input sample X.
    outputSample: ot.Sample(size, outputDimension)
        The output sample Y.
    outputIndex: int
        The index of an output.
    numberOfCuts: int, greater than 1
        The number of cuts of the quantile levels.

    Return
    ------
    grid: ot.GridLayout(inputDimension, numberOfCuts)
        The number of splits defines partition of the [0, 1] interval into
        sub-intervals of equal lengths.
        Each sub-interval defines a interval of quantile  of the inputIndex-th input.
        Each plot represents the unconditional and conditional distribution
        of the ouput with respect to the inputIndex-th input.
        The conditional distribution of the output is defined as Xi | Y in [a, b]
        where a and b are computed from a list of quantiles of the output.
    """
    sampleSize = inputSample.getSize()
    if outputSample.getSize() != sampleSize:
        raise ValueError(
            f"The size of the input sample is {sampleSize} which "
            f"does not match the size of the output sample {outputSample.getSize()}."
        )
    outputDimension = outputSample.getDimension()
    if outputIndex < 0 or outputIndex > outputDimension:
        raise ValueError(
            f"Output marginal index {outputIndex} is not in " f"[0, {outputDimension}]"
        )
    if numberOfCuts < 1:
        raise ValueError(
            f"The number of splits must be larger than 1, but is equal to {numberOfCuts}."
        )
    inputDimension = inputSample.getDimension()
    inputDescription = inputSample.getDescription()
    outputDescription = outputSample.getDescription()
    marginalOutputSample = outputSample.getMarginal(outputIndex)
    distributionInputList = computeConditionOutputQuantileDistributions(
        inputSample, marginalOutputSample, numberOfCuts
    )
    grid = ot.GridLayout(inputDimension, numberOfCuts)
    grid.setTitle(
        f"Sensitivity of {outputDescription[outputIndex]}, " f"n = {sampleSize}"
    )
    for inputIndex in range(inputDimension):
        distributionOutputList = distributionInputList[inputIndex]
        conditionalDistributionList = distributionOutputList[
            0
        ]  # There is only one output considered here
        inputMarginalSample = inputSample.getMarginal(inputIndex)
        # Unconditional input distribution. TODO: get the input distribution here.
        inputDistribution = ot.KernelSmoothing().build(inputMarginalSample)
        unconditionalPDFPlot = inputDistribution.drawPDF()
        unconditionalPDFCurve = unconditionalPDFPlot.getDrawable(0)
        unconditionalPDFCurve.setLineStyle("dashed")
        unconditionalPDFCurve.setLegend("Unconditional")
        # Comput common bounding box
        unconditionalPDFBoundingBox = unconditionalPDFPlot.getBoundingBox()
        unconditionalLowerBound = unconditionalPDFBoundingBox.getLowerBound()
        unconditionalUpperBound = unconditionalPDFBoundingBox.getUpperBound()
        ymin = unconditionalUpperBound[0]
        ymax = unconditionalUpperBound[1]
        #
        # Compute list of bounds
        alphaLevels = np.linspace(0.0, 1.0, 1 + numberOfCuts)
        # Search for maximum PDF
        for i in range(numberOfCuts):
            conditionalDistribution = conditionalDistributionList[i]
            curve = conditionalDistribution.drawPDF()
            curveMax = curve.getBoundingBox().getUpperBound()[1]
            ymax = max(ymax, curveMax)
        # Set common interval
        commonInterval = ot.Interval(unconditionalLowerBound, [ymin, ymax])
        for i in range(numberOfCuts):
            alphaLevelMin = alphaLevels[i]
            alphaLevelMax = alphaLevels[i + 1]
            conditionalDistribution = conditionalDistributionList[i]
            graph = ot.Graph("", f"{inputDescription[inputIndex]}", "PDF", True)
            graph.add(unconditionalPDFCurve)
            curve = conditionalDistribution.drawPDF().getDrawable(0)
            curve.setLegend("Conditional")
            graph.add(curve)
            #
            if inputIndex == 0:
                graph.setTitle(
                    f"{outputDescription[outputIndex]} in [{alphaLevelMin:.2f}, {alphaLevelMax:.2f}]"
                )
            if i < numberOfCuts - 1:
                graph.setLegends([""])
            graph.setColors(ot.Drawable().BuildDefaultPalette(2))
            if i > 0:
                graph.setYTitle("")
            if i == numberOfCuts - 1:
                graph.setLegendPosition("topright")
            graph.setBoundingBox(commonInterval)
            grid.setGraph(inputIndex, i, graph)
    return grid
