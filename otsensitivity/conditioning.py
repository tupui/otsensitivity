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
import numpy as np


# %%
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
    inputDimension = inputSample.getDimension()
    sampleSize = inputSample.getSize()
    if outputSample.getSize() != sampleSize:
        raise ValueError(
            f"The size of the input sample is {sampleSize} which "
            f"does not match the size of the output sample {outputSample.getSize()}."
        )
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
    sample : ot.Sample
        L'échantillon.
    quantile_value : TYPE
        DESCRIPTION.

    """
    if upperBound < lowerBound:
        raise ValueError(
            f"The lower bound {lowerBound} is greater "
            f"than the upper bound {upperBound}."
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


def computeOutputConditionalDistribution(
    inputSample, outputSample, inputMarginalIndex, boundsList, verbose=False
):
    """
    Condition the input and compute the conditional distribution of the output.

    Parameters
    ----------
    inputSample: ot.Sample(size, inputDimension)
        The input sample X.
    outputSample: ot.Sample(size, outputDimension)
        The output sample Y.
    inputMarginalIndex: int
        The index of an input.
    outputBoundList: list(numberOfBounds)
        The list of output bounds.

    Return
    ------
    outputDistributionList: list(numberOfSplit)
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
    if inputMarginalIndex < 0 or inputMarginalIndex >= inputDimension:
        raise ValueError(
            f"Unknown input marginal index {inputMarginalIndex}. "
            f"Must be in the [0, {inputDimension}] interval."
        )
    sampleSize = inputSample.getSize()
    if outputSample.getSize() != sampleSize:
        raise ValueError(
            f"The input sample has size {sampleSize} "
            f"but the output sample has size {outputSample.getSize()}."
        )
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
    outputMarginalIndex = inputDimension
    #
    if verbose:
        print("| i / k | lower bound | upper bound | size |")
        print("|-------|-------------|-------------|------|")
    outputDistributionList = []
    numberOfSplit = len(boundsList) - 1
    for i in range(numberOfSplit):
        lowerBound = boundsList[i]
        upperBound = boundsList[1 + i]
        conditionnedSample = filterSample(
            jointXYSample,
            lowerBound,
            upperBound,
            inputMarginalIndex,
        )
        if verbose:
            print(
                f"| {i} / {numberOfSplit} "
                f"| {lowerBound:.2f} "
                f"| {upperBound:.2f} "
                f"| {conditionnedSample.getSize()} |"
            )
        outputMarginalSample = conditionnedSample.getMarginal(outputMarginalIndex)
        kde = ot.KernelSmoothing().build(outputMarginalSample)
        outputDistributionList.append(kde)

    return outputDistributionList


def plotConditionInputAll(
    inputSample, outputSample, inputMarginalIndex, numberOfSplit=5, verbose=False
):
    """
    Condition on input with sequence of quantile levels and see all plots of the conditional output.

    Parameters
    ----------
    inputSample: ot.Sample(size, inputDimension)
        The input sample X.
    outputSample: ot.Sample(size, outputDimension)
        The output sample Y.
    inputMarginalIndex: int
        The index of an input.
    outputBoundList: list(numberOfBounds)
        The list of output bounds.

    Return
    ------
    grid: ot.GridLayout(outputDimension, numberOfSplit)
        The number of splits defines partition of the [0, 1] interval into
        sub-intervals of equal lengths.
        Each sub-interval defines a interval of quantile  of the inputMarginalIndex-th input.
        Each plot represents the unconditional and conditional distribution
        of the ouput with respect to the inputMarginalIndex-th input.
        The conditional distribution of the output is defined as Y | Xi in [a, b]
        where a and b are computed from a list of quantiles of the input.
    """
    inputDimension = inputSample.getDimension()
    outputDimension = outputSample.getDimension()
    sampleSize = inputSample.getSize()
    inputDescription = inputSample.getDescription()
    outputDescription = outputSample.getDescription()
    if inputMarginalIndex < 0 or inputMarginalIndex > inputDimension:
        raise ValueError(
            f"Input marginal index {inputMarginalIndex} is not in "
            f"[0, {inputDimension}]"
        )
    grid = ot.GridLayout(outputDimension, numberOfSplit)
    for outputIndex in range(outputDimension):
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
        grid.setTitle(
            f"Sensitivity of {inputDescription[inputMarginalIndex]}, "
            f"n = {sampleSize}"
        )
        #
        # Compute list of bounds
        alphaLevels = np.linspace(0.0, 1.0, 1 + numberOfSplit)
        boundsList = []
        for i in range(1 + numberOfSplit):
            quantilePoint = inputSample.computeQuantilePerComponent(alphaLevels[i])
            boundsList.append(quantilePoint[inputMarginalIndex])
        #
        # Compute conditional distributions
        distributionList = computeOutputConditionalDistribution(
            inputSample,
            outputMarginalSample,
            inputMarginalIndex,
            boundsList,
            verbose=verbose,
        )
        # Search for maximum PDF
        for i in range(numberOfSplit):
            conditionalDistribution = distributionList[i]
            curve = conditionalDistribution.drawPDF()
            curveMax = curve.getBoundingBox().getUpperBound()[1]
            ymax = max(ymax, curveMax)
        # Set common interval
        commonInterval = ot.Interval(unconditionalLowerBound, [ymin, ymax])
        for i in range(numberOfSplit):
            alphaLevelMin = alphaLevels[i]
            alphaLevelMax = alphaLevels[i + 1]
            conditionalDistribution = distributionList[i]
            graph = ot.Graph("", f"{outputDescription[outputIndex]}", "PDF", True)
            graph.add(unconditionalPDFCurve)
            curve = conditionalDistribution.drawPDF().getDrawable(0)
            curve.setLegend("Conditional")
            graph.add(curve)
            #
            if outputIndex == 0:
                graph.setTitle(
                    f"{inputDescription[inputMarginalIndex]} in [{alphaLevelMin:.2f}, {alphaLevelMax:.2f}]"
                )
            if i < numberOfSplit - 1:
                graph.setLegends([""])
            graph.setColors(ot.Drawable().BuildDefaultPalette(2))
            if i > 0:
                graph.setYTitle("")
            if i == numberOfSplit - 1:
                graph.setLegendPosition("topright")
            graph.setBoundingBox(commonInterval)
            grid.setGraph(outputIndex, i, graph)
    return grid


# %%
def createLighterPalette(baseColor, minimumValue, maximumValue, numberOfColors):
    r, g, b = baseColor
    h, s, v = ot.Drawable.ConvertFromRGBIntoHSV(r, g, b)
    valueArray = np.linspace(minimumValue, maximumValue, numberOfColors)
    colorPalette = []
    for i in range(numberOfColors):
        hexColor = ot.Drawable.ConvertFromHSV(h, s, valueArray[i])
        colorPalette.append(hexColor)
    return colorPalette


def plotConditionInputQuantileSequence(inputSample, outputSample, numberOfSplit=5):
    """
    Condition on input with sequence of quantile levels and see the conditional output.

    Parameters
    ----------
    inputSample: ot.Sample(size, inputDimension)
        The input sample X.
    outputSample: ot.Sample(size, outputDimension)
        The output sample Y.
    numberOfSplit: int
        The number of cuts of the quantile levels.

    Return
    ------
    grid: ot.GridLayout(outputDimension, inputDimension)
        The outputIndex-th, inputIndex-th plot presents all the
        conditional distributions of the output when the inputs has
        a conditional distribution in a given interval.
    """
    sampleSize = inputSample.getSize()
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
    conditionalColorPalette = createLighterPalette(baseColor, 0.5, 1.0, numberOfSplit)
    #
    alphaLevels = np.linspace(0.0, 1.0, 1 + numberOfSplit)
    grid = ot.GridLayout(outputDimension, inputDimension)
    grid.setTitle(f"n = {sampleSize}, number of quantiles : {numberOfSplit}")
    for outputIndex in range(outputDimension):
        outputMarginalSample = outputSample.getMarginal(outputIndex)
        # Unconditional output distribution
        outputDistribution = ot.KernelSmoothing().build(outputMarginalSample)
        unconditionalPDFPlot = outputDistribution.drawPDF()
        unconditionalPDFCurve = unconditionalPDFPlot.getDrawable(0)
        unconditionalPDFCurve.setLineStyle("dashed")
        unconditionalPDFCurve.setColor(unconditionalColor)
        # Compute the list of conditional distributions for all inputs
        distributionAllInputs = []
        for inputMarginalIndex in range(inputDimension):
            # Compute list of bounds
            boundsList = []
            for i in range(1 + numberOfSplit):
                quantilePoint = inputSample.computeQuantilePerComponent(alphaLevels[i])
                boundsList.append(quantilePoint[inputMarginalIndex])
            # Compute conditional distributions
            distributionList = computeOutputConditionalDistribution(
                inputSample,
                outputMarginalSample,
                inputMarginalIndex,
                boundsList,
                verbose=False,
            )
            distributionAllInputs.append(distributionList)
        # Compute common bounding box
        unconditionalPDFCurveBoundingBox = unconditionalPDFPlot.getBoundingBox()
        unconditionalPDFLowerBound = unconditionalPDFCurveBoundingBox.getLowerBound()
        unconditionalPDFUpperBound = unconditionalPDFCurveBoundingBox.getUpperBound()
        ymin = unconditionalPDFUpperBound[0]
        ymax = unconditionalPDFUpperBound[1]
        for inputMarginalIndex in range(inputDimension):
            distributionList = distributionAllInputs[inputMarginalIndex]
            for i in range(numberOfSplit):
                conditionalDistribution = distributionList[i]
                curve = conditionalDistribution.drawPDF()
                curveMaximumDensity = curve.getBoundingBox().getUpperBound()[1]
                ymax = max(ymax, curveMaximumDensity)

        # Set common interval
        interval = ot.Interval(unconditionalPDFLowerBound, [ymin, ymax])
        for inputMarginalIndex in range(inputDimension):
            distributionList = distributionAllInputs[inputMarginalIndex]
            graph = ot.Graph(
                inputDescription[inputMarginalIndex],
                outputDescription[outputIndex],
                "PDF",
                True,
            )
            if outputIndex > 0:
                graph.setTitle("")
            if inputMarginalIndex > 0:
                graph.setYTitle("")
            graph.add(unconditionalPDFCurve)
            for i in range(numberOfSplit):
                conditionalDistribution = distributionList[i]
                curve = conditionalDistribution.drawPDF().getDrawable(0)
                curve.setLegend(f"[{alphaLevels[i]:.1f}, {alphaLevels[1 + i]:.1f}]")
                curve.setLineWidth(1.0)
                curve.setColor(conditionalColorPalette[i])
                graph.add(curve)
            graph.setBoundingBox(interval)
            if inputMarginalIndex < inputDimension - 1:
                graph.setLegends([""])
            else:
                graph.setLegendPosition("topright")
            grid.setGraph(outputIndex, inputMarginalIndex, graph)
    return grid
