# Copyright (C) Michaël Baudin (2023)
# -*- coding: utf-8 -*-
"""
Let Y=g(X) be the scalar output of
the model g with vector input X with dimension nx.
Let a < b be two real numbers.
We consider the event {a < Y < b}.
We want to compute the sensitivity of that event with respect to each input Xi.

This script computes the conditional distribution of the
input Xi given that the output Y is
in the interval [a, b], for i=1,...,nx.
Compare that conditional distribution with
the unconditional distribution of Xi:
if there is no difference, then the input Xi is not influential for that event.

TODO-List
---------
- Implement a RSA with binary threshold on output: Y < s and Y > s.
  This is an option for plotConditionOutputQuantile, which replaces the 
  unconditional distribution with the opposite event.
- Rename plotConditionOutputQuantile into plotConditionOutputThreshold.
- Implement the Kolmogorov-Smirnov statistic in plotConditionOutputThreshold
  as suggested in (Pianosi, 2016) eq. 7 page 221.
  This can only be done with the CDF option.
  Move the PDF/CDF option to a attribute that does not have a "draw" name.
  Can we suggest a statistic for a PDF?

References
----------
- Estimating Global Sensitivity Measures:
  Torturing the Data Until They Confess.
  Elmar Plischke. Institut für Endlagerforschung. TU Clausthal
  St. Étienne, MASCOT-NUM, April 10, 2015.
- Pianosi, F., Beven, K., Freer, J., Hall, J. W., Rougier, J., 
  Stephenson, D. B., & Wagener, T. (2016). Sensitivity analysis of 
  environmental models: A systematic review with practical workflow. 
  Environmental Modelling & Software, 79, 214-232.
- Young, P.C., Spear, R.C., Hornberger, G.M., 1978. Modeling badly defined systems:
  some further thoughts. In: Proceedings SIMSIG Conference, Canberra,
  pp. 24-32.
- Spear, R., Hornberger, G., 1980. Eutrophication in peel inlet. II. Identification of
  critical uncertainties via generalized sensitivity analysis. Water Res. 14 (1),
  43-49.
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
        The joint (X, Y) sample with dimension equal
        to inputDimension + outputDimension.
    """
    sampleSize = inputSample.getSize()
    if outputSample.getSize() != sampleSize:
        raise ValueError(
            f"The size of the input sample is {sampleSize} which "
            "does not match the size of the "
            f"output sample {outputSample.getSize()}."
        )
    jointXYSample = ot.Sample(inputSample)  # Make a copy
    jointXYSample.stack(outputSample)
    return jointXYSample


def filterInputOutputSample(
    inputSample, outputSample, outputIndex, lowerBound, upperBound
):
    """
    Filter out the rows in the input and output sample.

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
            "does not match the size of the "
            f"output sample {outputSample.getSize()}."
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
    totalDimension = inputDimension + outputDimension
    conditionedOutputSample = conditionedXYSample[:, inputDimension:totalDimension]
    return conditionedInputSample, conditionedOutputSample


def createLighterPalette(
    baseColor, minimumValue, maximumValue, numberOfColors, alpha=0.75
):
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
        We must have maximumValue >= minimumValue.
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
        hexColor = ot.Drawable.ConvertFromHSVA(h, s, valueArray[i], alpha)
        colorPalette.append(hexColor)
    return colorPalette


class RegionalSensitivityAnalysis:
    def ComputeConditionOutputQuantileDistributions(
        inputSample, outputSample, densityEstimator, numberOfCuts=5
    ):
        """
        Condition on output with quantile levels and compute the conditional input distribution.

        See ComputeConditionInputQuantileDistributions() for the procedure when we condition
        on input.

        Parameters
        ----------
        inputSample: ot.Sample(size, inputDimension)
            The input sample X.
        outputSample: ot.Sample(size, outputDimension)
            The output sample Y.
        densityEstimator: ot.DistributionFactory()
            The density estimator.
        numberOfCuts: int
            The number of cuts of the quantile levels.

        Return
        ------
        distributionInputList: list(list(list(ot.Distribution())))
            This is a list of intputDimension lists.
            Each sub-list has outputDimension sub-sub-lists.
            Each sub-sub-list has numberOfCuts distributions.
            Each distribution is the distribution of the intput conditionnaly
            that the output is in a given range defined by its quantiles.
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
                    kde = densityEstimator.build(inputConditionedMarginalSample)
                    conditionalDistributionList.append(kde)
                distributionOutputList.append(conditionalDistributionList)
            distributionInputList.append(distributionOutputList)
        return distributionInputList

    def ComputeConditionInputQuantileDistributions(
        inputSample, outputSample, densityEstimator, numberOfCuts=5
    ):
        """
        Condition on input with quantile levels and compute the conditional output distribution.

        Parameters
        ----------
        inputSample: ot.Sample(size, inputDimension)
            The input sample X.
        outputSample: ot.Sample(size, outputDimension)
            The output sample Y.
        densityEstimator: ot.DistributionFactory()
            The density estimator.
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
                    kde = densityEstimator.build(outputConditionedMarginalSample)
                    conditionalDistributionList.append(kde)
                distributionInputList.append(conditionalDistributionList)
            distributionOutputList.append(distributionInputList)
        return distributionOutputList

    def __init__(
        self,
        inputSample,
        outputSample,
        densityEstimator=ot.KernelSmoothing(),
        isDrawPDF=True,
    ):
        """
        Create a Regional Sensitivity Analysis object.

        Parameters
        ----------
        inputSample: ot.Sample(size, inputDimension)
            The input sample X.
        outputSample: ot.Sample(size, outputDimension)
            The output sample Y.
        densityEstimator: ot.DistributionFactory()
            The density estimator.
            Default is KernelSmoothing.
        isDrawPDF: boolean
            If True, then draw the PDF.
            Otherwise, draw the CDF.

        """
        self.inputSample = inputSample
        self.outputSample = outputSample
        sampleSize = inputSample.getSize()
        if outputSample.getSize() != sampleSize:
            raise ValueError(
                f"The size of the input sample is {sampleSize} which "
                f"does not match the size of the output sample {outputSample.getSize()}."
            )
        self.densityEstimator = densityEstimator
        self.isDrawPDF = isDrawPDF
        return

    def setIsDrawPDF(self, isDrawPDF):
        self.isDrawPDF = isDrawPDF
        return

    def plotConditionOutputBounds(
        self,
        lowerBound,
        upperBound,
        inputDistribution,
        outputIndex=0,
    ):
        """
        Plot the sensitivity of the output with respect to the input.

        Let Y=g(X) be the scalar output of
        the model g with vector input X with dimension nx.
        Let a < b be two real numbers.
        We consider the event {a <= Y < b}.
        We want to compute the sensitivity of that event with
        respect to each input Xi.

        This script computes the conditional distribution of the
        input Xi given that the output Y is
        in the interval [a, b], for i=1,...,nx.
        Compare that conditional distribution with
        the unconditional distribution of Xi:
        if there is no difference, then the input Xi is not influential
        for that event.

        Parameters
        ----------
        lowerBound : float
            The lower bound for filtering.
        upperBound : float
            The upper bound for filtering.
        inputDistribution : ot.Distribution(inputDimension)
            The distribution of the input sample.
        outputIndex : int
            The index of a column in the output sample.
            Must be in the set {0, ..., outputDimension - 1}.
            The default value is outputIndex=0.

        Return
        ------
        grid: ot.GridLayout(1, 1 + inputDimension)
            The grid of sensitivity plots.
            The i-th plot presents the unconditional and conditional
            distribution of the i-th input to the outputIndex-th output.
            The last plot presents the unconditional and conditional
            distribution of the outputIndex-th output.
        """

        def plot_unconditional_and_conditional_distribution(
            unconditionalDistribution,
            conditionalDistribution,
            xTitle,
            marginalOutputDescription,
            lowerBound,
            upperBound,
        ):
            yTitle = "PDF" if self.isDrawPDF else "CDF"
            graph = ot.Graph("", xTitle, yTitle, True)
            # Plot unconditional distribution
            if self.isDrawPDF:
                curve = unconditionalDistribution.drawPDF().getDrawable(0)
            else:
                curve = unconditionalDistribution.drawCDF().getDrawable(0)
            curve.setLegend("Unconditional")
            curve.setLineStyle("dashed")
            graph.add(curve)
            # Plot conditional distribution
            if self.isDrawPDF:
                curve = conditionalDistribution.drawPDF().getDrawable(0)
            else:
                curve = conditionalDistribution.drawCDF().getDrawable(0)
            curve.setLegend(
                f"{marginalOutputDescription} in "
                f"[{lowerBound:.3e}, {upperBound:.3e}]"
            )
            graph.add(curve)
            #
            graph.setColors(ot.Drawable().BuildDefaultPalette(2))
            return graph

        dimension_input = self.inputSample.getDimension()
        sample_size = self.inputSample.getSize()
        # Filter the input, output sample
        conditionedInputSample, conditionedOutputSample = filterInputOutputSample(
            self.inputSample, self.outputSample, outputIndex, lowerBound, upperBound
        )
        conditionedSampleSize = conditionedInputSample.getSize()
        inputDescription = self.inputSample.getDescription()
        outputDescription = self.outputSample.getDescription()
        marginalOutputDescription = outputDescription[outputIndex]
        grid = ot.GridLayout(1, 1 + dimension_input)
        for i in range(dimension_input):
            # Plot unconditional distribution
            unconditionalDistribution = inputDistribution.getMarginal(i)
            conditionalDistribution = self.densityEstimator.build(
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
        unconditionalDistribution = self.densityEstimator.build(
            self.outputSample[:, outputIndex]
        )
        conditionalDistribution = self.densityEstimator.build(
            conditionedOutputSample[:, outputIndex]
        )
        outputDescription = self.outputSample.getDescription()
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
            f"Unconditioned n={sample_size}, "
            f"Conditioned n = {conditionedSampleSize}"
        )
        return grid

    def plotConditionOutputQuantile(self, quantileLevel, inputDistribution):
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
        - the minimum bound is the quantile of given level of the
        output marginal sample,
        - the maximum bound is the sample maximum of the output marginal sample.
        Then we call plot_event_from_bounds().
        Finally, we gather each marginal plot into a single grid of plots.

        Parameters
        ----------
        quantileLevel: float, in [0, 1]
            The quantile level.
        inputDistribution : ot.Distribution(inputDimension)
            The distribution of the input sample.

        Return
        ------
        grid: ot.GridLayout(1, 1 + inputDimension)
            The grid of sensitivity plots.
            The i-th plot presents the unconditional and conditional
            distribution of the i-th input to the outputIndex-th output.
            The last plot presents the unconditional and conditional
            distribution of the outputIndex-th output.
        """
        outputDimension = self.outputSample.getDimension()
        inputDimension = self.inputSample.getDimension()
        grid = ot.GridLayout(outputDimension, 1 + inputDimension)
        outputDescription = self.outputSample.getDescription()
        for indexOutput in range(outputDimension):
            quantileLowerPoint = self.outputSample.computeQuantilePerComponent(
                quantileLevel
            )
            lowerValue = quantileLowerPoint[indexOutput]
            maxPoint = self.outputSample.getMax()
            upperValue = maxPoint[indexOutput]
            subGrid = self.plotConditionOutputBounds(
                lowerValue,
                upperValue,
                inputDistribution,
                indexOutput,
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
        self,
        inputIndex,
        boundsList,
    ):
        """
        Condition distribution of the output conditionnaly of the input.

        Parameters
        ----------
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
        if self.outputSample.getDimension() != 1:
            raise ValueError(
                f"Output dimension is equal to {self.outputSample.getDimension()}"
                "instead of 1"
            )
        inputDimension = self.inputSample.getDimension()
        if inputIndex < 0 or inputIndex >= inputDimension:
            raise ValueError(
                f"Unknown input marginal index {inputIndex}. "
                f"Must be in the [0, {inputDimension}] interval."
            )
        # Joint the X and Y samples into a single one
        jointXYSample = joinInputOutputSample(self.inputSample, self.outputSample)
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
            kde = self.densityEstimator.build(outputMarginalSample)
            outputDistributionList.append(kde)

        return outputDistributionList

    def plotConditionInputAll(
        self,
        inputIndex,
        numberOfCuts=5,
    ):
        """
        Condition on input with quantile levels and plots of the conditional output.

        Parameters
        ----------
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
        sampleSize = self.inputSample.getSize()
        inputDimension = self.inputSample.getDimension()
        if inputIndex < 0 or inputIndex > inputDimension:
            raise ValueError(
                f"Input marginal index {inputIndex} is not in " f"[0, {inputDimension}]"
            )
        if numberOfCuts < 1:
            raise ValueError(
                f"The number of splits must be larger than 1, but is equal to {numberOfCuts}."
            )
        outputDimension = self.outputSample.getDimension()
        inputDescription = self.inputSample.getDescription()
        outputDescription = self.outputSample.getDescription()
        marginalInputSample = self.inputSample.getMarginal(inputIndex)
        distributionOutputList = (
            RegionalSensitivityAnalysis.ComputeConditionInputQuantileDistributions(
                marginalInputSample,
                self.outputSample,
                self.densityEstimator,
                numberOfCuts,
            )
        )
        grid = ot.GridLayout(outputDimension, numberOfCuts)
        yTitle = "PDF" if self.isDrawPDF else "CDF"
        grid.setTitle(
            f"{yTitle} of Y | {inputDescription[inputIndex]} in [a, b], "
            f"n = {sampleSize}"
        )
        for outputIndex in range(outputDimension):
            distributionInputList = distributionOutputList[outputIndex]
            conditionalDistributionList = distributionInputList[
                0
            ]  # There is only one input considered here
            outputMarginalSample = self.outputSample.getMarginal(outputIndex)
            # Unconditional output distribution
            outputDistribution = self.densityEstimator.build(outputMarginalSample)
            if self.isDrawPDF:
                unconditionalPDFPlot = outputDistribution.drawPDF()
            else:
                unconditionalPDFPlot = outputDistribution.drawCDF()
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
                if self.isDrawPDF:
                    curve = conditionalDistribution.drawPDF()
                else:
                    curve = conditionalDistribution.drawCDF()
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
                        f"{inputDescription[inputIndex]} in "
                        f"[{alphaLevelMin:.2f}, {alphaLevelMax:.2f}]"
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

    def plotConditionInputQuantileSequence(
        self,
        numberOfCuts=5,
        minimumColorValue=0.5,
        maximumColorValue=1.0,
    ):
        """
        Condition on input with sequence of quantile levels and see the conditional output.

        Parameters
        ----------
        numberOfCuts: int, greater than 1
            The number of cuts of the quantile levels.
        minimumValue: float, in [0, 1]
            The minimum value.
        maximumValue: float, in [0, 1]
            The maximum value.
            We must have maximumValue >= minimumValue.

        Return
        ------
        grid: ot.GridLayout(outputDimension, inputDimension)
            The outputIndex-th, inputIndex-th plot presents all the
            conditional distributions of the output when the inputs has
            a conditional distribution in a given interval.
        """
        sampleSize = self.inputSample.getSize()
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
                f"The minimum color value is equal to {minimumColorValue} "
                "(must be >= 0.0) and "
                f"the maximum color value is equal to {maximumColorValue} "
                "(must be <= 1.0), which is inconsistent."
            )
        inputDimension = self.inputSample.getDimension()
        outputDimension = self.outputSample.getDimension()
        inputDescription = self.inputSample.getDescription()
        outputDescription = self.outputSample.getDescription()
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
        distributionOutputList = (
            RegionalSensitivityAnalysis.ComputeConditionInputQuantileDistributions(
                self.inputSample, self.outputSample, self.densityEstimator, numberOfCuts
            )
        )
        grid = ot.GridLayout(outputDimension, inputDimension)
        yTitle = "PDF" if self.isDrawPDF else "CDF"
        grid.setTitle(
            f"{yTitle} of Y | X, n = {sampleSize}, number of quantiles : {numberOfCuts}"
        )
        for outputIndex in range(outputDimension):
            distributionInputList = distributionOutputList[outputIndex]
            outputMarginalSample = self.outputSample.getMarginal(outputIndex)
            # Unconditional output distribution
            outputDistribution = self.densityEstimator.build(outputMarginalSample)
            if self.isDrawPDF:
                unconditionalPDFPlot = outputDistribution.drawPDF()
            else:
                unconditionalPDFPlot = outputDistribution.drawCDF()
            unconditionalPDFCurve = unconditionalPDFPlot.getDrawable(0)
            unconditionalPDFCurve.setLineStyle("dashed")
            unconditionalPDFCurve.setColor(unconditionalColor)
            unconditionalPDFCurve.setLegend("Unconditional")
            # Compute common bounding box
            unconditionalPDFCurveBoundingBox = unconditionalPDFPlot.getBoundingBox()
            unconditionalPDFLowerBound = (
                unconditionalPDFCurveBoundingBox.getLowerBound()
            )
            unconditionalPDFUpperBound = (
                unconditionalPDFCurveBoundingBox.getUpperBound()
            )
            ymin = unconditionalPDFUpperBound[0]
            ymax = unconditionalPDFUpperBound[1]
            for inputIndex in range(inputDimension):
                conditionalDistributionList = distributionInputList[inputIndex]
                for i in range(numberOfCuts):
                    conditionalDistribution = conditionalDistributionList[i]
                    if self.isDrawPDF:
                        curve = conditionalDistribution.drawPDF()
                    else:
                        curve = conditionalDistribution.drawCDF()
                    curveMaximumDensity = curve.getBoundingBox().getUpperBound()[1]
                    ymax = max(ymax, curveMaximumDensity)

            # Set common interval
            interval = ot.Interval(unconditionalPDFLowerBound, [ymin, ymax])
            for inputIndex in range(inputDimension):
                conditionalDistributionList = distributionInputList[inputIndex]
                yTitle = "PDF" if self.isDrawPDF else "CDF"
                graph = ot.Graph(
                    "",
                    f"{outputDescription[outputIndex]} | {inputDescription[inputIndex]}",
                    yTitle,
                    True,
                )
                if inputIndex > 0:
                    graph.setYTitle("")
                graph.add(unconditionalPDFCurve)
                for i in range(numberOfCuts):
                    conditionalDistribution = conditionalDistributionList[i]
                    if self.isDrawPDF:
                        curve = conditionalDistribution.drawPDF().getDrawable(0)
                    else:
                        curve = conditionalDistribution.drawCDF().getDrawable(0)
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

    def plotConditionOutputAll(
        self,
        inputDistribution,
        numberOfCuts=5,
        outputIndex=0,
    ):
        """
        Condition on output with sequence of quantile levels and plot conditional input.

        See plotConditionInputAll() for the same procedure when
        we condition on input.

        Parameters
        ----------
        inputDistribution : ot.Distribution(inputDimension)
            The distribution of the input sample.
        numberOfCuts: int, greater than 1
            The number of cuts of the quantile levels.
        outputIndex: int
            The index of an output.
            The default value is outputIndex=0.

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
        sampleSize = self.inputSample.getSize()
        outputDimension = self.outputSample.getDimension()
        if outputIndex < 0 or outputIndex > outputDimension:
            raise ValueError(
                f"Output marginal index {outputIndex} is not in "
                f"[0, {outputDimension}]"
            )
        if numberOfCuts < 1:
            raise ValueError(
                f"The number of splits must be larger than 1, but is equal to {numberOfCuts}."
            )
        inputDimension = self.inputSample.getDimension()
        inputDescription = self.inputSample.getDescription()
        outputDescription = self.outputSample.getDescription()
        marginalOutputSample = self.outputSample.getMarginal(outputIndex)
        distributionInputList = (
            RegionalSensitivityAnalysis.ComputeConditionOutputQuantileDistributions(
                self.inputSample,
                marginalOutputSample,
                self.densityEstimator,
                numberOfCuts,
            )
        )
        grid = ot.GridLayout(inputDimension, numberOfCuts)
        yTitle = "PDF" if self.isDrawPDF else "CDF"
        grid.setTitle(
            f"{yTitle} of X | {outputDescription[outputIndex]}, " f"n = {sampleSize}"
        )
        for inputIndex in range(inputDimension):
            distributionOutputList = distributionInputList[inputIndex]
            conditionalDistributionList = distributionOutputList[
                0
            ]  # There is only one output considered here
            marginalInputDistribution = inputDistribution.getMarginal(inputIndex)
            if self.isDrawPDF:
                unconditionalPDFPlot = marginalInputDistribution.drawPDF()
            else:
                unconditionalPDFPlot = marginalInputDistribution.drawCDF()
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
                if self.isDrawPDF:
                    curve = conditionalDistribution.drawPDF()
                else:
                    curve = conditionalDistribution.drawCDF()
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
                if self.isDrawPDF:
                    curve = conditionalDistribution.drawPDF().getDrawable(0)
                else:
                    curve = conditionalDistribution.drawCDF().getDrawable(0)
                curve.setLegend("Conditional")
                graph.add(curve)
                #
                if inputIndex == 0:
                    graph.setTitle(
                        f"{outputDescription[outputIndex]} in "
                        f"[{alphaLevelMin:.2f}, {alphaLevelMax:.2f}]"
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

    def plotConditionOutputQuantileSequence(
        self,
        inputDistribution,
        numberOfCuts=5,
        minimumColorValue=0.5,
        maximumColorValue=1.0,
    ):
        """
        Condition on output with sequence of quantile levels and see the conditional intput.

        Parameters
        ----------
        inputDistribution : ot.Distribution(inputDimension)
            The distribution of the input sample.
        numberOfCuts: int, greater than 1
            The number of cuts of the quantile levels.
            The default value is numberOfCuts=5.
        minimumValue: float, in [0, 1]
            The minimum value.
        maximumValue: float, in [0, 1]
            The maximum value.
            We must have maximumValue >= minimumValue.

        Return
        ------
        grid: ot.GridLayout(outputDimension, inputDimension)
            The outputIndex-th, inputIndex-th plot presents all the
            conditional distributions of the output when the inputs has
            a conditional distribution in a given interval.
        """
        sampleSize = self.inputSample.getSize()
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
                f"The minimum color value is equal to {minimumColorValue} "
                "(must be >= 0.0) and "
                f"the maximum color value is equal to {maximumColorValue} "
                "(must be <= 1.0), which is inconsistent."
            )
        inputDimension = self.inputSample.getDimension()
        outputDimension = self.outputSample.getDimension()
        inputDescription = self.inputSample.getDescription()
        outputDescription = self.outputSample.getDescription()
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
        distributionInputList = (
            RegionalSensitivityAnalysis.ComputeConditionOutputQuantileDistributions(
                self.inputSample, self.outputSample, self.densityEstimator, numberOfCuts
            )
        )
        grid = ot.GridLayout(outputDimension, inputDimension)
        yTitle = "PDF" if self.isDrawPDF else "CDF"
        grid.setTitle(
            f"{yTitle} of X | Y, n = {sampleSize}, number of quantiles : {numberOfCuts}"
        )
        for inputIndex in range(inputDimension):
            distributionOutputList = distributionInputList[inputIndex]
            marginalInputDistribution = inputDistribution.getMarginal(inputIndex)
            if self.isDrawPDF:
                unconditionalPDFPlot = marginalInputDistribution.drawPDF()
            else:
                unconditionalPDFPlot = marginalInputDistribution.drawCDF()
            unconditionalPDFCurve = unconditionalPDFPlot.getDrawable(0)
            unconditionalPDFCurve.setLineStyle("dashed")
            unconditionalPDFCurve.setColor(unconditionalColor)
            unconditionalPDFCurve.setLegend("Unconditional")
            # Compute common bounding box
            unconditionalPDFCurveBoundingBox = unconditionalPDFPlot.getBoundingBox()
            unconditionalPDFLowerBound = (
                unconditionalPDFCurveBoundingBox.getLowerBound()
            )
            unconditionalPDFUpperBound = (
                unconditionalPDFCurveBoundingBox.getUpperBound()
            )
            ymin = unconditionalPDFUpperBound[0]
            ymax = unconditionalPDFUpperBound[1]
            for outputIndex in range(outputDimension):
                conditionalDistributionList = distributionOutputList[outputIndex]
                for i in range(numberOfCuts):
                    conditionalDistribution = conditionalDistributionList[i]
                    if self.isDrawPDF:
                        curve = conditionalDistribution.drawPDF()
                    else:
                        curve = conditionalDistribution.drawCDF()
                    curveMaximumDensity = curve.getBoundingBox().getUpperBound()[1]
                    ymax = max(ymax, curveMaximumDensity)

            # Set common interval
            interval = ot.Interval(unconditionalPDFLowerBound, [ymin, ymax])
            for outputIndex in range(outputDimension):
                conditionalDistributionList = distributionOutputList[outputIndex]
                yTitle = "PDF" if self.isDrawPDF else "CDF"
                graph = ot.Graph(
                    "",
                    f"{inputDescription[inputIndex]} | {outputDescription[outputIndex]}",
                    yTitle,
                    True,
                )
                if inputIndex > 0:
                    graph.setYTitle("")
                graph.add(unconditionalPDFCurve)
                for i in range(numberOfCuts):
                    conditionalDistribution = conditionalDistributionList[i]
                    if self.isDrawPDF:
                        curve = conditionalDistribution.drawPDF().getDrawable(0)
                    else:
                        curve = conditionalDistribution.drawCDF().getDrawable(0)
                    curve.setLegend(
                        f"{outputDescription[outputIndex]} in "
                        f"[{alphaLevels[i]:.1f}, {alphaLevels[1 + i]:.1f}]"
                    )
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
