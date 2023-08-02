import unittest
import numpy as np
import otsensitivity as ots
import openturns as ot
from openturns.usecases import flood_model
import openturns.viewer as otv
from matplotlib import pylab as plt


def getFloodingInputDistribution():
    dist_Q = ot.Gumbel(558.0, 1013.0)
    dist_Q = ot.TruncatedDistribution(dist_Q, 0)
    dist_Q.setDescription(["Q"])
    dist_Ks = ot.Normal(30, 7.5)
    dist_Ks = ot.TruncatedDistribution(dist_Ks, 0)
    dist_Ks.setDescription(["Ks"])
    dist_Zv = ot.Uniform(49.0, 51.0)
    dist_Zv.setDescription(["Zv"])
    dist_Zm = ot.Uniform(54.0, 56.0)
    dist_Zm.setDescription(["Zm"])
    inputDistribution = ot.ComposedDistribution([dist_Q, dist_Ks, dist_Zv, dist_Zm])
    return inputDistribution


def getFloodingSample(sampleSize):
    # Workaround
    physicalModel = ot.SymbolicFunction(
        ["Q", "Ks", "Zv", "Zm"],
        ["S"],
        "S := (Q / (Ks * 300.0 * sqrt((Zm - Zv) / 5000.0)))^(3.0 / 5.0) + Zv - 58.5",
    )
    inputDistribution = getFloodingInputDistribution()
    print(physicalModel)
    print(inputDistribution)
    inputSample = inputDistribution.getSample(sampleSize)
    outputSample = physicalModel(inputSample)
    return inputSample, outputSample


class CheckEvent(unittest.TestCase):
    def test_flooding_known_input_distribution(self):
        ot.Log.Show(ot.Log.NONE)

        sampleSize = 10000

        inputSample, outputSample = getFloodingSample(sampleSize)

        verbose = True

        print("+ Distribution used for the input variables")
        inputDistribution = getFloodingInputDistribution()

        # Fait l'analyse
        quantile_level = 0.8
        indexOutput = 0
        quantileLowerPoint = outputSample.computeQuantilePerComponent(quantile_level)
        lowerValue = quantileLowerPoint[indexOutput]
        maxPoint = outputSample.getMax()
        upperValue = maxPoint[indexOutput]
        grid = ots.plot_event(
            inputSample,
            outputSample,
            indexOutput,
            lowerValue,
            upperValue,
            inputDistribution,
            verbose,
        )
        title = grid.getTitle()
        grid.setTitle(f"Quantile at level {quantile_level}, {title}")
        view = otv.View(grid, figure_kw={"figsize": (8.0, 3.0)})
        plt.subplots_adjust(wspace=0.5, top=0.8, bottom=0.2)
        view.save("../doc/images/event_known_input.png")

    def test_flooding_given_data(self):
        ot.Log.Show(ot.Log.NONE)

        sampleSize = 10000

        inputSample, outputSample = getFloodingSample(sampleSize)

        verbose = True

        print("+ Distribution used for the input variables")
        # Si la loi est inconnue, on peut l'estimer par lissage à noyau
        smoothing = ot.KernelSmoothing()
        smoothing.setAutomaticLowerBound(True)
        smoothing.setAutomaticUpperBound(True)
        inputDistribution = smoothing.build(inputSample)

        # Fait l'analyse
        quantile_level = 0.8
        indexOutput = 0
        quantileLowerPoint = outputSample.computeQuantilePerComponent(quantile_level)
        lowerValue = quantileLowerPoint[indexOutput]
        maxPoint = outputSample.getMax()
        upperValue = maxPoint[indexOutput]
        grid = ots.plot_event(
            inputSample,
            outputSample,
            indexOutput,
            lowerValue,
            upperValue,
            inputDistribution,
            verbose,
        )
        title = grid.getTitle()
        grid.setTitle(f"Quantile at level {quantile_level}, {title}")
        view = otv.View(grid, figure_kw={"figsize": (8.0, 3.0)})
        plt.subplots_adjust(wspace=0.5, top=0.8, bottom=0.2)
        view.save("../doc/images/event_given_data.png")

    def test_flooding_given_data_bounded(self):
        ot.Log.Show(ot.Log.NONE)

        sampleSize = 10000

        inputSample, outputSample = getFloodingSample(sampleSize)

        verbose = True

        print("+ Distribution used for the input variables")
        # Si la loi est inconnue, on peut l'estimer par lissage à noyau
        # De plus, on applique une correction aux bords et on fait l'hypothèse
        # d'indépendance
        dimension = inputSample.getDimension()
        smoothing = ot.KernelSmoothing()
        smoothing.setBoundingOption(ot.KernelSmoothing.BOTH)
        smoothing.setBoundaryCorrection(True)
        list_of_marginals = []
        for i in range(dimension):
            marginal_sample = inputSample[:, i]
            marginal_distribution = smoothing.build(marginal_sample)
            list_of_marginals.append(marginal_distribution)
        inputDistribution = ot.ComposedDistribution(list_of_marginals)

        # Fait l'analyse
        quantile_level = 0.8
        indexOutput = 0
        quantileLowerPoint = outputSample.computeQuantilePerComponent(quantile_level)
        lowerValue = quantileLowerPoint[indexOutput]
        maxPoint = outputSample.getMax()
        upperValue = maxPoint[indexOutput]
        grid = ots.plot_event(
            inputSample,
            outputSample,
            indexOutput,
            lowerValue,
            upperValue,
            inputDistribution,
            verbose,
        )
        title = grid.getTitle()
        grid.setTitle(f"Quantile at level {quantile_level}, {title}")
        view = otv.View(grid, figure_kw={"figsize": (8.0, 3.0)})
        plt.subplots_adjust(wspace=0.5, top=0.8, bottom=0.2)
        view.save("../doc/images/event_given_data_with_bounds.png")


if __name__ == "__main__":
    import matplotlib

    # matplotlib.use("Agg")
    unittest.main()
