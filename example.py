import RModularity
import igraph as ig
from pathlib import Path
import pickle as pkl

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
# This is needed to run the script using multiprocessing
if __name__ == '__main__':
    # Use fast method or not
    useFast = True

    networkName = "road-euroroad"
    # networkName = "LFR_mu0.1"
    # networkName = "LFR_mu1.0"

    outputSuffix = ""
    figurePath = Path("Figures")
    figurePath.mkdir(parents=True, exist_ok=True)

    g = ig.Graph.Read_GML(str(Path("SampleNetworks")/("%s.gml" % networkName)))

    Q_diff = RModularity.modularityDifference(
        g.vcount(),
        g.get_edgelist(),
        g.is_directed()
    )

    print("Q_diff = ", Q_diff)

    Q_DL = RModularity.informationModularity(
        g.vcount(),
        g.get_edgelist(),
        g.is_directed()
    )
    print("Q_DL = ", Q_DL)

    if(useFast):
        #calculating R Modularity based on the fast Monte-Carlo method
        Q_rA = RModularity.RModularityFast(
            g.vcount(),
            g.get_edgelist(),
            g.is_directed(),
            )
        print("Q_rA = ", Q_rA)
    else:
        #calculate R Modularity using the complete algorithm and plot
        # TPR curves and DL curves
        Q_r, probabilities, TPRCurve, \
        DLCurvesTrivial, DLCurvesDetected = RModularity.RModularity(
            g.vcount(),
            g.get_edgelist(),
            g.is_directed(),
            outputCurves=True,
            )

        print("Q_r = ", Q_r)

        with open("%s%s.pkl" % (networkName, outputSuffix), "wb") as fd:
            pkl.dump((Q_r, probabilities, TPRCurve,
                    DLCurvesTrivial, DLCurvesDetected), fd)

        # Plotting TPR and DL curves

        with open("%s%s.pkl" % (networkName, outputSuffix), "rb") as fd:
            Q_r, probabilities, TPRCurve, DLCurvesTrivial, DLCurvesDetected = pkl.load(
                fd)

        avgDLCurvesTrivial = np.mean(DLCurvesTrivial, axis=1)
        avgDLCurvesDetected = np.mean(DLCurvesDetected, axis=1)
        stdDLCurvesTrivial = np.std(DLCurvesTrivial, axis=1)
        stdDLCurvesDetected = np.std(DLCurvesDetected, axis=1)

        diffDLCurves = (DLCurvesTrivial-DLCurvesDetected) / DLCurvesTrivial
        avgDiffDLCurves = np.mean(diffDLCurves, axis=1)
        stdDiffDLCurves = np.std(diffDLCurves, axis=1)

        # TPR Curve
        fig = plt.figure(figsize=(3*1.61803398875, 3))
        ax = plt.axes((0.2, 0.2, 0.70, 0.70), facecolor='w')
        nodeCount = g.vcount()
        averageDegree = np.mean(g.degree())
        TPRArea = Q_r
        trivialRatios = TPRCurve
        ax.plot(probabilities, trivialRatios, color="#262626", lw=2.0)
        ax.fill_between(probabilities, trivialRatios, 1, color="#E8EAEA")
        ax.set_xlabel("$p$")
        ax.set_ylabel("TPR")
        ax.set_title(networkName)
        ax.set_xlim(-0.00, 1.02)
        ax.set_ylim(-0.020, 1.020)
        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                                        lw=0, alpha=0)] * 3
        labels = []
        labels.append("$N$ = %d" % nodeCount)
        labels.append("$\\langle k\\rangle$ = %.2f" % averageDegree)
        labels.append("$Q_{r}$ = %.2f" % TPRArea)

        ax.legend(handles, labels, loc='best',
                fancybox=False, framealpha=0,
                handlelength=0, handletextpad=0)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for axis in ['bottom', 'left']:
            ax.spines[axis].set_linewidth(1.5)
        ax.tick_params(width=1.5)
        fig.savefig(figurePath/("TPR_%s%s.pdf" % (networkName, outputSuffix)))
        plt.close(fig)

        # DL Curve
        fig = plt.figure(figsize=(3*1.61803398875, 3))
        ax = plt.axes((0.2, 0.2, 0.70, 0.70), facecolor='w')
        ax.plot(probabilities, avgDLCurvesDetected, lw=2.0, label="Detected")
        ax.fill_between(probabilities, avgDLCurvesDetected-stdDLCurvesDetected,
                        avgDLCurvesDetected+stdDLCurvesDetected, alpha=0.2)

        ax.plot(probabilities, avgDLCurvesTrivial, lw=2.0, label="Trivial")
        ax.fill_between(probabilities, avgDLCurvesTrivial-stdDLCurvesTrivial,
                        avgDLCurvesTrivial+stdDLCurvesTrivial, alpha=0.2)

        ax.set_xlabel("$p$")
        ax.set_ylabel("DL")
        ax.set_title(networkName)
        ax.set_xlim(-0.00, 1.02)

        ax.legend()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for axis in ['bottom', 'left']:
            ax.spines[axis].set_linewidth(1.5)
        ax.tick_params(width=1.5)
        fig.savefig(figurePath/("DL_%s%s.pdf" % (networkName, outputSuffix)))
        plt.close(fig)

        # Plotting DL Diff distribution
        fig = plt.figure(figsize=(3*1.61803398875, 3))
        ax = plt.axes((0.2, 0.2, 0.70, 0.70), facecolor='w')
        ax.plot(probabilities, avgDiffDLCurves, lw=2.0, label="Detected")
        ax.fill_between(probabilities, avgDiffDLCurves-stdDiffDLCurves,
                        avgDiffDLCurves+stdDiffDLCurves, alpha=0.2)

        ax.set_xlabel("$p$")
        ax.set_ylabel("$Q_{\mathrm{DL}}$")
        ax.set_title(networkName)
        ax.set_xlim(-0.00, 1.02)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for axis in ['bottom', 'left']:
            ax.spines[axis].set_linewidth(1.5)
        ax.tick_params(width=1.5)
        fig.savefig(figurePath/("DLDiff_%s%s.pdf" % (networkName, outputSuffix)))
        plt.close(fig)
