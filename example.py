import RModularity
import xnetwork as xn
import multiprocessing as mp
#mpl_patches
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import numpy as np
import pickle as pkl

networkName = "bench_n5000_t2.000_T2.000_mu1.000_minc20_maxc500_k20.000_maxk250-r0"
addedSuffix = "_several"
g = xn.xnet2igraph("%s.xnet"%networkName)

if __name__ == '__main__':
    Qr,probabilities, TPRCurve, MDLCurvesTrivial, MDLCurvesDetected = RModularity.RModularity(g.vcount(), g.get_edgelist(), g.is_directed(),
            outputCurves=True,
            perturbationCount=75
            )

    print(Qr)

    with open("%s%s.pkl"%(networkName,addedSuffix), "wb") as fd:
        pkl.dump((Qr,probabilities, TPRCurve, MDLCurvesTrivial, MDLCurvesDetected), fd)


    # Plotting
    with open("%s%s.pkl"%(networkName,addedSuffix), "rb") as fd:
        Qr,probabilities, TPRCurve, MDLCurvesTrivial, MDLCurvesDetected = pkl.load(fd)

    avgMDLCurvesTrivial = np.mean(MDLCurvesTrivial, axis=1)
    avgMDLCurvesDetected = np.mean(MDLCurvesDetected, axis=1)
    stdMDLCurvesTrivial = np.std(MDLCurvesTrivial, axis=1)
    stdMDLCurvesDetected = np.std(MDLCurvesDetected, axis=1)

    diffMDLCurves = (MDLCurvesTrivial-MDLCurvesDetected) / MDLCurvesTrivial
    avgDiffMDLCurves = np.mean(diffMDLCurves, axis=1)
    stdDiffMDLCurves = np.std(diffMDLCurves, axis=1)

    fig = plt.figure(figsize=(3*1.61803398875,3))
    ax = plt.axes((0.2, 0.2, 0.70, 0.70), facecolor='w')
    nodeCount = g.vcount()
    averageDegree = np.mean(g.degree())
    TPRArea = Qr
    trivialRatios = TPRCurve
    ax.plot(probabilities,trivialRatios,color = "#262626",lw=2.0)
    ax.fill_between(probabilities,trivialRatios,1,color = "#E8EAEA")
    ax.set_xlabel("$p$")
    ax.set_ylabel("TPR")
    ax.set_title(networkName)
    ax.set_xlim(-0.00,1.02)
    ax.set_ylim(-0.020,1.020)

    # create a list with two empty handles (or more if needed)
    handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", 
                                    lw=0, alpha=0)] * 3

    # create the corresponding number of labels (= the text you want to display)
    labels = []
    labels.append("$N$ = %d"%nodeCount)
    labels.append("$\\langle k\\rangle$ = %.2f"%averageDegree)
    labels.append("$Q_{r}$ = %.2f"%TPRArea)

    # create the legend, supressing the blank space of the empty line symbol and the
    # padding between symbol and label by setting handlelenght and handletextpad
    ax.legend(handles, labels, loc='best', 
            fancybox=False, framealpha=0, 
            handlelength=0, handletextpad=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(1.5)
    ax.tick_params(width=1.5)
    fig.savefig("Figures/TPR_%s%s.pdf"%(networkName,addedSuffix))
    plt.close(fig)


    fig = plt.figure(figsize=(3*1.61803398875,3))
    ax = plt.axes((0.2, 0.2, 0.70, 0.70), facecolor='w')
    ax.plot(probabilities,avgMDLCurvesDetected,lw=2.0,label="Detected")
    ax.fill_between(probabilities,avgMDLCurvesDetected-stdMDLCurvesDetected,avgMDLCurvesDetected+stdMDLCurvesDetected,alpha=0.2)

    ax.plot(probabilities,avgMDLCurvesTrivial,lw=2.0,label="Trivial")
    ax.fill_between(probabilities,avgMDLCurvesTrivial-stdMDLCurvesTrivial,avgMDLCurvesTrivial+stdMDLCurvesTrivial,alpha=0.2)

    ax.set_xlabel("$p$")
    ax.set_ylabel("MDL")
    ax.set_title(networkName)
    ax.set_xlim(-0.00,1.02)

    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(1.5)
    ax.tick_params(width=1.5)
    fig.savefig("Figures/MDL_%s%s.pdf"%(networkName,addedSuffix))
    plt.close(fig)

    fig = plt.figure(figsize=(3*1.61803398875,3))
    ax = plt.axes((0.2, 0.2, 0.70, 0.70), facecolor='w')
    ax.plot(probabilities,avgDiffMDLCurves,lw=2.0,label="Detected")
    ax.fill_between(probabilities,avgDiffMDLCurves-stdDiffMDLCurves,avgDiffMDLCurves+stdDiffMDLCurves,alpha=0.2)

    ax.set_xlabel("$p$")
    ax.set_ylabel("$(\mathrm{MDL}_\mathrm{trivial}-\mathrm{MDL}_\mathrm{detected})/ \mathrm{MDL}_\mathrm{trivial}$")
    ax.set_title(networkName)
    ax.set_xlim(-0.00,1.02)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(1.5)
    ax.tick_params(width=1.5)
    fig.savefig("Figures/MDLDiff_%s%s.pdf"%(networkName,addedSuffix))
    plt.close(fig)

