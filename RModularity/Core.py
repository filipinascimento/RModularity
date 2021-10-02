

import louvain
import igraph as ig
import graph_tool as gt
import graph_tool.inference as gtInference
import numpy as np
from tqdm.auto import tqdm, trange
from collections import Counter
import multiprocessing as mp


def SBMMinimizeMembership(vertexCount, edges, directed=False, degreeCorrected=True):
    g = gt.Graph(directed=directed)
    for _ in range(0, vertexCount):
        g.add_vertex()
    for edge in edges:
        g.add_edge(edge[0], edge[1])
    state = gtInference.minimize.minimize_blockmodel_dl(
        g, state_args={"deg_corr": degreeCorrected})
    DLDetected = state.entropy()
    DLTrivial = gtInference.blockmodel.BlockState(
        g, B=1, deg_corr=degreeCorrected).entropy()
    return (list(state.get_blocks()), DLDetected, DLTrivial)


def rewireNetwork(nodeCount, edges, probability):
    edgesCount = len(edges)
    newEdges = np.array(edges)
    mask = np.random.random(edgesCount) < probability
    selectedEdgesIndices = np.where(mask)[0]
    generatedSelectedEdges = np.array(np.random.randint(
        0, nodeCount, (len(selectedEdgesIndices), 2)))
    newEdges[selectedEdgesIndices] = generatedSelectedEdges
    newEdges = [(fromIndex, toIndex) for fromIndex, toIndex in newEdges]
    return newEdges


def getMajorConnectedComponent(nodeCount, edges, directed=False):
    g = ig.Graph(nodeCount, edges, directed=directed)
    giant = g.components(mode="weak").giant()
    return (giant.vcount(), giant.get_edgelist())


def calculatePerturbedTrivialCount(args):
    trivialCount = 0
    nodeCount, edges, directed, probability, \
        useMajorConnectedComponent, detectionTrials = args
    newEdges = rewireNetwork(nodeCount, edges, probability)
    newNodeCount = nodeCount
    allDLDetected = []
    allDLTrivial = []
    if(useMajorConnectedComponent):
        (newNodeCount, newEdges) = getMajorConnectedComponent(
            nodeCount, newEdges, directed)
    for detectionIndex in range(0, detectionTrials):
        communities, DLDetected, DLTrivial = SBMMinimizeMembership(
            newNodeCount, newEdges, directed=directed)
        if(len(set(communities)) == 1):
            trivialCount += 1
        allDLDetected.append(DLDetected)
        allDLTrivial.append(DLTrivial)
    return trivialCount, allDLDetected, allDLTrivial


def RModularity(
    nodeCount,
    edges,
    directed=False,
    perturbationCount=25,
    detectionTrials=1,
    rewireResolution=51,
    useMajorConnectedComponent=True,
    outputCurves=False,
    showProgress=True,
    useMultiprocessing=True
):
    """
    Computes the Robustness Modularity of a networm.

    Parameters
    ----------
    nodeCount : int
        The number of nodes in the network.
    edges : list of tuples
        A list of the edges in the network.
    directed : int, optional
        Whether the network is directed or not.
    perturbationCount : int, optional
        The number of perturbations to perform.
        (defaults to 25)
    detectionTrials : int, optional
        The number of times to perform community
        detection for each perturbed network.
        (defaults to 1)
    useMajorConnectedComponent : bool, optional
        TPR is determined only in terms of the major
        connected component of the network
        (defaults to True)
    rewireResolution : int, optional
        The number values points for the rewire
        probabilities (from 0 to 1) to calculate
        the Trivial Partition Ratio (TPR) curves
        and Robustness Modularity.
        (defaults to 51)
    outputCurves : bool, optional
        Whether to save the TPR and MDL curves.
        (defaults to False)
    showProgress : bool, optional
        Shows a progress bar if enabled.
        (defaults to True)
    useMultiprocessing: bool, optional
        Uses parallelProcessing to calculate
        Rmodularity
        (defaults to False)
    Returns
    -------
    float 
        The RModularity of the network.
    (float, np.array dim=1, np.array dim=1, np.array dim=2, np.array dim=2) if outputCurves is True
        The tuple (RModularity, probabilities, TPR curves, MDL Detected, MDL Trivial).
    """
    TPRCurve = np.zeros(rewireResolution)
    MDLCurvesDetected = np.zeros(
        (rewireResolution, detectionTrials*perturbationCount))
    MDLCurvesTrivial = np.zeros(
        (rewireResolution, detectionTrials*perturbationCount))
    probabilities = np.linspace(0, 1, rewireResolution)


    if(useMajorConnectedComponent):
        (nodeCount, edges) = getMajorConnectedComponent(
            nodeCount, edges, directed)
    if(showProgress):
        probabilitiesIterator = tqdm(probabilities, desc="Current p")
    else:
        probabilitiesIterator = probabilities
    if(useMultiprocessing):
        num_processors = mp.cpu_count()
        # Disabling internal multithreading of graph_tool
        gt.openmp_set_num_threads(1)
    
    for probabilityIndex, probability in enumerate(probabilitiesIterator):
        trivialCount = 0
        if(useMultiprocessing):
            allArgs = [(nodeCount, edges, directed, probability,
                        useMajorConnectedComponent, detectionTrials)]*perturbationCount
            perturbationIndex = 0

            pool = mp.Pool(processes=num_processors)
            for newTrivialCount, allDLDetected, allDLTrivial in tqdm(pool.imap_unordered(func=calculatePerturbedTrivialCount, iterable=allArgs), total=len(allArgs), desc="Perturbation", leave=False):
                trivialCount += newTrivialCount
                MDLCurvesTrivial[probabilityIndex, perturbationIndex *
                                detectionTrials:(perturbationIndex+1)*detectionTrials] = allDLTrivial
                MDLCurvesDetected[probabilityIndex, perturbationIndex *
                                detectionTrials:(perturbationIndex+1)*detectionTrials] = allDLDetected
                perturbationIndex += 1
            pool.terminate()
            pool.close()
            TPRCurve[probabilityIndex] = trivialCount / \
                (perturbationCount*detectionTrials)
        else:
            for perturbationIndex in trange(0,perturbationCount, desc="Perturbation", leave=False):
                args = (nodeCount, edges, directed, probability, useMajorConnectedComponent, detectionTrials)
                newTrivialCount, allDLDetected,allDLTrivial  = calculatePerturbedTrivialCount(args)
                trivialCount += newTrivialCount
                MDLCurvesTrivial[probabilityIndex, perturbationIndex *
                                detectionTrials:(perturbationIndex+1)*detectionTrials] = allDLTrivial
                MDLCurvesDetected[probabilityIndex, perturbationIndex *
                                detectionTrials:(perturbationIndex+1)*detectionTrials] = allDLDetected
                perturbationIndex += 1
            TPRCurve[probabilityIndex] = trivialCount / \
                (perturbationCount*detectionTrials)

    RModularity = 1.0-np.trapz(TPRCurve, probabilities)

    if(outputCurves):
        return (RModularity, probabilities, TPRCurve, MDLCurvesTrivial, MDLCurvesDetected)
    else:
        return RModularity
