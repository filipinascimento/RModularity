

import louvain
import igraph as ig
from graph_tool import openmp_set_num_threads as gtOpenmp_set_num_threads
from graph_tool import Graph as gtGraph
import graph_tool.inference as gtInference
import numpy as np
from tqdm.auto import tqdm, trange
from collections import Counter
import multiprocessing as mp
import louvain
import os
import random


def LouvainModularity(aNetwork):
    partition = louvain.find_partition(
        aNetwork, louvain.ModularityVertexPartition)
    return partition.quality()


def SBMMinimizeMembership(vertexCount, edges, directed=False, degreeCorrected=True):
    g = gtGraph(directed=directed)
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


def calculateMaxModularity(g, trials=100):
    maxModularity = -1
    for _ in range(trials):
        modularity = LouvainModularity(g)
        if(modularity > maxModularity):
            maxModularity = modularity
    return maxModularity


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
    g = ig.Graph(nodeCount, edges, directed=directed).simplify()
    giant = g.components(mode="weak").giant()
    return (giant.vcount(), giant.get_edgelist())


def calculatePerturbedTrivialCount(args):
    # Reseeding
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    trivialCount = 0
    nodeCount, edges, directed, probability, \
        detectionTrials = args
    newEdges = rewireNetwork(nodeCount, edges, probability)
    newNodeCount = nodeCount
    allDLDetected = []
    allDLTrivial = []
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
    perturbationCount=24,
    detectionTrials=1,
    rewireResolution=51,
    outputCurves=False,
    showProgress=True,
    useMultiprocessing=True
):
    """
    Computes the Robustness Modularity of a network.

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
        (defaults to 24)
    detectionTrials : int, optional
        The number of times to perform community
        detection for each perturbed network.
        (defaults to 1)
    rewireResolution : int, optional
        The number values points for the rewire
        probabilities (from 0 to 1) to calculate
        the Trivial Partition Ratio (TPR) curves
        and Robustness Modularity.
        (defaults to 51)
    outputCurves : bool, optional
        Whether to save the TPR and DL curves.
        (defaults to False)
    showProgress : bool, optional
        Shows a progress bar if enabled.
        (defaults to True)
    useMultiprocessing: bool, optional
        Uses parallel processing to calculate
        Rmodularity.
        (defaults to True)
    Returns
    -------
    float 
        The RModularity of the network.
    (float, np.array dim=1, np.array dim=1, np.array dim=2, np.array dim=2) if outputCurves is True
        The tuple (RModularity, probabilities, TPR curves, DL Detected, DL Trivial) containing
        the Robustness Modularity, the rewire probabilities, the TPR curves, the Description
        lenghts for the detected and trivial partitions.
    """
    TPRCurve = np.zeros(rewireResolution)
    DLCurvesDetected = np.zeros(
        (rewireResolution, detectionTrials*perturbationCount))
    DLCurvesTrivial = np.zeros(
        (rewireResolution, detectionTrials*perturbationCount))
    probabilities = np.linspace(0, 1, rewireResolution)

    (nodeCount, edges) = getMajorConnectedComponent(
        nodeCount, edges, directed)
    if(showProgress):
        probabilitiesIterator = tqdm(probabilities, desc="Current p")
    else:
        probabilitiesIterator = probabilities
    if(useMultiprocessing):
        num_processors = mp.cpu_count()
        # Disabling internal multithreading of graph_tool
        gtOpenmp_set_num_threads(1)

    if(useMultiprocessing):
        pool = mp.Pool(processes=num_processors)
    for probabilityIndex, probability in enumerate(probabilitiesIterator):
        trivialCount = 0
        if(useMultiprocessing):
            allArgs = [(nodeCount, edges, directed, probability,
                        detectionTrials)]*perturbationCount
            perturbationIndex = 0

            for newTrivialCount, allDLDetected, allDLTrivial in tqdm(pool.imap_unordered(func=calculatePerturbedTrivialCount, iterable=allArgs), total=len(allArgs), desc="Perturbation", leave=False):
                trivialCount += newTrivialCount
                DLCurvesTrivial[probabilityIndex, perturbationIndex *
                                detectionTrials:(perturbationIndex+1)*detectionTrials] = allDLTrivial
                DLCurvesDetected[probabilityIndex, perturbationIndex *
                                 detectionTrials:(perturbationIndex+1)*detectionTrials] = allDLDetected
                perturbationIndex += 1

            TPRCurve[probabilityIndex] = trivialCount / \
                (perturbationCount*detectionTrials)
        else:
            for perturbationIndex in trange(0, perturbationCount, desc="Perturbation", leave=False):
                args = (nodeCount, edges, directed,
                        probability, detectionTrials)
                newTrivialCount, allDLDetected, allDLTrivial = calculatePerturbedTrivialCount(
                    args)
                trivialCount += newTrivialCount
                DLCurvesTrivial[probabilityIndex, perturbationIndex *
                                detectionTrials:(perturbationIndex+1)*detectionTrials] = allDLTrivial
                DLCurvesDetected[probabilityIndex, perturbationIndex *
                                 detectionTrials:(perturbationIndex+1)*detectionTrials] = allDLDetected
                perturbationIndex += 1
            TPRCurve[probabilityIndex] = trivialCount / \
                (perturbationCount*detectionTrials)
    if(useMultiprocessing):
        # pool.terminate()
        pool.close()
        pool.join()
    RModularity = 1.0-np.trapz(TPRCurve, probabilities)

    if(outputCurves):
        return (RModularity, probabilities, TPRCurve, DLCurvesTrivial, DLCurvesDetected)
    else:
        return RModularity


def modularityNullmodel(args):
    network, detectionTrials = args
    networkConfig = ig.Graph.Degree_Sequence(
        network.degree()).simplify().components(mode="weak").giant()
    return calculateMaxModularity(networkConfig, trials=detectionTrials)


def modularityDifference(
    nodeCount,
    edges,
    directed=False,
    detectionTrials=100,
    nullmodelCount=100,
    detectionTrialsNullModel=10,
    showProgress=True,
    useMultiprocessing=True
):
    """
        Computes the Modularity Difference of a network.

    Parameters
    ----------
    nodeCount : int
        The number of nodes in the network.
    edges : list of tuples
        A list of the edges in the network.
    directed : int, optional
        Whether the network is directed or not.
    detectionTrials : int, optional
        The number of times to perform community
        detection for each perturbed network.
        (defaults to 10)
    nullmodelCount : int, optional
        The number of times to perform community
        detection using nullmodels.
        (defaults to 200)
    detectionTrialsNullModel : int, optional
        The number of times to perform community
        detection for each nullmodel realization.
        (defaults to 200)
    showProgress : bool, optional
        Shows a progress bar if enabled.
        (defaults to True)
    Returns
    -------
    float 
        The Modularity Difference of the network.
    """
    network = ig.Graph(nodeCount, edges, directed=directed).simplify(
    ).components(mode="weak").giant()
    modularity = calculateMaxModularity(network, trials=detectionTrials)
    if(useMultiprocessing):
        num_processors = mp.cpu_count()
    nullModelModularities = []
    if(useMultiprocessing):
        allArgs = [(network, detectionTrials)]*nullmodelCount
        pool = mp.Pool(processes=num_processors)
        for nullModelModularity in tqdm(pool.imap_unordered(func=modularityNullmodel, iterable=allArgs), total=len(allArgs), desc="NullModel"):
            nullModelModularities.append(nullModelModularity)
        pool.terminate()
        pool.close()
    else:
        nullModelIterator = range(nullmodelCount)
        if(showProgress):
            nullModelIterator = tqdm(nullModelIterator, desc="Nullmodel")
        for _ in nullModelIterator:
            nullModelModularity = modularityNullmodel(
                (network, detectionTrials))
            nullModelModularities.append(nullModelModularity)
    modularityDifference = modularity - np.mean(nullModelModularities)
    return modularityDifference


def informationModularity(
        nodeCount,
        edges,
        directed=False):
    """
        Computes the Information Modularity of a network.

    Parameters
    ----------
    nodeCount : int
        The number of nodes in the network.
    edges : list of tuples
        A list of the edges in the network.
    directed : int, optional
        Whether the network is directed or not.
    Returns
    -------
    float 
        The Information Modularity of the network.
    """
    _, DLDetected, DLTrivial = SBMMinimizeMembership(
        nodeCount, edges, directed)
    return 1-(DLDetected/DLTrivial)



def RModularityFast_alt(
    nodeCount,
    edges,
    directed=False,
    perturbationCount=48,
    detectionTrials=1,
    outputCurves=False,
    showProgress=True,
    useMultiprocessing=True,
    coarseError = 0.02,
    fineError=0.01,
    minSimilarTrials=3,
):
    """
    Alternative implementation of the fast algorithm (currently unsupported)
    Computes the Robustness Modularity of a network.

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
        (defaults to 24)
    detectionTrials : int, optional
        The number of times to perform community
        detection for each perturbed network.
        (defaults to 1)
    outputCurves : bool, optional
        Whether to save the TPR and DL curves.
        (defaults to False)
    showProgress : bool, optional
        Shows a progress bar if enabled.
        (defaults to True)
    useMultiprocessing: bool, optional
        Uses parallel processing to calculate
        Rmodularity.
        (defaults to True)
    Returns
    -------
    float 
        The RModularity of the network.
    (float, np.array dim=1, np.array dim=1, np.array dim=2, np.array dim=2) if outputCurves is True
        The tuple (RModularity, probabilities, TPR curves, DL Detected, DL Trivial) containing
        the Robustness Modularity, the rewire probabilities, the TPR curves, the Description
        lenghts for the detected and trivial partitions.
    """
    TPRCurve = []
    DLCurvesDetected = []
    DLCurvesTrivial = []
    probabilities = []
    sortedOrder = []

    (nodeCount, edges) = getMajorConnectedComponent(
        nodeCount, edges, directed)
    if(useMultiprocessing):
        num_processors = mp.cpu_count()
        # Disabling internal multithreading of graph_tool
        gtOpenmp_set_num_threads(1)
    def calculateTPR(probability):
        trivialCount = 0
        DLCurvesDetectedSingle = np.zeros(detectionTrials*perturbationCount)
        DLCurvesTrivialSingle = np.zeros(detectionTrials*perturbationCount)
        if(useMultiprocessing):
            allArgs = [(nodeCount, edges, directed, probability,
                        detectionTrials)]*perturbationCount
            perturbationIndex = 0

            pool = mp.Pool(processes=num_processors)
            for newTrivialCount, allDLDetected, allDLTrivial in \
                tqdm(
                    pool.imap_unordered(
                        func=calculatePerturbedTrivialCount,
                        iterable=allArgs
                    ), 
                    total=len(allArgs),
                    desc="Perturbation", leave=False
                ):
                DLCurvesDetectedSingle[perturbationIndex*detectionTrials:(perturbationIndex+1)*detectionTrials] = allDLDetected
                DLCurvesTrivialSingle[perturbationIndex*detectionTrials:(perturbationIndex+1)*detectionTrials] = allDLTrivial
                trivialCount += newTrivialCount
                perturbationIndex += 1
            pool.terminate()
            pool.close()
        else:
            for perturbationIndex in trange(0, perturbationCount, desc="Perturbation", leave=False):
                args = (nodeCount, edges, directed,
                        probability, detectionTrials)
                newTrivialCount, allDLDetected, allDLTrivial = calculatePerturbedTrivialCount(
                    args)
                DLCurvesDetectedSingle[perturbationIndex*detectionTrials:(perturbationIndex+1)*detectionTrials] = allDLDetected
                DLCurvesTrivialSingle[perturbationIndex*detectionTrials:(perturbationIndex+1)*detectionTrials] = allDLTrivial
                trivialCount += newTrivialCount
                perturbationIndex += 1
        TPRValue = trivialCount / \
            (perturbationCount*detectionTrials)
        return TPRValue, DLCurvesDetectedSingle, DLCurvesTrivialSingle
    
    similarTrial = 0
    lastRModularity = -1
    currentRModularity = -1
    def addPointProbability(probability,targetError):
        nonlocal similarTrial,lastRModularity,TPRCurve,DLCurvesDetected,DLCurvesTrivial,sortedOrder,probabilities,currentRModularity
        TPRValue, DLCurvesDetectedSingle, DLCurvesTrivialSingle = calculateTPR(probability)
        probabilities.append(probability)
        TPRCurve.append(TPRValue)
        DLCurvesDetected.append(DLCurvesDetectedSingle)
        DLCurvesTrivial.append(DLCurvesTrivialSingle)
        sortedOrder = np.argsort(probabilities)
        probabilities = [ probabilities[i] for i in sortedOrder ]
        TPRCurve = [ TPRCurve[i] for i in sortedOrder ]
        DLCurvesDetected = [ DLCurvesDetected[i] for i in sortedOrder ]
        DLCurvesTrivial = [ DLCurvesTrivial[i] for i in sortedOrder ]
        currentRModularity = 1.0-np.trapz(TPRCurve, probabilities)
        absDiff = abs(currentRModularity-lastRModularity)/lastRModularity
        lastRModularity = currentRModularity
        if(absDiff < targetError):
            similarTrial += 1
        else:
            similarTrial = 0
        return absDiff
    
    pbar = tqdm(total = minSimilarTrials,leave=True)
    pbar.set_description("Calculating for probabilty = 0")
    addPointProbability(0,coarseError)
    pbar.set_description("Calculating for probabilty = 1")
    addPointProbability(1.0,coarseError)
    fineRange = (0,1)
    fineIterations = 0
    while(True):
        currentDeviation = 0.0
        for probIndex in range(len(probabilities)-1):
            probability = probabilities[probIndex]
            nextProbability = probabilities[probIndex+1]
            if(TPRCurve[probIndex]<1.0 and TPRCurve[probIndex+1]==1.0):
                fineRange = (0,nextProbability)
                print("Refine Range",(probability,nextProbability))
                currentDeviation = addPointProbability((probability+nextProbability)*0.5,coarseError)
                break
        pbar.set_description("COARSE Phase. Deviation: %g (target=%g) Trials" % (currentDeviation,coarseError))
        pbar.reset()
        pbar.update(similarTrial)
        
        if(currentDeviation<coarseError):
            break
                

    while(similarTrial<minSimilarTrials):
        probability=random.random()*(fineRange[1]-fineRange[0])+fineRange[0]
        absDiff = addPointProbability(probability,fineError)
        pbar.set_description("FINE Phase. Deviation: %g (target=%g) Trials" % (absDiff,fineError))
        pbar.reset()
        pbar.update(similarTrial)
    pbar.refresh()
    pbar.close()
    if(outputCurves):
        return (currentRModularity, np.array(probabilities), np.array(TPRCurve), np.array(DLCurvesTrivial), np.array(DLCurvesDetected))
    else:
        return currentRModularity


def RModularityFast(
    nodeCount,
    edges,
    directed=False,
    perturbationCount=48,
    detectionTrials=1,
    showProgress=True,
    useMultiprocessing=True,
    useCoarseStep = True,
    fineError=0.01,
    coarseError = 0.02,
    minSimilarTrials=2,
):
    """
    Computes the approximated Robustness Modularity of a network
    using a Monte-Carlo approach. Note that this approach can not
    procude the curves of TPR.

    Parameters
    ----------
    nodeCount : int
        The number of nodes in the network.
    edges : list of tuples
        A list of the edges in the network.
    directed : int, optional
        Whether the network is directed or not.
    perturbationCount : int, optional
        The number of perturbations to perform
        at each step.
        (defaults to 48)
    detectionTrials : int, optional
        The number of times to perform community
        detection for each perturbed network.
        (defaults to 1)
    showProgress : bool, optional
        Shows a progress bar if enabled.
        (defaults to True)
    useMultiprocessing: bool, optional
        Uses parallel processing to calculate
        Rmodularity.
        (defaults to True)
    useCoarseStep: bool, optional
        Finds the plateal region using a binary search before applying
        the Monte-Carlo approach.
        (defaults to True)
    fineError: float, optional
        Error tolerance for the fine step.
        (defaults to 0.01)
    coarseError: float, optional
        Error tolerance for the coarse step.
    minSimilarTrials: int, optional
        The minimum number of similar trials to perform before 
        stopping the Monte-Carlo approach.
        (defaults to 2)
    Returns
    -------
    float 
        The RModularity of the network.
    """
    
    sortedOrder = []

    (nodeCount, edges) = getMajorConnectedComponent(
        nodeCount, edges, directed)
    if(useMultiprocessing):
        num_processors = mp.cpu_count()
        # Disabling internal multithreading of graph_tool
        gtOpenmp_set_num_threads(1)
        pool = mp.Pool(processes=num_processors)
    def calculateTPR(probabilities):
        trivialCount = 0
        #check if probabilities is a number
        if(isinstance(probabilities,float) or isinstance(probabilities,int)):
            probabilities = [probabilities]*perturbationCount

        if(useMultiprocessing):
            allArgs = [(nodeCount, edges, directed, probability,
                        detectionTrials) for index, probability in enumerate(probabilities)]
            perturbationIndex = 0

            poolIterator = pool.imap_unordered(
                                func=calculatePerturbedTrivialCount,
                                iterable=allArgs
                            )
            if(showProgress):
                poolIterator = tqdm(poolIterator, 
                    total=len(allArgs),
                    desc="Perturbation", leave=False
                )
            for newTrivialCount, allDLDetected, allDLTrivial in poolIterator:
                trivialCount += newTrivialCount
                perturbationIndex += 1
            # pool.terminate()
            # pool.close()
        else:
            for perturbationIndex,probability in enumerate(trange(0, probabilities, desc="Perturbation", leave=False)):
                args = (nodeCount, edges, directed,
                        probability, detectionTrials)
                newTrivialCount, allDLDetected, allDLTrivial = calculatePerturbedTrivialCount(
                    args)
                trivialCount += newTrivialCount
                perturbationIndex += 1
        TPRValue = trivialCount / \
            (perturbationCount*detectionTrials)
        return TPRValue
    
    similarTrial = 0
    lastRModularity = -1
    currentRModularity = -1
    currentProbabilitiesRange = [0.0,1.0]
    if(showProgress):
        pbar = tqdm(total = minSimilarTrials,leave=True)
        pbar.set_description("COARSE phase. Calculating TPR for 0.0 to 1.0.")
    currentDeviation = 1.0
    if(useCoarseStep):
        currentTPRs = [calculateTPR(0.0),calculateTPR(1.0)]
        # print("\n----\nCURRENT TPRS: ",currentTPRs)
        threshold = 1.0
        if(currentTPRs[0]<1.0 and currentTPRs[1]==1.0):
            while(True):
                threshold = (currentProbabilitiesRange[0]+currentProbabilitiesRange[1])*0.5
                thresholdTPR = calculateTPR(threshold)
                currentDeviation = abs(threshold-currentProbabilitiesRange[1])/currentProbabilitiesRange[1]
                if(thresholdTPR==1.0):
                    currentProbabilitiesRange[1] = threshold
                else:
                    currentProbabilitiesRange[0] = threshold
                if(showProgress):
                    pbar.set_description("COARSE phase. Range: [%g - %g]. Deviation: %g (target=%g) Trials" % (currentProbabilitiesRange[0],currentProbabilitiesRange[1],currentDeviation,coarseError))
                if(currentDeviation<coarseError):
                    break
        elif(currentTPRs[0]==1.0):
            if(useMultiprocessing):
                pool.close()
                pool.join()
            return 0.0
    oldTPR = -1
    trivialCount= 0
    allPerturbationCount = 0
    # print("\n----\nCURRENT PROBABILITIES RANGE: ",currentProbabilitiesRange)
    while(similarTrial<minSimilarTrials):
        probabilities=np.random.random(perturbationCount)*(currentProbabilitiesRange[1])
        trivialCount += perturbationCount*detectionTrials*calculateTPR(probabilities)
        allPerturbationCount += perturbationCount*detectionTrials
        newTPR = 1.0-trivialCount/allPerturbationCount
        absDiff = 0
        
        if(oldTPR < 0):
            similarTrial+=1
        if(oldTPR < 1e-20): # zero
            absDiff = abs(newTPR-oldTPR)
            if(absDiff<fineError):
                similarTrial+=1
        else:
            absDiff = abs(newTPR-oldTPR)/oldTPR
            if(absDiff<fineError):
                similarTrial+=1
        
        oldTPR = newTPR
        
        if(showProgress):
            pbar.set_description("FINE Phase. Deviation: %g (target=%g) Trials" % (absDiff,fineError))
            pbar.reset()
            pbar.update(similarTrial)
    if(showProgress):
        pbar.refresh()
        pbar.close()
    if(useMultiprocessing):
        pool.close()
        pool.join()
    return currentProbabilitiesRange[1]*(1.0-trivialCount/allPerturbationCount)

