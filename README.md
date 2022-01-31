# Robustness Modularity
Small utility to calculate the robustness modularity, information modularity
and modularity difference.

## License
The software is distributed under the [MIT license](LICENSE.md).
  
## Authors
  * Filipi N. Silva <filsilva@iu.edu>
  * Santo Fortunato <santo@iu.edu>

## Dependencies
Robustness Modularity requires the following dependencies:
  
  * [Python 3.6+](https://www.python.org/downloads/)
  * [Numpy](http://www.numpy.org/)
  * [graph-tool](https://graph-tool.skewed.de)
  * [louvain](https://pypi.org/project/louvain/)
  * [python-igraph](https://igraph.org/python/)

Except for `graph-tool` all the other packages can be installed using pip:
```bash
pip install -r requirements.txt
```

To install `graph-tool` follow the instructions in the [graph-tool documentation](https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions).

We recommend to install it using [conda](https://conda.io/):
```bash
conda install -c conda-forge graph-tool
```

## Installation
After installing `graph-tool` dependency, the tool can be installed using pip:

```bash
pip install RModularity
```

or from source:
```bash
pip git+https://github.com/filipinascimento/RModularity.git
```

## Usage
We provide three networks for testing in the `SampleNetworks` folder.
A full usage example can be found in `example.py`.

First import the `RModularity` module:
```python
import RModularity
```

For this example we will be using `igraph` to load the sample networks and
`pathlib` to deal with the paths:
```python
import igraph as ig
from pathlib import Path
```

When multiprocessing is enabled, the all the calculations need to be done
in the main process, thus use `if __name__ == '__main__':`. Let's load a
network from the `SampleNetworks` folder and define some paths:
```python
if __name__ == '__main__':
    networkName = "road-euroroad"
    # networkName = "LFR_mu0.1"
    # networkName = "LFR_mu1.0"
    
    outputSuffix = ""
    figurePath = Path("Figures")
    figurePath.mkdir(parents=True, exist_ok=True)

    g = ig.Graph.Read_GML(str(Path("SampleNetworks")/("%s.gml" % networkName)))
```

You can calculate the approximated robustness modularity using the `RModularityFast` function, which implements the fast Monte-Carlo algorithm.
```python
    Q_rA = RModularity.RModularityFast(
        g.vcount(), # Number of nodes
        g.get_edgelist(), # Edges list
        g.is_directed(), # Directed or not
        )
    print("Q_rA = ", Q_rA)
```

You can use the `RModularity` function to calculate the robustness modularity without approximations:
```python
    Q_r, probabilities, TPRCurve, \
     DLCurvesTrivial, DLCurvesDetected = RModularity.RModularity(
         g.vcount(), # Number of nodes
         g.get_edgelist(), # Edges list
         g.is_directed(), # Directed or not
         outputCurves=True,
         )

    print("Q_r = ", Q_r)
```
By setting `outputCurves` to `True`, the Trivial Partition Ratio (TPR) and the description lengths of the detected and trivial partitions will be returned.

Modularity difference (Q_diff) can be calculated using the `modularityDifference` function:
```python
    Q_diff = RModularity.modularityDifference(
        g.vcount(),
        g.get_edgelist(),
        g.is_directed()
    )
```

Information modularity can be calculated using the `informationModularity`
function:
```python
    Q_DL = RModularity.informationModularity(
        g.vcount(),
        g.get_edgelist(),
        g.is_directed()
    )
    print("Q_DL = ", Q_DL)
```

Here we also illustrate how to generate the TPR and Description lengths plots.
First let's import a few extra packages
```python
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpl_patches
```

Let's calculate average and std for the curves:
```python
    avgDLCurvesTrivial = np.mean(DLCurvesTrivial, axis=1)
    avgDLCurvesDetected = np.mean(DLCurvesDetected, axis=1)
    stdDLCurvesTrivial = np.std(DLCurvesTrivial, axis=1)
    stdDLCurvesDetected = np.std(DLCurvesDetected, axis=1)

    diffDLCurves = (DLCurvesTrivial-DLCurvesDetected) / DLCurvesTrivial
    avgDiffDLCurves = np.mean(diffDLCurves, axis=1)
    stdDiffDLCurves = np.std(diffDLCurves, axis=1)
```

Now let's plot the TPR curve:
```python
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
```

Now let's plot the description length curve:
```python
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
```

And finally, let's plot the information modularity along `p`:
```python
    fig = plt.figure(figsize=(3*1.61803398875, 3))
    ax = plt.axes((0.2, 0.2, 0.70, 0.70), facecolor='w')
    ax.plot(probabilities, avgDiffDLCurves, lw=2.0, label="Detected")
    ax.fill_between(probabilities, avgDiffDLCurves-stdDiffDLCurves,
                    avgDiffDLCurves+stdDiffDLCurves, alpha=0.2)

    ax.set_xlabel("$p$")
    ax.set_ylabel("$Q_\mathrm{DL}$")
    ax.set_title(networkName)
    ax.set_xlim(-0.00, 1.02)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(1.5)
    ax.tick_params(width=1.5)
    fig.savefig(figurePath/("DLDiff_%s%s.pdf" % (networkName, outputSuffix)))
    plt.close(fig)
```

Please refer to the next section for more details on how to use this library.


## Full API documentation

### <kbd>function</kbd> `RModularityFast`
```python
RModularity(
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
)
```

Computes the approximated Robustness Modularity of a network
using a Monte-Carlo approach. Note that this approach can not
procude the curves of TPR.

Parameters 
  * `nodeCount` : `int`  
    The number of nodes in the network.
  * `edges` : list of tuples  
    A list of the edges in the network.
  * `directed` : `int`, optional  
    Whether the network is directed or not.
  * `perturbationCount` : `int`, optional  
    The number of perturbations to perform.  (defaults to 25)
  * `detectionTrials` : `int`, optional  
    The number of times to perform community  detection for each perturbed network. (defaults to 1)
  * `rewireResolution` : `int`, optional  
    The number values points for the rewire  probabilities (from 0 to 1) to calculate  the Trivial Partition Ratio (TPR) curves  and Robustness Modularity. (defaults to 51)
  * `showProgress` : `bool`, optional  
    Shows a progress bar if enabled.  (defaults to True)
  * `useMultiprocessing`: `bool`, optional  
    Uses parallel processing to calculate  Rmodularity.  (defaults to True)
  * `useCoarseStep`: `bool`, optional
    Finds the plateal region using a binary search before applying
    the Monte-Carlo approach. (defaults to True)
  * `fineError`: `float`, optional
    Error tolerance for the fine step. (defaults to 0.01)
  * `coarseError`: `float`, optional
      Error tolerance for the coarse step. (defaults to 0.02)
  * `minSimilarTrials`: `int`, optional
      The minimum number of similar trials to perform before 
      stopping the Monte-Carlo approach.(defaults to 2)
Returns 
  * `float` if `outputCurves` is `False`  
    The Robustness Modularity of the network.


---

### <kbd>function</kbd> `RModularity`
```python
RModularity(
    nodeCount,
    edges,
    directed=False,
    perturbationCount=25,
    detectionTrials=1,
    rewireResolution=51,
    outputCurves=False,
    showProgress=True,
    useMultiprocessing=True
)
```

Computes the Robustness Modularity of a network. 

Parameters 
  * `nodeCount` : `int`  
    The number of nodes in the network.
  * `edges` : list of tuples  
    A list of the edges in the network.
  * `directed` : `int`, optional  
    Whether the network is directed or not.
  * `perturbationCount` : `int`, optional  
    The number of perturbations to perform.  (defaults to 25)
  * `detectionTrials` : `int`, optional  
    The number of times to perform community  detection for each perturbed network. (defaults to 1)
  * `rewireResolution` : `int`, optional  
    The number values points for the rewire  probabilities (from 0 to 1) to calculate  the Trivial Partition Ratio (TPR) curves  and Robustness Modularity.(defaults to 51)
  * `outputCurves` : `bool`, optional  
Whether to save the TPR and DL curves. (defaults to False)
  * `showProgress` : `bool`, optional  
Shows a progress bar if enabled.  (defaults to True)
  * `useMultiprocessing`: `bool`, optional  
    Uses parallel processing to calculate  Rmodularity.  (defaults to True)

Returns 
  * `float` if `outputCurves` is `False`  
    The Robustness Modularity of the network.
  * `(float, np.array dim=1, np.array dim=1, np.array dim=2, np.array dim=2)` if `outputCurves` is `True`  
    Returns a tuple of 4 values containing the Robustness Modularity, the rewire probabilities, the TPR curves, and the Description lenghts for the detected and trivial partitions. 


---

### <kbd>function</kbd> `modularityDifference`

```python
modularityDifference(
    nodeCount,
    edges,
    directed=False,
    detectionTrials=100,
    nullmodelCount=100,
    detectionTrialsNullModel=10,
    showProgress=True,
    useMultiprocessing=True
)
```

Computes the Modularity Difference of a network. 

Parameters 
  * `nodeCount` : `int`  
    The number of nodes in the network.
  * `edges` : list of tuples  
    A list of the edges in the network.
  * `directed` : `int`, optional  
    Whether the network is directed or not.
  * `detectionTrials` : `int`, optional  
    The number of times to perform community detection for the input network  (defaults to 100)
  * `nullmodelCount` : `int`, optional  
    The number of realizations of the null-model (configuration model) used to calculate the null-model modularity. (defaults to 100)
  * `detectionTrialsNullModel` : `int`, optional  
    The number of times to perform community detection for each perturbed network  (defaults to 10)
  * `showProgress` : `bool`, optional  
Shows a progress bar if enabled.  (defaults to True)
  * `useMultiprocessing`: `bool`, optional  
    Uses parallel processing to calculate  Rmodularity.  (defaults to True)

Returns 
  * `float`  
    The Modularity Difference of the network.
    

### <kbd>function</kbd> `informationModularity`

```python
informationModularity(nodeCount, edges, directed=False)
```

Computes the Information Modularity of a network. 

Parameters 
  * `nodeCount` : `int`  
    The number of nodes in the network.
  * `edges` : list of tuples  
    A list of the edges in the network.
  * `directed` : `int`, optional  
    Whether the network is directed or not.

Returns 
  * `float`  
    The Information Modularity of the network.
    
