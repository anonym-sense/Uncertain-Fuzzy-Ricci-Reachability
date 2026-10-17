# Third-party notice

No third-party source code, models, or data are vendored in this repository. The MIT
license covers only the code written here. The following external dependencies are
installed separately and keep their own licenses:

| dependency | source | license |
|---|---|---|
| NumPy, SciPy, matplotlib | numpy.org, scipy.org, matplotlib.org | BSD-3-Clause / PSF-based |
| CVXPY, Clarabel | github.com/cvxpy/cvxpy, github.com/oxfordcontrol/Clarabel.rs | Apache-2.0 |
| POT (Python Optimal Transport) | github.com/PythonOT/POT | MIT |
| hj_reachability (optional) | github.com/StanfordASL/hj_reachability | MIT |
| ReachabilityAnalysis.jl, LazySets.jl (JuliaReach) | github.com/JuliaReach | MIT |

The benchmark models are standard textbook or competition systems (Van der Pol,
adaptive cruise control, planar quadrotor, Duffing chain), re-implemented here.
