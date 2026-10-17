using Pkg
Pkg.activate(@__DIR__)
Pkg.add(["ReachabilityAnalysis", "LazySets", "JSON"])
using ReachabilityAnalysis
println("RA ok ", pkgversion(ReachabilityAnalysis))
