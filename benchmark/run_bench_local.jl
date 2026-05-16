# Script for running the benchmarks locally and making a pretty table of results

using BenchmarkTools
using PrettyTables

include("benchmarks.jl") 

# Run the benchmarks
results = run(SUITE, verbose=true)

# Collect results 
sorted  = sort(collect(results["core"]), by=first)
names   = [k for (k,_) in sorted]
trials  = [v for (_,v) in sorted]

# Pack into matrix
data = hcat(
    names,
    [BenchmarkTools.prettytime(median(t).time) for t in trials],
    [BenchmarkTools.prettymemory(median(t).memory) for t in trials],
    [median(t).allocs for t in trials]
)

# Make pretty table
pretty_table(data;
    column_labels = ["Benchmark", "Median Time", "Memory", "Allocs"],
    alignment     = [:l, :r, :r, :r]
)
