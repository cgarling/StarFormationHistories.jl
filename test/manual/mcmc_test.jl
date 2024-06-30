# DynamicHMC.jl looks pretty good but it's API seems quite complicated; you need to follow the LogDensityProblems.jl API, but you can use TransformVariables.jl and TransformedLogDensities.jl to do variable transformations for you and .... just seems very complicated.
import LogDensityProblems
import TransformVariables
import TransformedLogDensities
import LogDensityProblemsAD
import ForwardDiff
import StaticArrays: SVector, SMatrix, @SVector
import DynamicHMC
import Random
import MCMCDiagnosticTools: ess_rhat
import Statistics
import Distributions
import LinearAlgebra: det, diag, transpose
import LineSearches # For comparison to DynamicHMC
import Optim        # For comparison to DynamicHMC
import StarFormationHistories: composite!, loglikelihood, ∇loglikelihood, ∇loglikelihood!

# import PyCall: @pyimport
# @pyimport matplotlib.pyplot as plt
# import LaTeXStrings: @L_str
import PyPlot as plt
import PyPlot: @L_str # For LatexStrings
plt.rc("text", usetex=true)
plt.rc("font", family="serif", serif=["Computer Modern"], size=20)
plt.rc("figure", figsize=(10,10))
plt.rc("patch", linewidth=1, edgecolor="k", force_edgecolor=true) # not sure this is good.


struct SFHModel{T,S,V}
    models::T
    composite::S
    data::V
end

# This model will return loglikelihood and gradient
LogDensityProblems.capabilities(::Type{<:SFHModel}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(problem::SFHModel) = length(problem.models)

function LogDensityProblems.logdensity_and_gradient(problem::SFHModel, logx)
    composite = problem.composite
    models = problem.models
    data = problem.data
    # dims = length(models)
    # Transform the provided x
    # x = SVector{dims}(exp(i) for i in logx)
    x = [ exp(i) for i in logx ]
    # Update the composite model matrix
    composite!( composite, x, models )
    # composite = sum( x .* models )
    logL = loglikelihood(composite, data) + sum(logx) # + sum(logx) is the log-Jacobian correction
    ∇logL = [ ∇loglikelihood(models[i], composite, data) * x[i] + 1 for i in eachindex(models,x) ] # The `* x[i] + 1` is the Jacobian correction
    # logL = loglikelihood(composite, data) 
    # ∇logL = [ ∇loglikelihood(models[i], composite, data) * x[i] for i in eachindex(models,x) ] 
    return logL, ∇logL
end
# Similar but for Optim.jl; expects -logL returned
function fg!(F,G,logx,models,data,composite) # This uses log-transform
    # Transform the provided x
    x = [exp(i) for i in logx]
    # Update the composite model matrix
    composite!( composite, x, models )
    logL = -loglikelihood(composite, data) - sum(logx) # + sum(logx) is the log-Jacobian correction
    for i in eachindex(G,logx)
        G[i] = -∇loglikelihood(models[i], composite, data) * x[i] - 1
    end
    return logL
end
# Without log-transform; this is worse
function fg2!(F,G,x,models,data,composite)
    # Update the composite model matrix
    composite!( composite, x, models )
    logL = -loglikelihood(composite, data)
    for i in eachindex(G,x)
        G[i] = -∇loglikelihood(models[i], composite, data)
    end
    return logL
end
# With x^2 transform; this performs very similarly to fg! with original log-transform
function fg3!(F,G,sqrtx,models,data,composite) 
    # Transform the provided x
    x = [i^2 for i in sqrtx]
    # Update the composite model matrix
    composite!( composite, x, models )
    logL = -loglikelihood(composite, data) - 2*sum(sqrtx) # + sum(logx) is the log-Jacobian correction
    for i in eachindex(G,sqrtx)
        G[i] = -∇loglikelihood(models[i], composite, data) * 2 * sqrtx[i] - 2
    end
    return logL
end
# Test code

nmodels = 100
histsize = (99,99)

models = [rand(histsize...) for i in 1:nmodels]
coeffs = rand(nmodels) .* 5
# data = sum(coeffs .* models) # This is easy; no error
data = rand.( Distributions.Poisson.(sum(coeffs .* models)) ) # This is harder

# Multiplie composite scratch arrays for parallel sampling
composite_arr = [ Matrix{Float64}(undef,histsize) for i in 1:Threads.nthreads() ]
composite = first(composite_arr)
# HMC Stuff
instances = [ SFHModel( models, composite_arr[i], data ) for i in 1:Threads.nthreads() ]
result = []
Threads.@threads for i in 1:4
    tid = Threads.threadid()
    r = DynamicHMC.mcmc_with_warmup(Random.default_rng(), instances[tid], 1000;
                                reporter=DynamicHMC.ProgressMeterReport())
    push!(result, r)
end
# Now compare with result from Optim.jl
optim_result = Optim.optimize( Optim.only_fg!( (F,G,logx) -> fg!(F,G,logx,models,data,composite) ),
                               # rand(nmodels) .* 5,
                               log.(rand(nmodels) .* 5),
                               Optim.BFGS(; alphaguess=LineSearches.InitialStatic(1.0,true),
                                          linesearch=LineSearches.HagerZhang()),
                               Optim.Options(; allow_f_increases=true, store_trace=true, extended_trace=true) )
# Now don't use the log-transform
# This finds exact result when no error but actually performs worse on sampled data
# than the log-transformed version above. The errors along the diagonal of the inv(H)
# estimate are also non-sensical ... 
# optim_result = Optim.optimize( Optim.only_fg!( (F,G,x) -> fg2!(F,G,x,models,data,composite) ),
#                                rand(nmodels) .* 5,
#                                Optim.BFGS(; alphaguess=LineSearches.InitialStatic(1.0,true),
#                                           linesearch=LineSearches.HagerZhang()),
#                                Optim.Options(; allow_f_increases=true, store_trace=true, extended_trace=true) )
# optim_coeffs = Optim.minimizer(optim_result)
# optim_std = sqrt.(diag(optim_result.trace[end].metadata["~inv(H)"]))
# Now use the x^2 transform
# optim_result = Optim.optimize( Optim.only_fg!( (F,G,sqrtx) -> fg3!(F,G,sqrtx,models,data,composite) ),
#                                sqrt.(rand(nmodels) .* 5),
#                                Optim.BFGS(; alphaguess=LineSearches.InitialStatic(1.0,true),
#                                           linesearch=LineSearches.HagerZhang()),
#                                Optim.Options(; allow_f_increases=true, store_trace=true, extended_trace=true) )
# optim_coeffs = Optim.minimizer(optim_result).^2
# optim_std = 2 .* sqrt.(diag(optim_result.trace[end].metadata["~inv(H)"]) .* optim_coeffs)


# We can also change where the mcmc starts;
# result = DynamicHMC.mcmc_with_warmup(Random.default_rng(), inst, 1000; reporter = DynamicHMC.ProgressMeterReport(), initialization = ( q = max.(log.(coeffs),-2.0),) )
# reporter=DynamicHMC.ProgressMeterReport()
# reporter=DynamicHMC.LogProgressReport(step_interval=1000, time_interval_s=1)
# reporter=DynamicHMC.NoProgressReport()
# @time DynamicHMC.mcmc_with_warmup(Random.default_rng(), inst, 100; reporter=DynamicHMC.NoProgressReport()) # 3.8s
# @time DynamicHMC.mcmc_with_warmup(Random.default_rng(), inst, 100; reporter=DynamicHMC.NoProgressReport(), initialization=(q=log.(coeffs),)) # not much change really

# HMC diagnostics
DynamicHMC.Diagnostics.summarize_tree_statistics(result[1].tree_statistics)
# ess_rhat(DynamicHMC.stack_posterior_matrices(result))
# ess_rhat(Statistics.mean, result.posterior_matrix)
result2 = DynamicHMC.stack_posterior_matrices(result) # dim=3
result3 = exp.( DynamicHMC.pool_posterior_matrices(result) ) # dim=2
hmc_coeffs = Statistics.median.(eachrow(result3))
hmc_std = Statistics.std.(eachrow(result3))

println(" HMC Diagnostics ")
println("=================================")
# println("ess_rhat: ",ess_rhat(Statistics.mean, result2))
# println("HMC Means: ",Statistics.mean.(eachrow(result3)))
# println("HMC STDs: ", Statistics.std.(eachrow(result3)))
# println("Correct Coeffs: ", coeffs)
std_errs = abs.(hmc_coeffs .- coeffs) ./ hmc_std
# println( "HMC Error in Standard Deviations: ", std_errs)
println( "Median HMC Error in Standard Deviations: ",
         Statistics.median(std_errs)) 
# If the uncertainties are realistic, the following should be 
# ~0.32 as 32% of the time the estimated parameter should be more
# than 1-σ away from the correct answer.
# And it is ... even for large `nmodels` like 100. 
println("Percent of HMC coefficients with errors greater than one STD: ",
        count(std_errs .> 1) / length(std_errs))

# Optim.jl diagnostics
println(" Optim.jl Diagnostics ")
println("=================================")
optim_coeffs = exp.(Optim.minimizer(optim_result))
# optim_std = sqrt.(diag(optim_result.trace[end].metadata["~inv(H)"])) .* optim_coeffs # This is correct for log-transform
optim_std = sqrt.(diag(optim_result.trace[end].metadata["~inv(H)"])) .* optim_coeffs # This is correct for log-transform
optim_std_errs = abs.(optim_coeffs.- coeffs) ./ optim_std
println( "Median Optim.jl Error in Standard Deviations: ",
         Statistics.median(optim_std_errs)) 
println("Percent of Optim.jl coefficients with errors greater than one STD: ",
        count(optim_std_errs .> 1) / length(optim_std_errs))
###########################################################################################
# Devise a test (maybe Gaussian distribution) to test the log transform that I'm using for SFH.jl. Make sure it's doing the right thing. Try to sample from a multivariate normal. 

# Make standard Distributions.jl model and sample
dist1 = Distributions.MvNormal([20.0, 30.0], [2.0 0.0; 0.0 5.0])
samp1 = rand(dist1, 10000)

# Set up model for DynamicHMC.jl
# This will be for the automatic transformation attempt. 
struct GaussModel{T,S}
    means::T
    cov::S
end

# This model will return loglikelihood
LogDensityProblems.capabilities(::Type{<:GaussModel}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.dimension(problem::GaussModel) = 2

# function LogDensityProblems.logdensity_and_gradient(problem::GaussModel, logx)
# end
function (problem::GaussModel)(θ)
    # x, y = θ
    Σ = problem.cov
    μ = problem.means
    # loglikelihood
    δx = θ - μ
    return -log(2π) - log(det(Σ))/2 - transpose(δx) * inv(Σ) * δx / 2
    # return log( inv(2π) * inv(sqrt(det(Σ))) * exp( -transpose(δx) * inv(Σ) * δx / 2 ) )
end

# dist2 = GaussModel([2.0, -3.0], [2.0 0.0; 0.0 5.0])
# @benchmark dist2($[1.0,0.0]) = 600 ns, 5 allocs
dist2 = GaussModel(SVector{2}(20.0, 30.0), SMatrix{2,2}(2.0, 0.0, 0.0, 5.0))
# @benchmark dist2($SVector(1.0,0.0)) = 50 ns, 1 allocs

trans = TransformVariables.as(Array, TransformVariables.asℝ₊, 2)
P = TransformedLogDensities.TransformedLogDensity( trans, dist2 )
∇P = LogDensityProblemsAD.ADgradient(:ForwardDiff, P)
# Run MCMC
result = DynamicHMC.mcmc_with_warmup(Random.default_rng(), ∇P, 10000; reporter = DynamicHMC.NoProgressReport())
# hmc_samples = TransformVariables.transform.(trans, eachcol(result.posterior_matrix))
hmc_samples = hcat( TransformVariables.transform.(trans, eachcol(result.posterior_matrix))...)


fig,ax1=plt.subplots()
ax1.scatter(view(samp1,1,:), view(samp1,2,:), label="Distributions.jl", s=0.1)
ax1.scatter(view(hmc_samples,1,:), view(hmc_samples,2,:), label="DynamicHMC.jl", s=0.1)
ax1.legend()

plt.show(fig)

# All of this looks good. Now try manual variable transformation.
################################################################################
struct GaussModel2{T,S}
    means::T
    cov::S
end

# This model will return loglikelihood and gradient
LogDensityProblems.capabilities(::Type{<:GaussModel2}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(problem::GaussModel2) = 2

function LogDensityProblems.logdensity_and_gradient(problem::GaussModel2, logθ)
    # x, y = θ
    # θ = SVector{length(logθ)}(exp(i) for i in logθ)
    # This is faster than above
    θ = SVector{LogDensityProblems.dimension(problem)}(exp(i) for i in logθ)
    Σ = problem.cov
    μ = problem.means
    # loglikelihood
    δx = θ - μ
    # logL = -log(2π) - log(det(Σ))/2 - transpose(δx) * inv(Σ) * δx / 2 
    # ∇logL = -Σ \ (θ - μ)
    logL = -log(2π) - log(det(Σ))/2 - transpose(δx) * inv(Σ) * δx / 2 + sum(logθ)
    ∇logL = -Σ \ (θ - μ) .* θ .+ 1 # not sure why the +1 is needed;
                                   # Actually I think it's for the sum(logθ) in logL. 
    # return logL, ∇logL
    return logL, Vector(∇logL)
    # return logL, tuple(∇logL...) # This actually works but I think it does an internal conversion to vector anyway so not faster.
end

dist3 = GaussModel2(SVector{2}(20.0, 30.0), SMatrix{2,2}(2.0, 0.0, 0.0, 5.0))
# Run MCMC
# This fails if logdensity_and_gradient returns a staticarray for the gradient; think it's because of q's type
result2 = DynamicHMC.mcmc_with_warmup(Random.default_rng(), dist3, 10000; reporter = DynamicHMC.NoProgressReport())
# This still doesn't work
# result2 = DynamicHMC.mcmc_with_warmup(Random.default_rng(), dist3, 10000; reporter = DynamicHMC.ProgressMeterReport(), initialization=(q=(@SVector rand(2)).*2,) )
# hmc_samples = TransformVariables.transform.(trans, eachcol(result.posterior_matrix))
hmc_samples2 = exp.(result2.posterior_matrix)

fig,ax1=plt.subplots()
ax1.scatter(view(samp1,1,:), view(samp1,2,:), label="Distributions.jl", s=0.1)
# ax1.scatter(view(hmc_samples,1,:), view(hmc_samples,2,:), label="DynamicHMC.jl", s=0.1)
ax1.scatter(view(hmc_samples2,1,:), view(hmc_samples2,2,:), label="DynamicHMC.jl Manual", s=0.1)
ax1.legend()

plt.show(fig)
# Samples again look good. Let's check this more explicitly though.
# Check that the logdensities are the same
@assert isapprox( LogDensityProblems.logdensity(P,[1.0,1.0]), LogDensityProblems.logdensity_and_gradient(dist3,[1.0,1.0])[1]; rtol=1e-3 )
# Check that the gradients are the same
@assert isapprox( ForwardDiff.gradient(x->LogDensityProblems.logdensity(P,x),[1.0,1.0]),
                  LogDensityProblems.logdensity_and_gradient(dist3,[1.0,1.0])[2]; rtol=1e-3 )
