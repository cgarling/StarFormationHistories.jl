# Idea for a unified metallicity evolution model
# Specify a time-evolution model for the metallicity, for example
# MH(age,α,β) = α * age + β, a model that is linear with age in years.
# At a fixed age, the mean metallicity will be given by the metallicity model,
# μ(age,α,β) = MH(age,α,β),
# and the coefficients on the different metallicity isochrones at that fixed
# age will be set according to a Gaussian pdf with mean μ and fixed std σ. 

using Symbolics
@variables A α β age σ MH T_max
# Mean of MDF is equal to α * age + β, linear metallicity evolution model
μ = α * (T_max - age) + β
dα = Differential(α) # differential with respect to α
dβ = Differential(β) # differential with respect to β
# 1D Gaussian PDF, with normalization factor A
# gausspdf(x,μ,σ) = A * inv(σ * sqrt(2π)) * exp( -((x-μ)/σ)^2 / 2 ) 
gausspdf(x,μ,σ) = A * exp( -((x-μ)/σ)^2 / 2 ) # / σ
# Derivative of gaussian pdf with respect to α
dGdα = expand_derivatives(dα(gausspdf(MH,μ,σ)))
dGdβ = expand_derivatives(dβ(gausspdf(MH,μ,σ)))

# Try to redefine with the full A
@variables α β age σ MH1 MH2 MH3 T_max
μ = α * (T_max - age) + β
dα = Differential(α) # differential with respect to α
dβ = Differential(β) # differential with respect to β
gausspdf(x,μ,σ,A) = A / σ * exp( -((x-μ)/σ)^2 / 2 ) 
A = gausspdf(MH1,μ,σ,1) + gausspdf(MH2,μ,σ,1) + gausspdf(MH3,μ,σ,1)
V1 = gausspdf(MH1,μ,σ,A)
# expand_derivatives(dβ(V1))
# simplify(expand_derivatives(dβ(V1)))


# Try to derive the original dL/dr_k
@variables M C D rk vp
# M is individual models, C is composite of all models, D is data, rk is per-model coefficients,
# vp is per log-age coefficients
C = rk * M
L = D - C - D * log(D/C)
drk = Differential(rk)
dL_drk = simplify(expand_derivatives(drk(L)))
# @variables M[1:2,1:2] C[1:2,1:2] D[1:2,1:2] rk[1:2] vp
# C = sum( rk .* M )
# L = sum( @. D - C - D * log(D/C) )
# drk = Differential(rk)
# dL_drk = simplify(expand_derivatives(drk(L)))
