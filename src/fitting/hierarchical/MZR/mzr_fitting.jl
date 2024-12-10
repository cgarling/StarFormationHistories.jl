# This file contains SFH fitting and sampling methods that support the mass-metallicity relations defined in mzr_models.jl

function calculate_coeffs(mzr_model::AbstractMZR{T}, disp_model::AbstractDispersionModel{U},
                          mstars::AbstractVector{<:Number}, 
                          logAge::AbstractVector{<:Number},
                          metallicities::AbstractVector{<:Number}) where {T, U}
    unique_logAge = unique(logAge)
    @assert(length(mstars) == length(unique_logAge),
            "Length of `mstars` must be the same as `unique_logAge`.")
    @assert length(logAge) == length(metallicities)
    S = promote_type(eltype(mstars), eltype(logAge), eltype(metallicities), T, U)

    # To calculate cumulative stellar mass, we need unique_logAge sorted in reverse order
    s_idxs = sortperm(unique_logAge; rev=true)

    # Set up to calculate coefficients
    coeffs = Vector{S}(undef, length(logAge))
    # norm_vals = Vector{S}(undef, length(unique_logAge))
    # Calculate cumulative stellar mass vector, properly sorted, then put back into original order
    cum_mstar = cumsum(mstars[s_idxs])[invperm(s_idxs)]# [reverse(s_idxs)]
    for i in eachindex(unique_logAge)
        la = unique_logAge[i]
        # Find the mean metallicity of this age bin based on the cumulative stellar mass
        μ = mzr_model(cum_mstar[i])
        idxs = findall(==(la), logAge) # Find all entries that match this logAge
        tmp_coeffs = [disp_model(metallicities[ii], μ) for ii in idxs] # Calculate relative weights
        A = sum(tmp_coeffs)
        # norm_vals[i] = A
        # Make sure sum over tmp_coeffs equals 1 and write to coeffs
        coeffs[idxs] .= tmp_coeffs .* mstars[i] ./ A
    end
    return coeffs
end


# Function to compute objective and gradient for MZR models
function fg_mzr!(F, G, mzr0::AbstractMZR{T}, disp0::AbstractDispersionModel{U},
                 variables::AbstractVector{<:Number},
                 models::Union{AbstractMatrix{<:Number},
                               AbstractVector{<:AbstractMatrix{<:Number}}},
                 data::Union{AbstractVector{<:Number}, AbstractMatrix{<:Number}},
                 composite::Union{AbstractVector{<:Number}, AbstractMatrix{<:Number}},
                 logAge::AbstractVector{<:Number},
                 metallicities::AbstractVector{<:Number}) where {T, U}
    # `variables` should have length `length(unique(logAge)) + 3`; coeffs for each unique
    # entry in logAge, plus α and β to define the MZR and σ to define Gaussian width
    # Removing gaussian width for now so just length `length(unique(logAge)) + 2`
    @assert axes(data) == axes(composite)
    S = promote_type(eltype(variables), eltype(eltype(models)), eltype(eltype(data)),
                     eltype(composite), eltype(logAge), eltype(metallicities),
                     T, U)
    # Number of fittable parameters in MZR model;
    # in G, these come after the stellar mass coefficients R_j
    mzrpar = nparams(mzr0)
    # Number of fittable parameters in metallicity dispersion model;
    # in G, these come after the MZR parameters
    disppar = nparams(disp0)

    # Construct new instance of mzr0 with updated parameters
    mzr_model = update_params(mzr0, @view(variables[end-(mzrpar+disppar)+1:(end-disppar)]))
    # Get indices of free parameters in MZR model
    mzr_free = BitVector(free_params(mzr_model))
    # Construct new instance of disp0 with updated parameters
    disp_model = update_params(disp0, @view(variables[end-(disppar)+1:end]))
    # Get indices of free parameters in metallicity dispersion model
    disp_free = BitVector(free_params(disp_model))
    # Calculate all coefficients r_{j,k} for each template
    coeffs = calculate_coeffs(mzr_model, disp_model,
                              @view(variables[begin:end-(mzrpar+disppar)]),
                              logAge, metallicities)

    # Fill the composite array with the equivalent of sum( coeffs .* models )
    # composite = sum( coeffs .* models )
    # return -loglikelihood(composite, data)
    composite!(composite, coeffs, models)
    # Need to do compute logL before ∇loglikelihood! because it will overwrite composite
    logL = loglikelihood(composite, data)

    if (F != nothing) & (G != nothing) # Optim.optimize wants objective and gradient
        @assert axes(G) == axes(variables)
        # Calculate the ∇loglikelihood with respect to model coefficients
        fullG = Vector{eltype(G)}(undef, length(coeffs))
        ∇loglikelihood!(fullG, composite, models, data)

        unique_logAge = unique(logAge)
        # Cumulative stellar mass, from earliest time to most recent
        # cum_mstar = cumsum(@view(variables[begin:end-(mzrpar+disppar)]))
        s_idxs = sortperm(unique_logAge; rev=true)
        cum_mstar = cumsum(variables[s_idxs])[invperm(s_idxs)]
    
        # Find indices into unique_logAge for each entry in logAge
        jidx = [findfirst(==(logAge[i]), unique_logAge) for i in eachindex(logAge)]
         # Find indicies into logAge for each entry in unique_logAge
        jidx_inv = [findall(==(unique_logAge[i]), logAge) for i in eachindex(unique_logAge)]
        
        # Calculate quantities from MZR model
        μvec = mzr_model.(cum_mstar) # Find the mean metallicity of each time bin
        # Calculate full gradient of MZR model with respect to parameters and
        # cumulative stellar masses
        gradμ = tups_to_mat(values.(gradient.(mzr_model, cum_mstar)))
        # \frac{\partial μ_j}{\partial M_*}; This should always be the last row in gradμ
        dμ_dRj_vec = gradμ[end,:]
        
        # Calculate quantities from metallicity dispersion model
        # Relative weights A_{j,k} for *all* templates
        tmp_coeffs_vec = disp_model.(metallicities, μvec[jidx])
        # Full gradient of the dispersion model with respect to parameters
        # and mean metallicity μ_j
        grad_disp = tups_to_mat(values.(gradient.(disp_model, metallicities, μvec[jidx])))
        # # \frac{\partial A_{j,k}}{\partial \mu_j}
        dAjk_dμj = grad_disp[end,:]
        # Calculate sum_k A_{j,k} for all j
        A_vec = [sum(tmp_coeffs_vec[ii] for ii in idxs) for idxs in jidx_inv]

        # \frac{\partial A_{j,k}}{\partial R_j}; one entry for every template
        dAjk_dRj = dAjk_dμj .* dμ_dRj_vec[jidx]
        # sum_k dAjk_dRj; one entry per entry in unique(logAge)
        ksum_dAjk_dRj = [sum(dAjk_dRj[ii] for ii in idxs) for idxs in jidx_inv]
        # \frac{\partial F}{\partial r_{j,k}} * \frac{\partial r_{j,k}}{\partial R_j}
        drjk_dRj = fullG .* (variables[jidx] ./ A_vec[jidx] .*
            (dAjk_dRj .- (tmp_coeffs_vec .* ksum_dAjk_dRj[jidx] ./ A_vec[jidx]) ) )
        # \sum_k of above
        ksum_drjk_dRj = [sum(drjk_dRj[ii] for ii in idxs) for idxs in jidx_inv]
        # Calculate cumulative sum for sum_{j=0}^{j=j^\prime}, remembering to permute
        # by s_idxs to get the indices in order from earliest time to latest time
        cum_drjk_dRj = reverse!(cumsum(reverse!(ksum_drjk_dRj[s_idxs])))

        # Zero-out gradient vector in preparation to accumulate sums
        G .= zero(eltype(G))

        # Add in the j^\prime \neq j terms, remembering to permute G
        # by s_idxs to put the cum_drjk_dRj back in their original order
        for i in eachindex(cum_drjk_dRj)[begin+1:end]
            G[s_idxs[i-1]] -= cum_drjk_dRj[i]
        end

        # Loop over j
        for i in eachindex(unique_logAge)
            A = A_vec[i]
            idxs = jidx_inv[i]
            # Add the j^\prime == j term
            G[i] -= sum(fullG[idx] * (coeffs[idx] / variables[i] +
                (dAjk_dRj[idx] - (ksum_dAjk_dRj[i] * tmp_coeffs_vec[idx] / A)) *
                variables[i] / A) for idx in idxs)
            
            # Add MZR terms
            ksum_dAjk_dμj = sum(dAjk_dμj[j] for j in idxs)
            psum = -sum( fullG[j] * variables[i] / A *
                (dAjk_dμj[j] - tmp_coeffs_vec[j] / A * ksum_dAjk_dμj) for j in idxs)
            for par in (1:mzrpar)[mzr_free]
                G[end-(mzrpar+disppar-par)] += psum * gradμ[par,i]
            end

            # Add metallicity dispersion terms
            for par in (1:disppar)[disp_free]
                # View into grad_disp giving the partial derivative of A_jk
                # with respect to parameter P
                dAjk_dP = view(grad_disp, par, :)
                ksum_dAjk_dP = sum(dAjk_dP[j] for j in idxs)
                G[end-(disppar-par)] -= sum( fullG[j] * variables[i] / A *
                    (dAjk_dP[j] - tmp_coeffs_vec[j] / A * ksum_dAjk_dP) for j in idxs)
            end
        end

        return -logL
        
    elseif F != nothing # Optim.optimize wants only objective
        return -logL
    end
end

# Define struct to hold optimization-time constants
# needed to calculate the logL and gradient with fg_mzr!,
# used for sampling with DynamicHMC and it may also be easier
# to reuse for BFGS optimization with Optim.jl rather
# than rewriting the closure

struct MZROptimizer{A,B,C,D,E,F,G,H}
    mzr0::A
    disp0::B
    models::C
    data::D
    composite::E
    logAge::F
    metallicities::G
    G::H
    jacobian_corrections::Bool # Whether or not to apply Jacobian corrections for variable transformations
end

# This model will return loglikelihood and gradient
LogDensityProblems.capabilities(::Type{<:MZROptimizer}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(problem::MZROptimizer) = length(problem.G)

function LogDensityProblems.logdensity_and_gradient(problem::MZROptimizer, xvec)
    # Unpack struct
    mzr0 = problem.mzr0
    disp0 = problem.disp0
    models = problem.models
    data = problem.data
    composite = problem.composite
    logAge = problem.logAge
    metallicities = problem.metallicities
    G = problem.G
    jacobian_corrections = problem.jacobian_corrections
    
    @assert axes(G) == axes(xvec)
    mzrpar = nparams(mzr0)
    disppar = nparams(disp0)
    # Calculate number of age bins from length of xvec and number of MZR, disp parameters
    Nbins = lastindex(xvec) - mzrpar - disppar
    # Transform the provided x
    # All stellar mass coefficients are transformed as they must be > 0,
    # but MZR and dispersion model coefficients may or may not be similarly constrained.
    # Use the transforms() function to determine which parameters should be transformed.
    x = similar(xvec)
    # These are the stellar mass coefficients
    for i in eachindex(xvec)[begin:Nbins]; x[i] = exp(xvec[i]); end
    tf = (transforms(mzr0)..., transforms(disp0)...)
    par = @view(xvec[Nbins+1:end])
    free = SVector(free_params(mzr0)..., free_params(disp0)...)
    # Apply logarithmic transformations
    x_mzrdisp = exptransform(par, SVector(tf))
    # Concatenate transformed stellar mass coefficients and MZR / disp parameters
    x[Nbins+1:end] .= x_mzrdisp

    nlogL = fg_mzr!(true, G, mzr0, disp0, x, models, data, composite, logAge, metallicities)
    # Add Jacobian corrections for transformed variables if jacobian_corrections == true
    # fg_mzr! returns -logL and fills G with -∇logL, so remember to invert signs in Jacobian corrections
    ptf = findall(==(1), tf)  # Find indices of variables constrained to always be positive
    ptf = ptf[free[ptf]]      # Only keep indices for variables that we are fitting
    ptf_idx = ptf .+ Nbins
    ntf = findall(==(-1), tf) # Find indices of variables constrained to always be negative
    ntf = ntf[free[ntf]]      # Only keep indices for variables that we are fitting
    if jacobian_corrections
        # For positive-constrained parameters, including stellar mass coefficients
        for i in vcat(eachindex(G)[begin:Nbins], ptf_idx)
            nlogL -= xvec[i]
            G[i] = G[i] * x[i] - 1
        end
        for i in ntf
            @warn "Negative transformations for MZR models have not yet been validated."
            i += Nbins
            nlogL += xvec[i]
            G[i] = -G[i] * x[i] + 1
        end
    else
        # Still have to correct gradient for transform, even if not adding Jacobian correction to logL
        for i in vcat(eachindex(G)[begin:Nbins], ptf_idx)
            G[i] = G[i] * x[i]
        end
        for i in ntf
            @warn "Negative transformations for MZR models have not yet been validated."
            G[i] = -G[i] * x[i]
        end        
    end

    # Optimizers and samplers honoring LogDensityProblems's API will expect positive logL and ∇logL,
    # not -logL and -∇logL as we have (nlogL and G), so return the negatives of these values.
    return -nlogL, -G
    
end


# Optim.jl BFGS fitting routine
function fit_sfh(mzr0::AbstractMZR{T}, disp0::AbstractDispersionModel{U},
                 models::AbstractMatrix{S},
                 data::AbstractVector{<:Number},
                 logAge::AbstractVector{<:Number},
                 metallicities::AbstractVector{<:Number};
                 x0::AbstractVector{<:Number} = construct_x0_mdf(logAge, convert(S, 13.7); normalize_value=1e6),
                 kws...) where {T, U, S <: Number}

    unique_logAge = unique(logAge)
    Nbins = length(x0) # Number of unique logAge bins
    @assert length(x0) == length(unique_logAge)
    @assert size(models, 1) == length(data)
    @assert size(models, 2) == length(logAge) == length(metallicities)
    composite = Vector{S}(undef, length(data)) # Scratch matrix for storing complex Hess model
    # G = Vector{S}(undef, length(x0) + nparams(mzr0) + nparams(disp0)) # Scratch matrix for storing gradient
    # Perform logarithmic transformation on the provided x0 (stellar mass coefficients)
    x0 = copy(x0) # We don't actually want to modify x0 externally to this program, so copy
    for i in eachindex(x0); x0[i] = log(x0[i]); end
    # Perform logarithmic transformation on MZR and dispersion parameters
    par = (values(fittable_params(mzr0))..., values(fittable_params(disp0))...)
    tf = (transforms(mzr0)..., transforms(disp0)...)
    free = SVector(free_params(mzr0)..., free_params(disp0)...)
    # Apply logarithmic transformations
    x0_mzrdisp = logtransform(par, tf)
    # Concatenate transformed stellar mass coefficients and MZR / disp parameters
    x0 = vcat(x0, x0_mzrdisp)

    # Set up options for the optimization
    # The InitialStatic(1.0,true) alphaguess helps to regularize the optimization and 
    # makes it less sensitive to initial x0.
    bfgs_struct = Optim.BFGS(; alphaguess=LineSearches.InitialStatic(1.0, true),
                             linesearch=LineSearches.HagerZhang())
    # The extended trace will contain the BFGS estimate of the inverse Hessian, aka the
    # covariance matrix, which we can use to make parameter uncertainty estimates
    bfgs_options = Optim.Options(; allow_f_increases=true, store_trace=true, extended_trace=true, kws...)
    # Calculate result
    function fg_map!(F, G, X)
        # Creating structs doesn't copy data so this should be free
        tmpstruct = MZROptimizer(mzr0, disp0, models, data, composite, logAge, metallicities, G, true)
        return -LogDensityProblems.logdensity_and_gradient(tmpstruct, X)[1]
    end
    function fg_mle!(F, G, X)
        # Creating structs doesn't copy data so this should be free
        tmpstruct = MZROptimizer(mzr0, disp0, models, data, composite, logAge, metallicities, G, false)
        return -LogDensityProblems.logdensity_and_gradient(tmpstruct, X)[1]
    end
    result_map = Optim.optimize(Optim.only_fg!(fg_map!), x0, bfgs_struct, bfgs_options)
    result_mle = Optim.optimize(Optim.only_fg!(fg_mle!), Optim.minimizer(result_map), bfgs_struct, bfgs_options)

    # Extract best-fit
    μ_map = deepcopy(Optim.minimizer(result_map))
    μ_mle = deepcopy(Optim.minimizer(result_mle))
    # Random sampling from the inverse Hessian approximation to the Gaussian covariance
    # matrix will use Distributions.MvNormal, which requires a
    # PDMat input, which includes the Cholesky decomposition of the matrix.
    # For sampling efficiency, we will construct this object here.
    invH_map = PDMat(hermitianpart(Optim.trace(result_map)[end].metadata["~inv(H)"]))
    invH_mle = PDMat(hermitianpart(Optim.trace(result_mle)[end].metadata["~inv(H)"]))
    # diagonal of invH gives vector of parameter variances -- standard error is sqrt
    σ_map = sqrt.(diag(invH_map))
    σ_mle = sqrt.(diag(invH_mle))
    # Correct for variable transformation for stellar mass coefficients
    for i in eachindex(μ_map, μ_mle, σ_map, σ_mle)[begin:Nbins]
        μ_map[i] = exp(μ_map[i])
        μ_mle[i] = exp(μ_mle[i])
        σ_map[i] = μ_map[i] * σ_map[i] # Account for transformation
        σ_mle[i] = μ_mle[i] * σ_mle[i] # Account for transformation
    end
    # Correct for variable transformation for MZR and dispersion parameters
    μ_map[Nbins+1:end] .= exptransform(μ_map[Nbins+1:end], SVector(tf))
    μ_mle[Nbins+1:end] .= exptransform(μ_mle[Nbins+1:end], SVector(tf))
    for (i, t) in enumerate(tf)
        idx = Nbins + i
        if free[i]
            if t == 1 # μ is always positive
                σ_map[idx] = μ_map[idx] * σ_map[idx]
                σ_mle[idx] = μ_mle[idx] * σ_mle[idx]
            elseif t == -1 # μ is always negative
                σ_map[idx] = -μ_map[idx] * σ_map[idx]
                σ_mle[idx] = -μ_mle[idx] * σ_mle[idx]
            # elseif t == 0 # μ can be positive or negative
            #     continue
            end
        else
            σ_map[idx] = 0 # Set uncertainty to 0 for fixed quantities
            σ_mle[idx] = 0
        end
    end
    
    return CompositeBFGSResult( BFGSResult(μ_map, σ_map, invH_map, result_map,
                                           update_params(mzr0, @view(μ_map[Nbins+1:Nbins+nparams(mzr0)])),
                                           update_params(disp0, @view(μ_map[Nbins+nparams(mzr0)+1:end]))),
                                BFGSResult(μ_mle, σ_mle, invH_mle, result_mle,
                                           update_params(mzr0, @view(μ_mle[Nbins+1:Nbins+nparams(mzr0)])),
                                           update_params(disp0, @view(μ_mle[Nbins+nparams(mzr0)+1:end]))) )
                                
    
end
# For models, data that do not follow the stacked data layout (see stack_models in fitting/utilities.jl)
fit_sfh(mzr0::AbstractMZR, disp0::AbstractDispersionModel, models::AbstractVector{<:AbstractMatrix{<:Number}}, data::AbstractMatrix{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}; kws...) = fit_sfh(mzr0, disp0, stack_models(models), vec(data), logAge, metallicities; kws...)



# HMC sampling routine; uses stacked data layout
# Use BFGS result to get initial position, Gaussian kinetic energy matrix
function sample_sfh(bfgs_result::CompositeBFGSResult, # ::NamedTuple{(:map, :mle), NTuple{2, T}},
                    models::AbstractMatrix{S},
                    data::AbstractVector{<:Number},
                    logAge::AbstractVector{<:Number},
                    metallicities::AbstractVector{<:Number},
                    Nsteps::Integer;
                    ϵ::Real = 0.05, # HMC step size
                    reporter = DynamicHMC.ProgressMeterReport(),
                    show_convergence::Bool=true,
                    composite::AbstractVector{<:Number}=Vector{S}(undef,length(data)),
                    rng::AbstractRNG=default_rng(),
                    kws...) where {S <: Number}

    # Will use MLE for best-fit values, MAP for invH
    MAP, MLE = bfgs_result.map, bfgs_result.mle
    # Best-fit values from optimization in transformed fitting variables
    x0 = Optim.minimizer(MLE.result)
    Zmodel, dispmodel = MLE.Zmodel, MLE.dispmodel
    
    # Get transformation parameters
    tf = (transforms(Zmodel)..., transforms(dispmodel)...)
    free = (free_params(Zmodel)..., free_params(dispmodel)...)
    if false in free
        @warn "sample_sfh is not optimized for use cases with fixed parameters."
    end

    # Setup
    instance = MZROptimizer(Zmodel, dispmodel, models, data, composite, logAge, metallicities,
                            Vector{S}(undef, length(unique(logAge)) + 3), true)
    warmup_state = DynamicHMC.initialize_warmup_state(rng, instance;
        q = x0, # Initial position vector
        κ = DynamicHMC.GaussianKineticEnergy(MAP.invH), # Kinetic energy
        ϵ = ϵ) # HMC step size
    sampling_logdensity = DynamicHMC.SamplingLogDensity(rng, instance, DynamicHMC.NUTS(), reporter)
    
    # Sample
    result = DynamicHMC.mcmc(sampling_logdensity, Nsteps, warmup_state)

    # Test convergence
    tree_stats = DynamicHMC.Diagnostics.summarize_tree_statistics(result.tree_statistics)
    show_convergence && display(tree_stats)
    if tree_stats.a_mean < 0.8
        @warn "Acceptance ratio for samples less than 80%, recommend re-running with smaller step size ϵ."
    end
    if tree_stats.termination_counts.divergence > (0.1 * Nsteps)
        @warn "More than 10% of samples diverged, recommend re-running with smaller step size ϵ."
    end

    # Transform samples
    exptransform_samples!(result.posterior_matrix, MLE.μ, tf, free)
    return result
end
# For models, data that do not follow the stacked data layout (see stack_models in fitting/utilities.jl)
sample_sfh(bfgs_result::CompositeBFGSResult, models::AbstractVector{<:AbstractMatrix{<:Number}}, data::AbstractMatrix{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, Nsteps::Integer; kws...) = sample_sfh(bfgs_result, stack_models(models), vec(data), logAge, metallicities, Nsteps; kws...)
