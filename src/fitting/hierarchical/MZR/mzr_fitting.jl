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
    mzrpar = npar(mzr0)
    # Number of fittable parameters in metallicity dispersion model;
    # in G, these come after the MZR parameters
    disppar = npar(disp0)

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

        # Require that the logAge values be sorted from largest lookback time (earliest)
        # to smallest lookback time (most recent).
        # Required for computing cumulative stellar mass later.
        unique_logAge = unique(logAge)
        @assert issorted(unique_logAge; rev=true)
        # Cumulative stellar mass, from earliest time to most recent
        cum_mstar = cumsum(@view(variables[begin:end-(mzrpar+disppar)]))
    
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
        # Calculate cumulative sum for sum_{j=0}^{j=j^\prime}
        cum_drjk_dRj = reverse!(cumsum(reverse!(ksum_drjk_dRj)))

        # Zero-out gradient vector in preparation to accumulate sums
        G .= zero(eltype(G))
        
        # Add in the j^\prime \neq j terms
        for i in eachindex(cum_drjk_dRj)[begin+1:end]
            G[i-1] -= cum_drjk_dRj[i]
        end

        # Loop over j
        for i in eachindex(unique_logAge)
            A = A_vec[i]
            idxs = jidx_inv[i]
            # Add the j^\prime == j term
            G[i] -= sum(fullG[idx] * (coeffs[idx] / variables[i] +
                (dAjk_dRj[idx] - (ksum_dAjk_dRj[i] * tmp_coeffs_vec[idx] / A)) *
                variables[i] / A) for idx in jidx_inv[i])
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
