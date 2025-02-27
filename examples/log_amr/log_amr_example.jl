# import StarFormationHistories: Z_from_MH, MH_from_Z, calculate_coeffs_logamr, calculate_αβ_logamr
using StarFormationHistories: Z_from_MH, MH_from_Z, LogarithmicAMR, GaussianDispersion,
    calculate_coeffs, fittable_params
using Plots
gr()

# Set up a grid for logAge and [M/H]
unique_logAge=8.0:0.01:10.13
unique_MH=-3.0:0.01:-0.25
logAge = repeat(unique_logAge; inner=length(unique_MH))
MH = repeat(unique_MH; outer=length(unique_logAge))

# Set coefficients for the logarithmic age-metallicity relation
T_max::Float64 = 13.7 # exp10(maximum(unique_logAge)-9)
MH_model = LogarithmicAMR((-0.8, 0.0), (-2.5, 13.7), T_max)
α, β = fittable_params(MH_model)
σ::Float64 = 0.2 # Metallicity spread at fixed logAge; Units of dex
disp_model = GaussianDispersion(σ)
# Calculate the relative weights such that the sum of all coefficients
# across a single logAge entry logAge[i] is 1.
relative_weights = calculate_coeffs(MH_model, disp_model, ones(length(unique_logAge)), logAge, MH)
# Reshape vector into matrix for display
relweights_matrix = reshape(relative_weights,(length(unique_MH), length(unique_logAge)))
# <Z>(unique_logAge) for plotting
plot_Z = α .* (T_max .- exp10.(unique_logAge)./1e9) .+ β

# xbins = vcat(unique_MH .- step(unique_MH)/2,last(unique_MH) + step(unique_MH)/2)
# ybins = vcat(unique_logAge .- step(unique_logAge)/2,last(unique_logAge) + step(unique_logAge)/2)
l = @layout [ a{0.33h} ; b{0.33h}; c{0.33h} ]
p1 = heatmap(exp10.(unique_logAge) ./ 1e9, Z_from_MH.(unique_MH), relweights_matrix; xlim=extrema(exp10.(unique_logAge) ./ 1e9), ylim = extrema(Z_from_MH.(unique_MH)), xlabel="Lookback Time [Gyr]", ylabel="Metal Mass Fraction (Z)")
plot!(exp10.(unique_logAge) ./ 1e9, plot_Z; c="black", ls=:dash)
p2 = heatmap(unique_logAge, unique_MH, relweights_matrix; xlim=extrema(unique_logAge), ylim=extrema(unique_MH), xlabel="log10(age [Gyr])", ylabel="[M/H]")
plot!(unique_logAge, MH_from_Z.(plot_Z), c="black", ls=:dash)
p3 = heatmap(exp10.(unique_logAge) ./ 1e9, unique_MH, relweights_matrix; xlabel="Lookback Time [Gyr]", ylabel="[M/H]", xlim=extrema(exp10.(unique_logAge) ./ 1e9), ylim=extrema(unique_MH))
plot!(exp10.(unique_logAge) ./ 1e9, MH_from_Z.(plot_Z); c="black", ls=:dash)
plot(p1, p2, p3; layout=l, size=size=(500,900), left_margin=(6,:mm), right_margin=(5,:mm), minorticks=true, tick_direction=:out, legend=false, colorbar=true, xflip=true)
