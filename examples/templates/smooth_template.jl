import StarFormationHistories as SFH
import InitialMassFunctions: Kroupa2001
import DelimitedFiles: readdlm
import Printf: @sprintf

# Set up for plotting
import PyPlot as plt
import PyPlot: @L_str # For LatexStrings
plt.rc("text", usetex=true)
plt.rc("font", family="serif", serif=["Computer Modern"], size=14)
plt.rc("figure", figsize=(5,5))
plt.rc("patch", linewidth=1, edgecolor="k", force_edgecolor=true)
# Disable interactive plotting when running on CI or building docs
# if ("CI" in keys(ENV) && (ENV["CI"] == "true")) | (("DOCS_RUN" in keys(ENV)) && (ENV["DOCS_RUN"] == "true"))
#     ENV["MPLBACKEND"] = "agg"
#     plt.ioff()
# end

# Load example isochrone
# Path is relative to location of script, so use @__DIR__
isochrone, mag_names = readdlm(joinpath(@__DIR__, "../isochrone.txt"), ' ', Float64, '\n'; header=true)
# Unpack
m_ini = isochrone[:,1]
F090W = isochrone[:,2]
F150W = isochrone[:,3]

# Set distance modulus for example
distmod::Float64 = 25.0 # Distance modulus 

# Set bins for Hess diagram
edges = (range(-0.2, 1.2, length=75), range(distmod-6.0, distmod+5.0, length=100))

# Set total stellar mass to normalize template to
template_norm::Float64 = 1e7

# Construct error and completeness functions
F090W_complete(m) = SFH.Martin2016_complete(m,1.0,28.5,0.7)
F150W_complete(m) = SFH.Martin2016_complete(m,1.0,27.5,0.7)
F090W_error(m) = min( SFH.exp_photerr(m, 1.03, 15.0, 36.0, 0.02), 0.4 )
F150W_error(m) = min( SFH.exp_photerr(m, 1.03, 15.0, 35.0, 0.02), 0.4 )

# Set IMF
imf = Kroupa2001(0.08, 100.0)

# Construct template
template = SFH.partial_cmd_smooth( m_ini,
                                   [F090W, F150W],
                                   [F090W_error, F150W_error],
                                   2,
                                   [1,2],
                                   imf,
                                   [F090W_complete, F150W_complete]; 
                                   dmod = distmod,
                                   normalize_value = template_norm,
                                   edges = edges )

# Sample analogous population
starcat_mags = SFH.generate_stars_mass(m_ini, [F090W, F150W], ["F090W", "F150W"], template_norm, imf; dist_mod=distmod, binary_model=SFH.NoBinaries())[2] # index [1] is sampled masses, dont need them

# Model photometric error and incompleteness
obs_mags = SFH.model_cmd( starcat_mags, [F090W_error, F150W_error], [F090W_complete, F150W_complete])

# Concatenate into 2D matrix
obs_mags = reduce(hcat,obs_mags)

# Make Hess diagram
obs_hess = SFH.bin_cmd(view(obs_mags,1,:) .- view(obs_mags,2,:), view(obs_mags,2,:), edges=edges).weights

# using Plots
# gr()
# clims = (-16,-10)
# colorbar_ticks = log.( exp10.(-8:1.0:-3) )
# colorbar_tick_labels = [L"10^%$i" for i in -8:1:-3]
# cticks = (colorbar_ticks, colorbar_tick_labels)
# model_plot = heatmap(edges[1], edges[2], log.(permutedims(template.weights) ./ template_norm), yflip=true, c=cgrad(:greys, rev=true), colorbar_ticks=cticks, clims=clims)
# # model_plot = heatmap(template, transpose_z=true)
# plot(model_plot; layout=grid(2,2), size=(700,700))

# Residual / σ; sometimes called Pearson residual
signif = (permutedims(obs_hess) .- permutedims(template.weights)) ./ sqrt.(permutedims(template.weights))
signif[permutedims(obs_hess) .== 0] .= NaN

# Plot
fig,axs=plt.subplots(nrows=1,ncols=4,sharex=true,sharey=true,figsize=(20,5))
fig.subplots_adjust(hspace=0.0,wspace=0.0)
fig.suptitle(@sprintf("Stellar Mass: %.2e M\$_\\odot\$",template_norm))

axs[1].scatter(view(obs_mags,1,:) .- view(obs_mags,2,:), view(obs_mags,2,:), s=1, marker=".", c="k", alpha=0.05, rasterized=true, label="CMD-Sampled")
axs[1].text(0.1,0.9,"Sampled CMD",transform=axs[1].transAxes)

im1 = axs[3].imshow(permutedims(template.weights), origin="lower", 
                    extent=(extrema(edges[1])..., extrema(edges[2])...), 
                    aspect="auto", cmap="Greys", norm=plt.matplotlib.colors.LogNorm(vmin=2.5 + log10(template_norm/1e7)), rasterized=true)
axs[3].text(0.1,0.9,"Smooth Model",transform=axs[3].transAxes)

axs[2].imshow(permutedims(obs_hess), origin="lower", 
              extent=(extrema(edges[1])..., extrema(edges[2])...), 
              aspect="auto", cmap="Greys", norm=plt.matplotlib.colors.LogNorm(vmin=2.5 + log10(template_norm/1e7),vmax=im1.get_clim()[2]), rasterized=true, label="CMD-Sampled")
axs[2].text(0.1,0.9,"Sampled Hess Diagram",transform=axs[2].transAxes)

im4 = axs[4].imshow( signif, 
                     origin="lower", extent=(extrema(edges[1])..., extrema(edges[2])...), 
                     aspect="auto", clim=(-2,2), rasterized=true)
axs[4].text(0.1,0.9,L"(Obs - Model) / $\sigma$",transform=axs[4].transAxes)

plot_isochrones::Bool = true
for i in eachindex(axs)
    axs[i].set_xlabel(L"F090W$-$F150W")
    if plot_isochrones & (i != 4) # Don't plot on residual
        axs[i].scatter(F090W .- F150W, F150W .+ distmod, marker=".", c="orange", s=1, alpha=0.3)
    end
end
axs[1].set_ylabel("F150W")
axs[1].set_ylim(reverse(extrema(edges[2]))) 
axs[1].set_xlim(extrema(edges[1]))

fig.colorbar(im1, ax=axs[1:3], pad=0.005, fraction=0.075) # fraction prevents too much padding on right
fig.colorbar(im4, ax=axs[4], pad=0.015)
# println( "sum(obs) - sum(template): ", sum(obs_hess) .- sum(template.weights))
# println( "Difference of sums / sum: ", (sum(obs_hess) .- sum(template.weights)) ./ sum(obs_hess))
# println( "Sum of residuals: ", sum(abs, permutedims(obs_hess) .- permutedims(template.weights)) )
plt.savefig(joinpath(@__DIR__,"template_compare.svg"), bbox_inches="tight")

#################################
# Distribution of σ discrepancies
import Distributions: pdf, Normal, Poisson
import StatsBase: mean
fig, ax1 = plt.subplots()
hist1 = ax1.hist(filter(isfinite, signif), range=(-4,4), bins=25, density=true)
ax1.set_xlim( extrema(hist1[2]) )
ax1.set_xlabel("Residual / Standard Deviation")
ax1.set_ylabel("PDF")
let xplot = first(hist1[2]):0.01:last(hist1[2])
    # mean_σ = mean(filter(Base.Fix1(<,-5), filter(isfinite,signif)))
    mean_σ = mean(filter(isfinite,signif))
    ax1.plot( xplot, pdf.(Normal(mean_σ,1.0),xplot))
    ax1.axvline(mean_σ, c="k", ls="--")
end
plt.savefig(joinpath(@__DIR__,"sigma_distribution.svg"), bbox_inches="tight")

