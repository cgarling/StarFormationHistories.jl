import StarFormationHistories as SFH
import InitialMassFunctions: Kroupa2001
import DelimitedFiles: readdlm
import Printf: @sprintf
import QuadGK: quadgk

# Set up for plotting
import PyPlot as plt
import PyPlot: @L_str # For LatexStrings
plt.rc("text", usetex=true)
plt.rc("font", family="serif", serif=["Computer Modern"], size=14)
# This gets close but not quite
# plt.matplotlib.rcParams["axes.formatter.use_mathtext"] = true
# plt.rc("font", family="serif", serif=["cmr10"], size=14)
plt.rc("figure", figsize=(5,5))
plt.rc("patch", linewidth=1, edgecolor="k", force_edgecolor=true)
# https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html
plt.rc("image", interpolation="none")

# Bool whether to save figure as .svg or not; only save when building docs
savefig = ("DOCSBUILD" in keys(ENV)) && (ENV["DOCSBUILD"] == "true")

# Load example isochrone
# Path is relative to location of script, so use @__DIR__
isochrone, mag_names = readdlm(joinpath(@__DIR__, "../../data/isochrone.txt"), ' ',
                               Float64, '\n'; header=true)
# Unpack
m_ini = isochrone[:,1]
F090W = isochrone[:,2]
F150W = isochrone[:,3]

# Set distance modulus for example
distmod = 17.5 # Distance modulus

# Set bins for Hess diagram
edges = (range(-0.2, 1.4, length=75),
         range(distmod-0.0, distmod+10.0, length=200))

# Set total stellar mass to normalize template to
template_norm = 1e5

# Set binary model to use
binary_model = SFH.RandomBinaryPairs(0.4)

# Construct error and completeness functions
F090W_complete(m) = SFH.Martin2016_complete(m, 1.0, 28.5, 0.7)
F150W_complete(m) = SFH.Martin2016_complete(m, 1.0, 27.5, 0.7)
F090W_error(m) = min(SFH.exp_photerr(m, 1.03, 15.0, 36.0, 0.02), 0.4)
F150W_error(m) = min(SFH.exp_photerr(m, 1.03, 15.0, 35.0, 0.02), 0.4)

# Set IMF
imf = Kroupa2001(0.08, 100.0)
# imf = Kroupa2001(minimum(m_ini), 100.0)
# imf = Kroupa2001(extrema(m_ini)...)

# Construct template
template = SFH.partial_cmd_smooth(m_ini,
                                  [F090W, F150W],
                                  [F090W_error, F150W_error],
                                  2,
                                  [1,2],
                                  imf,
                                  [F090W_complete, F150W_complete]; 
                                  dmod=distmod,
                                  normalize_value=template_norm,
                                  edges=edges, binary_model=binary_model)

# Scale IMF and template_norm to only the valid range for m_ini;
# helps to prevent sampling binary companions outside the valid
# range of m_ini. You don't need to do this for the smooth Hess model
tscale = quadgk(x -> x * SFH.dispatch_imf(imf,x), extrema(m_ini)...)[1] / SFH.mean(imf)
# Sample analogous population; index [1] is sampled masses, dont need them
starcat_mags = SFH.generate_stars_mass(m_ini, [F090W, F150W],
                                       ["F090W", "F150W"], template_norm * tscale,
                                       Kroupa2001(extrema(m_ini)...);
                                       dist_mod=distmod,
                                       binary_model=binary_model)[2]

# Model photometric error and incompleteness
obs_mags = SFH.model_cmd(starcat_mags, [F090W_error, F150W_error],
                         [F090W_complete, F150W_complete])

# Concatenate into 2D matrix
obs_mags = reduce(hcat,obs_mags)

# Make Hess diagram
obs_hess = SFH.bin_cmd(view(obs_mags,1,:) .- view(obs_mags,2,:),
                       view(obs_mags,2,:), edges=edges).weights

# Residual / σ; sometimes called Pearson residual
signif = (permutedims(obs_hess) .- permutedims(template.weights)) ./
    sqrt.(permutedims(template.weights))
signif_plot = copy(signif)
signif_plot[permutedims(obs_hess) .== 0] .= NaN

############################################################################
# Plot

fig,axs=plt.subplots(nrows=1, ncols=4, sharex=true, sharey=true, figsize=(20,5))
fig.subplots_adjust(hspace=0.0, wspace=0.0)
# fig.suptitle(@sprintf("Stellar Mass: %.2e M\$_\\odot\$",template_norm))

axs[1].scatter(view(obs_mags,1,:) .- view(obs_mags,2,:), view(obs_mags,2,:),
               s=1, marker=".", c="k", alpha=0.1, rasterized=true,
               label="CMD-Sampled")
axs[1].text(0.05, 0.95,
            @sprintf("a) Sampled CMD\nM\$_*\$ = %.2e M\$_\\odot\$", template_norm),
            transform=axs[1].transAxes, va="top", ha="left")

im1 = axs[3].imshow(permutedims(template.weights), origin="lower", 
                    extent=(extrema(edges[1])..., extrema(edges[2])...), 
                    aspect="auto", cmap="Greys", rasterized=true,
                    norm=plt.matplotlib.colors.LogNorm(vmin=2.5 +
                        log10(template_norm/1e7)))
axs[3].text(0.05, 0.95, "c) Smooth Model",
            transform=axs[3].transAxes, va="top", ha="left")

axs[2].imshow(permutedims(obs_hess), origin="lower",
              extent=(extrema(edges[1])..., extrema(edges[2])...), 
              aspect="auto", cmap="Greys", rasterized=true,
              norm=plt.matplotlib.colors.LogNorm(vmin=2.5 +
                  log10(template_norm/1e7),vmax=im1.get_clim()[2]),
              label="CMD-Sampled")
axs[2].text(0.05, 0.95, "b) Sampled Hess Diagram", transform=axs[2].transAxes,
            va="top", ha="left")

im4 = axs[4].imshow(signif_plot, origin="lower",
                    extent=(extrema(edges[1])..., extrema(edges[2])...), 
                    aspect="auto", clim=(-2,2), rasterized=true)
axs[4].text(0.05, 0.95, L"d) (Obs - Model) / $\sigma$",
            transform=axs[4].transAxes, va="top", ha="left")

plot_isochrones = true
for i in eachindex(axs)
    axs[i].set_xlabel(L"F090W$-$F150W")
    if plot_isochrones & (i != 4) # Don't plot on residual
        axs[i].scatter(F090W .- F150W, F150W .+ distmod, marker=".", c="orange",
                       s=1, alpha=1.0)
    end
end
axs[1].set_ylabel("F150W")
axs[1].set_ylim(reverse(extrema(edges[2]))) 
axs[1].set_xlim(extrema(edges[1]))

# fraction prevents too much padding on right
fig.colorbar(im1, ax=axs[1:3], pad=0.005, fraction=0.075) 
fig.colorbar(im4, ax=axs[4], pad=0.015)
# println( "sum(obs) - sum(template): ", sum(obs_hess) .- sum(template.weights))
# println( "Difference of sums / sum: ", (sum(obs_hess) .- sum(template.weights)) ./
#     sum(obs_hess))
# println( "Sum of residuals: ", sum(abs, permutedims(obs_hess) .-
#     permutedims(template.weights)) )
# plt.savefig("template_compare.pdf", bbox_inches="tight")
if savefig
    plt.savefig(joinpath(@__DIR__, "template_compare_binaries.svg"), bbox_inches="tight")
end

#################################
# Distribution of σ discrepancies
import Distributions: pdf, Normal, Poisson
import StatsBase: mean, median, std

fig, ax1 = plt.subplots()
hist1 = ax1.hist(filter(isfinite, signif_plot), range=(-3,3), bins=25, density=true)
ax1.set_xlim(extrema(hist1[2]))
ax1.set_xlabel("Residual / Standard Deviation")
ax1.set_ylabel("PDF")
let xplot = first(hist1[2]):0.01:last(hist1[2])
    tmpgood   = signif |> filter(isfinite) |> filter(x -> first(hist1[2]) <= x <= last(hist1[2]))
    tmpgood_p = signif_plot |> filter(isfinite) |> filter(x -> first(hist1[2]) <= x <= last(hist1[2]))
    mean_σ = mean(tmpgood_p)
    med_σ = median(tmpgood_p)
    std_σ = std(tmpgood_p)
    ax1.plot(xplot, pdf.(Normal(0.0,1.0),xplot), label="Ideal")
    ax1.plot(xplot, pdf.(Normal(mean_σ,std_σ),xplot),
             label="Observed", c="magenta")
    ax1.axvline(mean_σ, c="k", ls="--")
    ax1.text(0.05, 0.95,
             # (@sprintf("Median: %.2f\n\$\\sigma\$: %.2f", med_σ, std_σ)),
             (@sprintf("Mean: %.2f\n\$\\sigma\$: %.2f", mean_σ, std_σ)),
             # (@sprintf("Mean: %.2f\nMedian: %.2f\n\$\\sigma\$: %.2f", mean_σ, med_σ, std_σ)),
             transform=ax1.transAxes, va="top", ha="left")
end
ax1.legend(loc="upper right")
# plt.savefig("sigma_distribution.pdf", bbox_inches="tight")
if savefig
    plt.savefig(joinpath(@__DIR__, "sigma_distribution_binaries.svg"), bbox_inches="tight")
end
if !plt.isinteractive()
    plt.close("all");
end
