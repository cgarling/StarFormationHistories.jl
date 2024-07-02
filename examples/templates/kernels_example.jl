import StarFormationHistories: GaussianPSFAsymmetric, GaussianPSFCovariant, gaussian_psf_covariant, addstar!, evaluate, parameters, histogram_pix
import StatsBase: fit, Histogram

# Set up for plotting
import PyPlot as plt
import PyPlot: @L_str # For LatexStrings
plt.rc("text", usetex=true)
plt.rc("font", family="serif", serif=["Computer Modern"], size=14)
plt.rc("figure", figsize=(5,5))
plt.rc("patch", linewidth=1, edgecolor="k", force_edgecolor=true)

# Example showing kernels where mags are {B,V,R} for three different cases of Hess
# diagram configuration.
# Y=B, X=B-V
# Y=V, X=B-V
# Y=R, X=B-V

npoints = 100_000
mags = ("B", "V", "R")
σ = (B=0.1, V=0.1, R=0.1) # Magnitude errors
centers = (B=20.0, V=19.0, R=18.0)
hist_size = (50, 100)


##################################################################################
# For mags {B,V,R}, test case y=B, x=B-V
yy = randn(npoints) .* σ.B .+ centers.B
xx = yy .- (randn(npoints) .* σ.V .+ centers.V)
# cov_mult = 1 for y=V and x=B-V, -1 for y=B and x=B-V, 0 for y=R and x=B-V
model = GaussianPSFCovariant(centers.B - centers.V, centers.B, σ.V, σ.B, -1.0, 1.0, 0.0)

fig,axs=plt.subplots(nrows=1, ncols=3, sharex=true, sharey=true, figsize=(15,5))
axs[1].set_ylabel("B")
axs[1].text(0.05,0.95,"MC Sample",transform=axs[1].transAxes,va="top",ha="left",c="white")
axs[2].text(0.05,0.95,"Kernel Model",transform=axs[2].transAxes,va="top",ha="left",c="white")
axs[3].text(0.05,0.95,"Residual",transform=axs[3].transAxes,va="top",ha="left",c="white")
fig.subplots_adjust(hspace=0.0,wspace=0.0)

# This works when comparing against the PSF evaluation
# but not the PRF (pixel-integrated) evaluation 
# hist1 = axs[1].hist2d(xx,yy; bins=(50,100), density=true)
# hist_centers = (x = range(hist1[2][1] + (hist1[2][2] - hist1[2][1])/2,
#                           hist1[2][end] - (hist1[2][2] - hist1[2][1])/2;
#                           length=length(hist1[2])-1),
#                 y = range(hist1[3][1] + (hist1[3][2] - hist1[3][1])/2,
#                       hist1[3][end] - (hist1[3][2] - hist1[3][1])/2;
#                       length=length(hist1[3])-1))
# nbins is only a suggestion, this function tries to use nice/round bin widths
# rather than give you the exact number of bins you asked for, so use explicit ranges
# hist1 = fit(Histogram, (xx, yy); closed=:left, nbins=(100,50))
hist1 = fit(Histogram, (xx, yy), (range(extrema(xx)...; length=hist_size[1]),
                                  range(extrema(yy)...; length=hist_size[2])); closed=:left)
hist1_data = hist1.weights
hist1_data = hist1_data ./ sum(hist1_data)
hist1_bins = (x = hist1.edges[1], y = hist1.edges[2])

# Get centers of the histogram bins
hist_centers = (x = range(hist1_bins.x[1] + (hist1_bins.x[2] - hist1_bins.x[1])/2,
                          hist1_bins.x[end] - (hist1_bins.x[2] - hist1_bins.x[1])/2;
                          length=length(hist1_bins.x)-1),
                y = range(hist1_bins.y[1] + (hist1_bins.y[2] - hist1_bins.y[1])/2,
                          hist1_bins.y[end] - (hist1_bins.y[2] - hist1_bins.y[1])/2;
                          length=length(hist1_bins.y)-1))

# Explicit evaluation
z = [gaussian_psf_covariant(i, j, (hist1_bins.x[2] - hist1_bins.x[1])/2, (hist1_bins.y[2] - hist1_bins.y[1])/2,
                            parameters(model)...) for i=hist_centers.x, j=hist_centers.y]
# Use addstar! instead
zz = Histogram(Tuple(hist1_bins), zeros(size(hist1_data)), :left, false)
addstar!(zz, model, (100,100))
zz = zz.weights # Choose matrix to display
# z ≈ zz # Explicit evaluation and addstar! result should be approximately the same

hist_im = axs[1].imshow(permutedims(hist1_data); aspect="auto", origin="lower",
                        extent=(extrema(hist1_bins.x)..., extrema(hist1_bins.y)...))
axs[1].set_ylim(reverse(axs[1].get_ylim()))
axs[2].imshow(permutedims(zz); aspect="auto", origin="lower",
              extent=(extrema(hist1_bins.x)..., extrema(hist1_bins.y)...), clim=hist_im.get_clim())
axs[3].imshow(permutedims(hist1_data .- zz); aspect="auto", origin="lower",
              extent=(extrema(hist1_bins.x)..., extrema(hist1_bins.y)...))
# Mark center point
for ax in axs
    ax.scatter(centers.B - centers.V, centers.B, c="k", marker="x")
    ax.set_xlabel("B-V")
end

fig.colorbar(hist_im, ax=axs[1:2], pad=0.0, fraction=0.15)

##################################################################################
# For mags {B,V,R}, test case y=V, x=B-V

yy = randn(npoints) .* σ.V .+ centers.V
xx = (randn(npoints) .* σ.B .+ centers.B) .- yy
# cov_mult = 1 for y=V and x=B-V, -1 for y=B and x=B-V, 0 for y=R and x=B-V
model = GaussianPSFCovariant(centers.B - centers.V, centers.V, σ.V, σ.B, 1.0, 1.0, 0.0)

fig,axs=plt.subplots(nrows=1, ncols=3, sharex=true, sharey=true, figsize=(15,5))
axs[1].set_ylabel("V")
axs[1].text(0.05,0.95,"MC Sample",transform=axs[1].transAxes,va="top",ha="left",c="white")
axs[2].text(0.05,0.95,"Kernel Model",transform=axs[2].transAxes,va="top",ha="left",c="white")
axs[3].text(0.05,0.95,"Residual",transform=axs[3].transAxes,va="top",ha="left",c="white")
fig.subplots_adjust(hspace=0.0,wspace=0.0)

hist1 = fit(Histogram, (xx, yy), (range(extrema(xx)...; length=hist_size[1]),
                                  range(extrema(yy)...; length=hist_size[2])); closed=:left)
hist1_data = hist1.weights
hist1_data = hist1_data ./ sum(hist1_data)
hist1_bins = (x = hist1.edges[1], y = hist1.edges[2])

# Get centers of the histogram bins
hist_centers = (x = range(hist1_bins.x[1] + (hist1_bins.x[2] - hist1_bins.x[1])/2,
                          hist1_bins.x[end] - (hist1_bins.x[2] - hist1_bins.x[1])/2;
                          length=length(hist1_bins.x)-1),
                y = range(hist1_bins.y[1] + (hist1_bins.y[2] - hist1_bins.y[1])/2,
                          hist1_bins.y[end] - (hist1_bins.y[2] - hist1_bins.y[1])/2;
                          length=length(hist1_bins.y)-1))

zz = Histogram(Tuple(hist1_bins), zeros(size(hist1_data)), :left, false)
addstar!(zz, model, (100,100))
zz = zz.weights # Choose matrix to display

hist_im = axs[1].imshow(permutedims(hist1_data); aspect="auto", origin="lower",
                        extent=(extrema(hist1_bins.x)..., extrema(hist1_bins.y)...))
axs[1].set_ylim(reverse(axs[1].get_ylim()))
axs[2].imshow(permutedims(zz); aspect="auto", origin="lower",
              extent=(extrema(hist1_bins.x)..., extrema(hist1_bins.y)...), clim=hist_im.get_clim())
axs[3].imshow(permutedims(hist1_data .- zz); aspect="auto", origin="lower",
              extent=(extrema(hist1_bins.x)..., extrema(hist1_bins.y)...))
# Mark center point
for ax in axs
    ax.scatter(centers.B - centers.V, centers.V, c="k", marker="x")
    ax.set_xlabel("B-V")
end

fig.colorbar(hist_im, ax=axs[1:2], pad=0.0, fraction=0.15)

##################################################################################
# For mags {B,V,R}, test case y=R, x=B-V

yy = randn(npoints) .* σ.R .+ centers.R
xx = (randn(npoints) .* σ.B .+ centers.B) .- (randn(npoints) .* σ.V .+ centers.V)
# cov_mult = 1 for y=V and x=B-V, -1 for y=B and x=B-V, 0 for y=R and x=B-V
# This does work, but try pixel-space kernel below
# model = GaussianPSFCovariant(centers.B - centers.V, centers.R,
#                              sqrt(σ.B^2 + σ.V^2), σ.R, 0.0, 1.0, 0.0)

fig,axs=plt.subplots(nrows=1, ncols=3, sharex=true, sharey=true, figsize=(15,5))
axs[1].set_ylabel("R")
axs[1].text(0.05,0.95,"MC Sample",transform=axs[1].transAxes,va="top",ha="left",c="white")
axs[2].text(0.05,0.95,"Kernel Model",transform=axs[2].transAxes,va="top",ha="left",c="white")
axs[3].text(0.05,0.95,"Residual",transform=axs[3].transAxes,va="top",ha="left",c="white")
fig.subplots_adjust(hspace=0.0,wspace=0.0)

hist1 = fit(Histogram, (xx, yy), (range(extrema(xx)...; length=hist_size[1]),
                                  range(extrema(yy)...; length=hist_size[2])); closed=:left)
hist1_data = hist1.weights
hist1_data = hist1_data ./ sum(hist1_data)
hist1_bins = (x = hist1.edges[1], y = hist1.edges[2])

# Pixel-space kernel for no covariance with equivalent σ widths
model = GaussianPSFAsymmetric(histogram_pix(centers.B - centers.V, hist1_bins.x),
                              histogram_pix(centers.R, hist1_bins.y),
                              sqrt(σ.B^2 + σ.V^2) / (hist1_bins.x[2] - hist1_bins.x[1]),
                              σ.R / (hist1_bins.y[2] - hist1_bins.y[1]),
                              1.0, 0.0)

# Get centers of the histogram bins
hist_centers = (x = range(hist1_bins.x[1] + (hist1_bins.x[2] - hist1_bins.x[1])/2,
                          hist1_bins.x[end] - (hist1_bins.x[2] - hist1_bins.x[1])/2;
                          length=length(hist1_bins.x)-1),
                y = range(hist1_bins.y[1] + (hist1_bins.y[2] - hist1_bins.y[1])/2,
                          hist1_bins.y[end] - (hist1_bins.y[2] - hist1_bins.y[1])/2;
                          length=length(hist1_bins.y)-1))

zz = Histogram(Tuple(hist1_bins), zeros(size(hist1_data)), :left, false)
addstar!(zz, model)
zz = zz.weights # Choose matrix to display

hist_im = axs[1].imshow(permutedims(hist1_data); aspect="auto", origin="lower",
                        extent=(extrema(hist1_bins.x)..., extrema(hist1_bins.y)...))
axs[1].set_ylim(reverse(axs[1].get_ylim()))
axs[2].imshow(permutedims(zz); aspect="auto", origin="lower",
              extent=(extrema(hist1_bins.x)..., extrema(hist1_bins.y)...), clim=hist_im.get_clim())
axs[3].imshow(permutedims(hist1_data .- zz); aspect="auto", origin="lower",
              extent=(extrema(hist1_bins.x)..., extrema(hist1_bins.y)...))
# Mark center point
for ax in axs
    ax.scatter(centers.B - centers.V, centers.R, c="k", marker="x")
    ax.set_xlabel("B-V")
end

fig.colorbar(hist_im, ax=axs[1:2], pad=0.0, fraction=0.15)
