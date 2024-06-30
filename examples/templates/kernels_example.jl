import StarFormationHistories: GaussianPSFAsymmetric, GaussianPSFCovariant, addstar!, evaluate
import StatsBase: Histogram

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


##################################################################################
# For mags {B,V,R}, test case y=B, x=B-V
yy = randn(npoints) .* σ.B .+ centers.B
xx = yy .- (randn(npoints) .* σ.V .+ centers.V)
# cov_mult = 1 for y=V and x=B-V, -1 for y=B and x=B-V, 0 for y=R and x=B-V
model = GaussianPSFCovariant(centers.B - centers.V, centers.B, σ.V, σ.B, -1.0, 1.0, 0.0)

fig,axs=plt.subplots(nrows=1, ncols=3, sharex=true, sharey=true, figsize=(15,6))
fig.subplots_adjust(hspace=0.0,wspace=0.0)
hist1 = axs[1].hist2d(xx,yy; bins=(50,100), density=true)

# z = [evaluate(model, i, j) for i=hist1[2],j=hist1[3]]
zz = Histogram((range(first(hist1[2]), last(hist1[2]); length=length(hist1[2])),
                range(first(hist1[3]), last(hist1[3]); length=length(hist1[3]))),
               zeros(length(hist1[2])-1, length(hist1[3])-1), :left, false)
addstar!(zz, model, (reduce(-,reverse(extrema(hist1[2]))),
                     reduce(-,reverse(extrema(hist1[3])))))

axs[1].set_ylim(reverse(axs[1].get_ylim()))
# axs[2].imshow(permutedims(z); aspect="auto", origin="lower", extent=(extrema(hist1[2])..., extrema(hist1[3])...), clim=hist1[4].get_clim())
# axs[3].imshow(permutedims(hist1[1] .- z[begin:end-1,begin:end-1]); aspect="auto", origin="lower", extent=(extrema(hist1[2])..., extrema(hist1[3])...))
axs[2].imshow(permutedims(zz.weights); aspect="auto", origin="lower",
              extent=(extrema(hist1[2])..., extrema(hist1[3])...), clim=hist1[4].get_clim())
axs[3].imshow(permutedims(hist1[1] .- zz.weights); aspect="auto", origin="lower",
              extent=(extrema(hist1[2])..., extrema(hist1[3])...))
# Mark center point
for ax in axs
    ax.scatter(centers.B - centers.V, centers.B, c="k", marker="x")
end

fig.colorbar(hist1[4], ax=axs[1:2], pad=0.0, fraction=0.1)

##################################################################################
# For mags {B,V,R}, test case y=V, x=B-V

##################################################################################
# For mags {B,V,R}, test case y=R, x=B-V
