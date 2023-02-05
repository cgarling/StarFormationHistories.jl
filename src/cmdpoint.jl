import Distributions: Normal
import Statistics: cov, mean, median, std, var

struct CMDPoint{T <: Real}
    x0::T
    y0::T
    σx::T
    σy::T
    A::T
    B::T
    function CMDPoint(x0::Real,y0::Real,σx::Real,σy::Real)
        T = promote(x0,y0,σx,σy)
        T_type = eltype(T)
        new{T_type}(T[1],T[2],T[3],T[4],one(T_type),zero(T_type))
    end
    function CMDPoint(x0::Real,y0::Real,σx::Real,σy::Real,A::Real)
        T = promote(x0,y0,σx,σy,A)
        T_type = eltype(T)
        new{T_type}(T[1],T[2],T[3],T[4],T[5],zero(T_type))
    end
    function CMDPoint(x0::Real,y0::Real,σx::Real,σy::Real,A::Real,B::Real)
        T = promote(x0,y0,σx,σy,A,B)
        new{eltype(T)}(T...)
    end
end
Base.Broadcast.broadcastable(m::CMDPoint) = Ref(m)
parameters(model::CMDPoint) = (model.x0, model.y0, model.σx, model.σy, model.A, model.B)
function Base.size(model::CMDPoint)
    σx,σy = model.σx,model.σy
    return (ceil(Int,σx) * 10, ceil(Int,σy) * 10)
end
centroid(model::CMDPoint) = (model.x0, model.y0)
function cmdpoint(x::Real,y::Real,x0::Real,y0::Real,σx::Real,σy::Real,A::Real,B::Real)
    δy = y - y0
    x0 = y0 - x0 + δy # This assumes that, e.g., y=B and x=B-V 
    # x0 = -(y0 - x0 + δy) # This assumes that, e.g., y=V and x=B-V 
    δx = x - x0
    inv(σy * σx * 2π) * exp( -(δy/σy)^2 / 2 ) * exp( -(δx/σx)^2 / 2 )
end
evaluate(model::CMDPoint, x::Real, y::Real) = 
    cmdpoint(x, y, parameters(model)...)

#################################################################################
# Test code

yy = rand(Normal(20.0,0.05), 100000)
xx = yy .- rand(Normal(19.0,0.05),100000)
# xx = rand(Normal(19.0,0.05),100000) .- yy
# xx = rand(Normal(20.0,0.05), 100000) .- rand(Normal(19.0,0.05),100000)

fig,axs=plt.subplots(nrows=1,ncols=3,sharex=true,sharey=true,figsize=(15,6))
fig.subplots_adjust(hspace=0.0,wspace=0.0)
hist1 = axs[1].hist2d(xx,yy; bins=(100,100), density=true)
axs[1].set_ylim( reverse(axs[1].get_ylim()) )

# model = CMDPoint(19.0,20.0,0.05,0.05)
model = CMDPoint(19.0,20.0,0.05,0.05)
zz = [ evaluate(model, i, j) for i=hist1[2],j=hist1[3] ]
axs[2].imshow(zz; aspect="auto", origin="lower", extent=(extrema(hist1[2])..., extrema(hist1[3])...), clim=hist1[4].get_clim() )
axs[3].imshow(hist1[1] .- zz[begin:end-1,begin:end-1]; aspect="auto", origin="lower", extent=(extrema(hist1[2])..., extrema(hist1[3])...))
fig.colorbar(hist1[4], ax=axs[2], pad=0.0, fraction=0.2)

# We can do this with a 2x2 covariance matrix;
# for y=g, x=g-r: cov=[σg² + σr^2 σg^2; σg^2 σg^2]
# for y=r, x=g-r: cov=[σg² + σr^2 -σr^2; -σr^2 σr^2]
# for y=i, x=g-r: cov=[σg² + σi^2 0; 0 σi^2]

########################################################################################
# Set up the covariant gaussian
import StaticArrays: SMatrix, SVector
import LinearAlgebra: det, transpose

# covmat(σx,σy) = SMatrix{2,2}(σx, 0, 0, σy)
struct Gaussian2D{T <: Real, S <: AbstractMatrix{<:Real}}
    x0::T
    y0::T
    Σ::S
    A::T
    B::T
    # function Gaussian2D(x0::Real,y0::Real,σx::Real,σy::Real)
    #     T = promote(x0,y0,σx,σy)
    #     T_type = eltype(T)
    #     new{T_type}(T[1],T[2],covmat(T[3],T[4]),one(T_type),zero(T_type))
    # end
    # function Gaussian2D(x0::Real,y0::Real,σx::Real,σy::Real,A::Real)
    #     T = promote(x0,y0,σx,σy,A)
    #     T_type = eltype(T)
    #     new{T_type}(T[1],T[2],covmat(T[3],T[4]),T[5],zero(T_type))
    # end
    # function Gaussian2D(x0::Real,y0::Real,covariance::AbstractMatrix{<:Real},A::Real,B::Real)
    #     T = promote(x0,y0,covariance,A,B)
    #     new{eltype(T)}(T...)
    # end
    # function Gaussian2D(x0::Real,y0::Real,σx²::Real,σxσy::Real,σyσx::Real,σy²::Real)
    function Gaussian2D(x0::Real, y0::Real, Σ::AbstractMatrix{<:Real})
        T = promote(x0, y0)
        T_type = eltype(T)
        new{T_type,typeof(Σ)}(T[1], T[2], Σ, one(T_type), zero(T_type))
    end
    function Gaussian2D(x0::Real, y0::Real, Σ::AbstractMatrix{<:Real}, A::Real)
        T = promote(x0, y0, A)
        T_type = eltype(T)
        new{T_type,typeof(Σ)}(T[1], T[2], Σ, T[3], zero(T_type))
    end
    function Gaussian2D(x0::Real, y0::Real, Σ::AbstractMatrix{<:Real}, A::Real, B::Real)
        T = promote(x0,y0,A,B)
        new{eltype(T),typeof(Σ)}(T[1], T[2], Σ, T[3], T[4])
    end
end
Base.Broadcast.broadcastable(m::Gaussian2D) = Ref(m)
parameters(model::Gaussian2D) = (model.x0, model.y0, model.Σ, model.A, model.B)
function Base.size(model::Gaussian2D)
    # σx,σy = diag(model.Σ)
    # return (ceil(Int,σx) * 10, ceil(Int,σy) * 10)
    Σ = model.Σ
    return (ceil(Int,sqrt(first(Σ))) * 10,
            ceil(Int,sqrt(last(Σ))) * 10)
end
centroid(model::Gaussian2D) = (model.x0, model.y0)
@inline function gauss2D(x::Real,y::Real,x0::Real,y0::Real,Σ::AbstractMatrix{<:Real},A::Real,B::Real)
    @assert axes(Σ) == (1:2, 1:2)
    detΣ = Σ[1] * Σ[4] - Σ[2] * Σ[3] # 2x2 Matrix determinant
    invdetΣ = inv(detΣ)
    invΣ = SMatrix{2,2}(Σ[4]*invdetΣ, -Σ[3]*invdetΣ, -Σ[2]*invdetΣ, Σ[1]*invdetΣ) # 2x2 Matrix inverse
    δx = SVector{2}( x-x0, y-y0 )
    return exp( -transpose(δx) * invΣ * δx / 2) / 2π / sqrt(detΣ)
end
@inline function gauss2D(x::Real,y::Real,x0::Real,y0::Real,Σ::SMatrix{2,2,<:Real,4},A::Real,B::Real)
    detΣ = det(Σ) 
    invΣ = inv(Σ) 
    δx = SVector{2}( x-x0, y-y0 )
    return exp( -transpose(δx) * invΣ * δx / 2) / 2π / sqrt(detΣ)
end
# Gauss-Legendre integration over [x-0.5,x+0.5] and [y-0.5,y+0.5] or just half of the regular gauss-legendre intervals.
const legendre_x_halfpix3 = SVector{3,Float64}(-0.3872983346207417, 0.0, 0.3872983346207417) 
# const legendre_w_halfpix3 = SVector{3,Float64}(0.5555555555555556,0.8888888888888889,0.5555555555555556)
const legendre_w_halfpix3 = SVector{3,Float64}(0.2777777777777778,0.4444444444444444,0.2777777777777778)
const legendre_x_halfpix5 = SVector{5,Float64}(-0.453089922969332, -0.2692346550528415, 0.0, 0.2692346550528415, 0.453089922969332) 
const legendre_w_halfpix5 = SVector{5,Float64}(0.11846344252809454, 0.23931433524968324, 0.28444444444444444, 0.23931433524968324, 0.11846344252809454)
# @inline gauss2d_integral_halfpix(x::Real,y::Real,x0::Real,y0::Real,Σ::AbstractMatrix{<:Real},A::Real,B::Real) = sum( legendre_w_halfpix5[i] * legendre_w_halfpix5[j] * gauss2D(x + legendre_x_halfpix5[i], y + legendre_x_halfpix5[j], x0, y0, Σ, A, B) for i=1:5,j=1:5 )
# @inline gauss2d_integral_halfpix(x::Real,y::Real,x0::Real,y0::Real,Σ::AbstractMatrix{<:Real},A::Real,B::Real) = sum( legendre_w_halfpix3[i] * legendre_w_halfpix3[j] * gauss2D(x + legendre_x_halfpix3[i], y + legendre_x_halfpix3[j], x0, y0, Σ, A, B) for i=1:3,j=1:3 )
@inline function gauss2d_integral_halfpix(x::Real,y::Real,x0::Real,y0::Real,Σ::AbstractMatrix{<:Real},A::Real,B::Real)
    result = 0.0
    for i=1:3, j=1:3
        @inbounds result += legendre_w_halfpix3[i] * legendre_w_halfpix3[j] * gauss2D(x + legendre_x_halfpix3[i], y + legendre_x_halfpix3[j], x0, y0, Σ, A, B)
    end
    return result
end

evaluate(model::Gaussian2D, x::Real, y::Real) = 
    # gauss2D(x, y, parameters(model)...)
    # The Gauss-Legendre integration is 25x slower for a few percent precision increase.
    # Might keep if making the templates ends up not taking very long. 
    gauss2d_integral_halfpix(x, y, parameters(model)...)


######################################################################################
# Test code
σx = 0.05
σy = 0.07
yy = rand(Normal(20.0,σy), 100000)
xx = yy .- rand(Normal(19.0,σx),100000)
# xx = rand(Normal(19.0,σx),100000) .- yy
# xx = rand(Normal(20.0,σx), 100000) .- rand(Normal(19.0,0.05),100000)

fig,axs=plt.subplots(nrows=1,ncols=3,sharex=true,sharey=true,figsize=(15,6))
fig.subplots_adjust(hspace=0.0,wspace=0.0)
hist1 = axs[1].hist2d(xx,yy; bins=(100,100), density=true)
axs[1].set_ylim( reverse(axs[1].get_ylim()) )

# This model is for y=g; x=g-r
# model = Gaussian2D(1.0, 20.0, SMatrix{2,2}(σx^2+σy^2,σy^2,σy^2,σy^2))
# This model is for y=r; x=g-r
model = Gaussian2D(-1.0, 20.0, SMatrix{2,2}(σx^2+σy^2,-σy^2,-σy^2,σy^2)) 
# model = Gaussian2D(-1.0, 20.0, cov([xx yy]))
zz = [ evaluate(model, i, j) for i=hist1[2],j=hist1[3] ]
axs[2].imshow(zz; aspect="auto", origin="lower", extent=(extrema(hist1[2])..., extrema(hist1[3])...), clim=hist1[4].get_clim() )
# fig.colorbar(hist1[4], ax=axs[2], pad=0.0, fraction=0.2)
hist3 = axs[3].imshow(hist1[1] .- zz[begin:end-1,begin:end-1]; aspect="auto", origin="lower", extent=(extrema(hist1[2])..., extrema(hist1[3])...))
fig.colorbar(hist3, ax=axs[3], pad=0.0, fraction=0.2)
