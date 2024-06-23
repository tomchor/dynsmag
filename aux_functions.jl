using Oceananigans.Operators

AG = Oceananigans.Grids.AbstractGrid
@inline ℱx²ᵟ(i, j, k, grid::AG{FT}, ϕ) where FT = @inbounds FT(0.5) * ϕ[i, j, k] + FT(0.25) * (ϕ[i-1, j, k] + ϕ[i+1, j,  k])
@inline ℱy²ᵟ(i, j, k, grid::AG{FT}, ϕ) where FT = @inbounds FT(0.5) * ϕ[i, j, k] + FT(0.25) * (ϕ[i, j-1, k] + ϕ[i,  j+1, k])
@inline ℱz²ᵟ(i, j, k, grid::AG{FT}, ϕ) where FT = @inbounds FT(0.5) * ϕ[i, j, k] + FT(0.25) * (ϕ[i, j, k-1] + ϕ[i,  j, k+1])

@inline ℱx²ᵟ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * f(i, j, k, grid, args...) + FT(0.25) * (f(i-1, j, k, grid, args...) + f(i+1, j, k, grid, args...))
@inline ℱy²ᵟ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * f(i, j, k, grid, args...) + FT(0.25) * (f(i, j-1, k, grid, args...) + f(i, j+1, k, grid, args...))
@inline ℱz²ᵟ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * f(i, j, k, grid, args...) + FT(0.25) * (f(i, j, k-1, grid, args...) + f(i, j, k+1, grid, args...))

@inline ℱxy²ᵟ(i, j, k, grid, f, args...)  = ℱy²ᵟ(i, j, k, grid, ℱx²ᵟ, f, args...)
@inline ℱyz²ᵟ(i, j, k, grid, f, args...)  = ℱz²ᵟ(i, j, k, grid, ℱy²ᵟ, f, args...)
@inline ℱxz²ᵟ(i, j, k, grid, f, args...)  = ℱz²ᵟ(i, j, k, grid, ℱz²ᵟ, f, args...)
@inline ℱxyz²ᵟ(i, j, k, grid, f, args...) = ℱz²ᵟ(i, j, k, grid, ℱxy²ᵟ, f, args...)


#####
##### Velocity gradients
#####

# Diagonal
@inline ∂x_u(i, j, k, grid, u) = ∂xᶜᶜᶜ(i, j, k, grid, u)
@inline ∂y_v(i, j, k, grid, v) = ∂yᶜᶜᶜ(i, j, k, grid, v)
@inline ∂z_w(i, j, k, grid, w) = ∂zᶜᶜᶜ(i, j, k, grid, w)

@inline ∂x_ū(i, j, k, grid, u) = ∂xᶜᶜᶜ(i, j, k, grid, ℱxyz²ᵟ, u)
@inline ∂y_v̄(i, j, k, grid, v) = ∂yᶜᶜᶜ(i, j, k, grid, ℱxyz²ᵟ, v)
@inline ∂z_w̄(i, j, k, grid, w) = ∂zᶜᶜᶜ(i, j, k, grid, ℱxyz²ᵟ, w)

# Off-diagonal
@inline ∂x_v(i, j, k, grid, v) = ∂xᶠᶠᶜ(i, j, k, grid, v)
@inline ∂x_w(i, j, k, grid, w) = ∂xᶠᶜᶜ(i, j, k, grid, w)

@inline ∂x_v̄(i, j, k, grid, v) = ∂xᶠᶠᶜ(i, j, k, grid, ℱxyz²ᵟ, v)
@inline ∂x_w̄(i, j, k, grid, w) = ∂xᶠᶜᶜ(i, j, k, grid, ℱxyz²ᵟ, w)

@inline ∂y_u(i, j, k, grid, u) = ∂yᶠᶠᶜ(i, j, k, grid, u)
@inline ∂y_w(i, j, k, grid, w) = ∂yᶜᶠᶜ(i, j, k, grid, w)

@inline ∂y_ū(i, j, k, grid, u) = ∂yᶠᶠᶜ(i, j, k, grid, ℱxyz²ᵟ, u)
@inline ∂y_w̄(i, j, k, grid, w) = ∂yᶜᶠᶜ(i, j, k, grid, ℱxyz²ᵟ, w)

@inline ∂z_u(i, j, k, grid, u) = ∂zᶠᶜᶠ(i, j, k, grid, u)
@inline ∂z_v(i, j, k, grid, v) = ∂zᶜᶠᶠ(i, j, k, grid, v)

@inline ∂z_ū(i, j, k, grid, u) = ∂zᶠᶜᶠ(i, j, k, grid, ℱxyz²ᵟ, u)
@inline ∂z_v̄(i, j, k, grid, v) = ∂zᶜᶠᶠ(i, j, k, grid, ℱxyz²ᵟ, v)

#####
##### Strain components
#####

# ccc strain components
@inline Σ₁₁(i, j, k, grid, u) = ∂x_u(i, j, k, grid, u)
@inline Σ₂₂(i, j, k, grid, v) = ∂y_v(i, j, k, grid, v)
@inline Σ₃₃(i, j, k, grid, w) = ∂z_w(i, j, k, grid, w)

@inline tr_Σ(i, j, k, grid, u, v, w) = Σ₁₁(i, j, k, grid, u) + Σ₂₂(i, j, k, grid, v) + Σ₃₃(i, j, k, grid, w)
@inline tr_Σ²(ijk...) = Σ₁₁(ijk...)^2 +  Σ₂₂(ijk...)^2 +  Σ₃₃(ijk...)^2

@inline Σ̄₁₁(i, j, k, grid, u) = ∂x_ū(i, j, k, grid, u)
@inline Σ̄₂₂(i, j, k, grid, v) = ∂y_v̄(i, j, k, grid, v)
@inline Σ̄₃₃(i, j, k, grid, w) = ∂z_w̄(i, j, k, grid, w)

@inline tr_Σ̄(i, j, k, grid, u, v, w) = Σ̄₁₁(i, j, k, grid, u) + Σ̄₂₂(i, j, k, grid, v) + Σ̄₃₃(i, j, k, grid, w)
@inline tr_Σ̄²(ijk...) = Σ̄₁₁(ijk...)^2 + Σ̄₂₂(ijk...)^2 + Σ̄₃₃(ijk...)^2

# ffc
@inline Σ₁₂(i, j, k, grid::AG{FT}, u, v) where FT = FT(0.5) * (∂y_u(i, j, k, grid, u) + ∂x_v(i, j, k, grid, v))
@inline Σ̄₁₂(i, j, k, grid::AG{FT}, u, v) where FT = FT(0.5) * (∂y_ū(i, j, k, grid, u) + ∂x_v̄(i, j, k, grid, v))

@inline Σ₁₂²(i, j, k, grid, u, v) = Σ₁₂(i, j, k, grid, u, v)^2
@inline Σ̄₁₂²(i, j, k, grid, u, v) = Σ̄₁₂(i, j, k, grid, u, v)^2


# fcf
@inline Σ₁₃(i, j, k, grid::AG{FT}, u, w) where FT = FT(0.5) * (∂z_u(i, j, k, grid, u) + ∂x_w(i, j, k, grid, w))
@inline Σ̄₁₃(i, j, k, grid::AG{FT}, u, w) where FT = FT(0.5) * (∂z_ū(i, j, k, grid, u) + ∂x_w̄(i, j, k, grid, w))

@inline Σ₁₃²(i, j, k, grid, u, w) = Σ₁₃(i, j, k, grid, u, w)^2
@inline Σ̄₁₃²(i, j, k, grid, u, w) = Σ̄₁₃(i, j, k, grid, u, w)^2

# cff
@inline Σ₂₃(i, j, k, grid::AG{FT}, v, w) where FT = FT(0.5) * (∂z_v(i, j, k, grid, v) + ∂y_w(i, j, k, grid, w))
@inline Σ̄₂₃(i, j, k, grid::AG{FT}, v, w) where FT = FT(0.5) * (∂z_v̄(i, j, k, grid, v) + ∂y_w̄(i, j, k, grid, w))

@inline Σ₂₃²(i, j, k, grid, v, w) = Σ₂₃(i, j, k, grid, v, w)^2
@inline Σ̄₂₃²(i, j, k, grid, v, w) = Σ̄₂₃(i, j, k, grid, v, w)^2

#####
##### Renamed functions for consistent function signatures
#####


@inline Σ₁₁(i, j, k, grid, u, v, w) = Σ₁₁(i, j, k, grid, u)
@inline Σ₂₂(i, j, k, grid, u, v, w) = Σ₂₂(i, j, k, grid, v)
@inline Σ₃₃(i, j, k, grid, u, v, w) = Σ₃₃(i, j, k, grid, w)

@inline Σ̄₁₁(i, j, k, grid, u, v, w) = Σ̄₁₁(i, j, k, grid, u)
@inline Σ̄₂₂(i, j, k, grid, u, v, w) = Σ̄₂₂(i, j, k, grid, v)
@inline Σ̄₃₃(i, j, k, grid, u, v, w) = Σ̄₃₃(i, j, k, grid, w)

@inline Σ₁₂(i, j, k, grid, u, v, w) = Σ₁₂(i, j, k, grid, u, v)
@inline Σ₁₃(i, j, k, grid, u, v, w) = Σ₁₃(i, j, k, grid, u, w)
@inline Σ₂₃(i, j, k, grid, u, v, w) = Σ₂₃(i, j, k, grid, v, w)

@inline Σ̄₁₂(i, j, k, grid, u, v, w) = Σ̄₁₂(i, j, k, grid, u, v)
@inline Σ̄₁₃(i, j, k, grid, u, v, w) = Σ̄₁₃(i, j, k, grid, u, w)
@inline Σ̄₂₃(i, j, k, grid, u, v, w) = Σ̄₂₃(i, j, k, grid, v, w)

# Symmetry relations
const Σ₂₁ = Σ₁₂
const Σ₃₁ = Σ₁₃
const Σ₃₂ = Σ₂₃

# Trace and squared strains
@inline tr_Σ²(ijk...) = Σ₁₁(ijk...)^2 +  Σ₂₂(ijk...)^2 +  Σ₃₃(ijk...)^2
@inline tr_Σ̄²(ijk...) = Σ̄₁₁(ijk...)^2 +  Σ̄₂₂(ijk...)^2 +  Σ̄₃₃(ijk...)^2

@inline Σ₁₂²(i, j, k, grid, u, v, w) = Σ₁₂²(i, j, k, grid, u, v)
@inline Σ₁₃²(i, j, k, grid, u, v, w) = Σ₁₃²(i, j, k, grid, u, w)
@inline Σ₂₃²(i, j, k, grid, u, v, w) = Σ₂₃²(i, j, k, grid, v, w)

@inline Σ̄₁₂²(i, j, k, grid, u, v, w) = Σ̄₁₂²(i, j, k, grid, u, v)
@inline Σ̄₁₃²(i, j, k, grid, u, v, w) = Σ̄₁₃²(i, j, k, grid, u, w)
@inline Σ̄₂₃²(i, j, k, grid, u, v, w) = Σ̄₂₃²(i, j, k, grid, v, w)

"Return the double dot product of strain at `ccc`."
@inline function ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)
    return (tr_Σ²(i, j, k, grid, u, v, w)
            + 2 * ℑxyᶜᶜᵃ(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 * ℑxzᶜᵃᶜ(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 * ℑyzᵃᶜᶜ(i, j, k, grid, Σ₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `ffc`."
@inline function ΣᵢⱼΣᵢⱼᶠᶠᶜ(i, j, k, grid, u, v, w)
    return (
                  ℑxyᶠᶠᵃ(i, j, k, grid, tr_Σ², u, v, w)
            + 2 *   Σ₁₂²(i, j, k, grid, u, v, w)
            + 2 * ℑyzᵃᶠᶜ(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 * ℑxzᶠᵃᶜ(i, j, k, grid, Σ₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `fcf`."
@inline function ΣᵢⱼΣᵢⱼᶠᶜᶠ(i, j, k, grid, u, v, w)
    return (
                  ℑxzᶠᵃᶠ(i, j, k, grid, tr_Σ², u, v, w)
            + 2 * ℑyzᵃᶜᶠ(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 *   Σ₁₃²(i, j, k, grid, u, v, w)
            + 2 * ℑxyᶠᶜᵃ(i, j, k, grid, Σ₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `cff`."
@inline function ΣᵢⱼΣᵢⱼᶜᶠᶠ(i, j, k, grid, u, v, w)
    return (
                  ℑyzᵃᶠᶠ(i, j, k, grid, tr_Σ², u, v, w)
            + 2 * ℑxzᶜᵃᶠ(i, j, k, grid, Σ₁₂², u, v, w)
            + 2 * ℑxyᶜᶠᵃ(i, j, k, grid, Σ₁₃², u, v, w)
            + 2 *   Σ₂₃²(i, j, k, grid, u, v, w)
            )
end



"Return the double dot product of strain at `ccc` on a 2δ test grid."
@inline function Σ̄ᵢⱼΣ̄ᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)
    return (tr_Σ̄²(i, j, k, grid, u, v, w)
            + 2 * ℑxyᶜᶜᵃ(i, j, k, grid, Σ̄₁₂², u, v, w)
            + 2 * ℑxzᶜᵃᶜ(i, j, k, grid, Σ̄₁₃², u, v, w)
            + 2 * ℑyzᵃᶜᶜ(i, j, k, grid, Σ̄₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `ffc`."
@inline function Σ̄ᵢⱼΣ̄ᵢⱼᶠᶠᶜ(i, j, k, grid, u, v, w)
    return (
                  ℑxyᶠᶠᵃ(i, j, k, grid, tr_Σ̄², u, v, w)
            + 2 *   Σ̄₁₂²(i, j, k, grid, u, v, w)
            + 2 * ℑyzᵃᶠᶜ(i, j, k, grid, Σ̄₁₃², u, v, w)
            + 2 * ℑxzᶠᵃᶜ(i, j, k, grid, Σ̄₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `fcf`."
@inline function Σ̄ᵢⱼΣ̄ᵢⱼᶠᶜᶠ(i, j, k, grid, u, v, w)
    return (
                  ℑxzᶠᵃᶠ(i, j, k, grid, tr_Σ̄², u, v, w)
            + 2 * ℑyzᵃᶜᶠ(i, j, k, grid, Σ̄₁₂², u, v, w)
            + 2 *   Σ̄₁₃²(i, j, k, grid, u, v, w)
            + 2 * ℑxyᶠᶜᵃ(i, j, k, grid, Σ̄₂₃², u, v, w)
            )
end

"Return the double dot product of strain at `cff`."
@inline function Σ̄ᵢⱼΣ̄ᵢⱼᶜᶠᶠ(i, j, k, grid, u, v, w)
    return (
                  ℑyzᵃᶠᶠ(i, j, k, grid, tr_Σ̄², u, v, w)
            + 2 * ℑxzᶜᵃᶠ(i, j, k, grid, Σ̄₁₂², u, v, w)
            + 2 * ℑxyᶜᶠᵃ(i, j, k, grid, Σ̄₁₃², u, v, w)
            + 2 *   Σ̄₂₃²(i, j, k, grid, u, v, w)
            )
end



@inline fψ_plus_gφ²(i, j, k, grid, f, ψ, g, φ) = (f(i, j, k, grid, ψ) + g(i, j, k, grid, φ))^2
function strain_rate_tensor_modulus_ccc(i, j, k, grid, u, v, w)
    Sˣˣ² = ∂xᶜᶜᶜ(i, j, k, grid, u)^2
    Sʸʸ² = ∂yᶜᶜᶜ(i, j, k, grid, v)^2
    Sᶻᶻ² = ∂zᶜᶜᶜ(i, j, k, grid, w)^2

    Sˣʸ² = ℑxyᶜᶜᵃ(i, j, k, grid, fψ_plus_gφ², ∂yᶠᶠᶜ, u, ∂xᶠᶠᶜ, v) / 4
    Sˣᶻ² = ℑxzᶜᵃᶜ(i, j, k, grid, fψ_plus_gφ², ∂zᶠᶜᶠ, u, ∂xᶠᶜᶠ, w) / 4
    Sʸᶻ² = ℑyzᵃᶜᶜ(i, j, k, grid, fψ_plus_gφ², ∂zᶜᶠᶠ, v, ∂yᶜᶠᶠ, w) / 4

    return √(Sˣˣ² + Sʸʸ² + Sᶻᶻ² + 2 * (Sˣʸ² + Sˣᶻ² + Sʸᶻ²))
end


@inline fψ̄_plus_gφ̄²(i, j, k, grid, f, ψ, g, φ) = (f(i, j, k, grid, ℱxyz²ᵟ, ψ) + g(i, j, k, grid, ℱxyz²ᵟ, φ))^2
function filtered_strain_rate_tensor_modulus_ccc(i, j, k, grid, u, v, w)
    Sˣˣ² = ∂xᶜᶜᶜ(i, j, k, grid, ℱxyz²ᵟ, u)^2
    Sʸʸ² = ∂yᶜᶜᶜ(i, j, k, grid, ℱxyz²ᵟ, v)^2
    Sᶻᶻ² = ∂zᶜᶜᶜ(i, j, k, grid, ℱxyz²ᵟ, w)^2

    Sˣʸ² = ℑxyᶜᶜᵃ(i, j, k, grid, fψ_plus_gφ², ∂yᶠᶠᶜ, u, ∂xᶠᶠᶜ, v) / 4
    Sˣᶻ² = ℑxzᶜᵃᶜ(i, j, k, grid, fψ_plus_gφ², ∂zᶠᶜᶠ, u, ∂xᶠᶜᶠ, w) / 4
    Sʸᶻ² = ℑyzᵃᶜᶜ(i, j, k, grid, fψ_plus_gφ², ∂zᶜᶠᶠ, v, ∂yᶜᶠᶠ, w) / 4

    return √(Sˣˣ² + Sʸʸ² + Sᶻᶻ² + 2 * (Sˣʸ² + Sˣᶻ² + Sʸᶻ²))
end


