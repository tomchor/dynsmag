using Oceananigans
using Oceananigans.Operators: volume
using Statistics
using Oceanostics.ProgressMessengers: BasicTimeMessenger
#using Oceanostics.FlowDiagnostics: strain_rate_tensor_modulus_ccc

#+++ get model
grid = RectilinearGrid(size=(32, 32, 32), extent=(2π, 2π, 2π), topology=(Periodic, Periodic, Periodic))
model = NonhydrostaticModel(; grid, timestepper = :RungeKutta3, advection = UpwindBiasedFifthOrder(), closure = SmagorinskyLilly())
u, v, w = model.velocities
uᵢ = rand(size(u)...); vᵢ = rand(size(v)...)
uᵢ .-= mean(uᵢ); vᵢ .-= mean(vᵢ)
set!(model, u=uᵢ, v=vᵢ)
#---


AG = Oceananigans.Grids.AbstractGrid
@inline ℱx²ᵟ(i, j, k, grid::AG{FT}, ϕ) where FT = @inbounds FT(0.5) * ϕ[i, j, k] + FT(0.25) * (ϕ[i-1, j, k] + ϕ[i+1, j,  k])
@inline ℱy²ᵟ(i, j, k, grid::AG{FT}, ϕ) where FT = @inbounds FT(0.5) * ϕ[i, j, k] + FT(0.25) * (ϕ[i, j-1, k] + ϕ[i,  j+1, k])
@inline ℱz²ᵟ(i, j, k, grid::AG{FT}, ϕ) where FT = @inbounds FT(0.5) * ϕ[i, j, k] + FT(0.25) * (ϕ[i, j, k-1] + ϕ[i,  j, k+1])

@inline ℱx²ᵟ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * f(i, j, k, grid, args...) + FT(0.25) * (f(i-1, j, k, grid, args...) + f(i+1, j, k, grid, args...))
@inline ℱy²ᵟ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * f(i, j, k, grid, args...) + FT(0.25) * (f(i, j-1, k, grid, args...) + f(i, j+1, k, grid, args...))
@inline ℱz²ᵟ(i, j, k, grid::AG{FT}, f::F, args...) where {FT, F<:Function} = FT(0.5) * f(i, j, k, grid, args...) + FT(0.25) * (f(i, j, k-1, grid, args...) + f(i, j, k+1, grid, args...))

@inline ℱxy²ᵟ(i, j, k, grid, f, args...) = ℱy²ᵟ(i, j, k, grid, ℱx²ᵟ, f, args...)
@inline ℱyz²ᵟ(i, j, k, grid, f, args...) = ℱz²ᵟ(i, j, k, grid, ℱy²ᵟ, f, args...)
@inline ℱxz²ᵟ(i, j, k, grid, f, args...) = ℱz²ᵟ(i, j, k, grid, ℱz²ᵟ, f, args...)
@inline ℱxyz²ᵟ(i, j, k, grid, f, args...) = ℱz²ᵟ(i, j, k, grid, ℱxy²ᵟ, f, args...)


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

ϕ_times_fψ(i, j, k, grid, ϕ, f::Function, ψ) = ϕ
function MᵢⱼMᵢⱼ_ccc(i, j, k, grid, u, v, w, p)
    S_abs = strain_rate_tensor_modulus_ccc(i, j, k, grid, u, v, w)
    S̄_abs = filtered_strain_rate_tensor_modulus_ccc(i, j, k, grid, u, v, w)

    var"⟨|S|S₁₁⟩" = ℱxyz²ᵟ(i, j, k, grid, ϕ_times_fψ, S_abs, ∂xᶜᶜᶜ, u)
    var"⟨|S|S₂₂⟩" = ℱxyz²ᵟ(i, j, k, grid, ϕ_times_fψ, S_abs, ∂yᶜᶜᶜ, v)
    var"⟨|S|S₃₃⟩" = ℱxyz²ᵟ(i, j, k, grid, ϕ_times_fψ, S_abs, ∂zᶜᶜᶜ, w)

    var"α²β|S̄|S̄₁₁" = p.α^2 * p.β * S̄_abs * ∂xᶜᶜᶜ(i, j, k, grid, ℱxyz²ᵟ, u)
    var"α²β|S̄|S̄₂₂" = p.α^2 * p.β * S̄_abs * ∂yᶜᶜᶜ(i, j, k, grid, ℱxyz²ᵟ, v)
    var"α²β|S̄|S̄₃₃" = p.α^2 * p.β * S̄_abs * ∂zᶜᶜᶜ(i, j, k, grid, ℱxyz²ᵟ, w)

    M₁₁² = (var"⟨|S|S₁₁⟩" - var"α²β|S̄|S̄₁₁")^2
    M₂₂² = (var"⟨|S|S₂₂⟩" - var"α²β|S̄|S̄₂₂")^2
    M₃₃² = (var"⟨|S|S₃₃⟩" - var"α²β|S̄|S̄₃₃")^2

    Δ = volume(i, j, k, grid, Center(), Center(), Center())
    return 4*Δ^4 * (M₁₁² + M₂₂² + M₃₃²)
end


u, v, w = model.velocities

ω = ∂x(v) - ∂y(u)
ω̃ = KernelFunctionOperation{Face, Face, Center}(ℱxy²ᵟ, grid, ω)

ū = KernelFunctionOperation{Face, Center, Center}(ℱxyz²ᵟ, grid, u)
v̄ = KernelFunctionOperation{Center, Face, Center}(ℱxyz²ᵟ, grid, v)
w̄ = KernelFunctionOperation{Center, Center, Face}(ℱxyz²ᵟ, grid, w)

S = KernelFunctionOperation{Center, Center, Center}(strain_rate_tensor_modulus_ccc, model.grid, u, v, w)
S̄ = KernelFunctionOperation{Center, Center, Center}(filtered_strain_rate_tensor_modulus_ccc, model.grid, u, v, w)
S̄2 = KernelFunctionOperation{Center, Center, Center}(strain_rate_tensor_modulus_ccc, model.grid, Field(ū), Field(v̄), Field(w̄))
@show compute!(Field(S))
@show compute!(Field(S̄))
@show compute!(Field(S̄2))

params = (α = 2, β = 1)
Mij² = KernelFunctionOperation{Center, Center, Center}(MᵢⱼMᵢⱼ_ccc, model.grid, u, v, w, params)
compute!(Field(Mij²))


#+++ Set up simulation
simulation = Simulation(model, Δt=0.2, stop_time=50)

wizard = TimeStepWizard(cfl=0.7, max_change=1.1, max_Δt=0.5)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))
add_callback!(simulation, BasicTimeMessenger(), IterationInterval(100))
#---

#+++ Writer and run!
filename = "two_dimensional_turbulence"
simulation.output_writers[:fields] = JLD2OutputWriter(model, (; ω, ω̃, S, S̄, S̄2),
                                                      schedule = TimeInterval(0.6),
                                                      filename = filename * ".jld2",
                                                      overwrite_existing = true)
run!(simulation)
#---

#+++ Plotting
ω_timeseries = FieldTimeSeries(filename * ".jld2", "ω")
ω̃_timeseries = FieldTimeSeries(filename * ".jld2", "ω̃")
S_timeseries = FieldTimeSeries(filename * ".jld2", "S")
S̄_timeseries = FieldTimeSeries(filename * ".jld2", "S̄")
S̄2_timeseries = FieldTimeSeries(filename * ".jld2", "S̄2")

times = ω_timeseries.times

xω, yω, zω = nodes(ω_timeseries)
xc, yc, zc = nodes(S_timeseries)

using CairoMakie
set_theme!(Theme(fontsize = 18))

fig = Figure(size = (800, 800))

axis_kwargs = (xlabel = "x", ylabel = "y", limits = ((0, 2π), (0, 2π)), aspect = AxisAspect(1))

ax_1 = Axis(fig[2, 1]; title = "Vorticity", axis_kwargs...)
#ax_2 = Axis(fig[2, 2]; title = "Filtered vorticity", axis_kwargs...)
ax_2 = Axis(fig[2, 2]; title = "Filtered Strain rate2", axis_kwargs...)
ax_3 = Axis(fig[3, 1]; title = "Strain rate", axis_kwargs...)
ax_4 = Axis(fig[3, 2]; title = "Filtered Strain rate", axis_kwargs...)


n = Observable(1)

ω = @lift interior(ω_timeseries[$n], :, :, 1)
ω̃ = @lift interior(ω̃_timeseries[$n], :, :, 1)
S = @lift interior(S_timeseries[$n], :, :, 1)
S̄ = @lift interior(S̄_timeseries[$n], :, :, 1)
S̄2 = @lift interior(S̄2_timeseries[$n], :, :, 1)

heatmap!(ax_1, xω, yω, ω; colormap = :balance, colorrange = (-2, 2))
#heatmap!(ax_2, xω, yω, ω̃; colormap = :balance, colorrange = (-2, 2))
heatmap!(ax_2, xc, yc, S̄2; colormap = :speed, colorrange = (0, 3))
heatmap!(ax_3, xc, yc, S; colormap = :speed, colorrange = (0, 3))
heatmap!(ax_4, xc, yc, S̄; colormap = :speed, colorrange = (0, 3))

title = @lift "t = " * string(round(times[$n], digits=2))
Label(fig[1, 1:2], title, fontsize=24, tellwidth=false)

frames = 1:length(times)
@info "Making a neat animation of vorticity and speed..."
record(fig, filename * ".mp4", frames, framerate=24) do i
    n[] = i
end
