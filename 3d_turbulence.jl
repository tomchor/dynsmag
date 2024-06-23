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


include("aux_functions.jl")

u, v, w = model.velocities

ω = ∂x(v) - ∂y(u)
ω̃ = KernelFunctionOperation{Face, Face, Center}(ℱxy²ᵟ, grid, ω)

ū = KernelFunctionOperation{Face, Center, Center}(ℱ²ᵟ, grid, u)
v̄ = KernelFunctionOperation{Center, Face, Center}(ℱ²ᵟ, grid, v)
w̄ = KernelFunctionOperation{Center, Center, Face}(ℱ²ᵟ, grid, w)

S = KernelFunctionOperation{Center, Center, Center}(strain_rate_tensor_modulus_ccc, model.grid, u, v, w)
S̄ = KernelFunctionOperation{Center, Center, Center}(strain_rate_tensor_modulus_ccc, model.grid, ū, v̄, w̄)
#S̄ = KernelFunctionOperation{Center, Center, Center}(filtered_strain_rate_tensor_modulus_ccc, model.grid, u, v, w)
S̄2 = KernelFunctionOperation{Center, Center, Center}(Σ̄ᵢⱼΣ̄ᵢⱼᶜᶜᶜ, model.grid, u, v, w)
@show compute!(Field(S))
@show compute!(Field(S̄))
@show compute!(Field(S̄2))


@inline var"|S|S₁₁ᶜᶜᶜ"(i, j, k, grid, u, v, w) = √(ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)) * Σ₁₁(i, j, k, grid, u, v, w) # ccc
@inline var"|S|S₂₂ᶜᶜᶜ"(i, j, k, grid, u, v, w) = √(ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)) * Σ₂₂(i, j, k, grid, u, v, w) # ccc
@inline var"|S|S₃₃ᶜᶜᶜ"(i, j, k, grid, u, v, w) = √(ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)) * Σ₃₃(i, j, k, grid, u, v, w) # ccc

@inline var"|S|S₁₂ᶜᶜᶜ"(i, j, k, grid, u, v, w) = √(ΣᵢⱼΣᵢⱼᶠᶠᶜ(i, j, k, grid, u, v, w)) * Σ₁₂(i, j, k, grid, u, v, w) # ffc
@inline var"|S|S₁₃ᶜᶜᶜ"(i, j, k, grid, u, v, w) = √(ΣᵢⱼΣᵢⱼᶠᶜᶠ(i, j, k, grid, u, v, w)) * Σ₁₃(i, j, k, grid, u, v, w) # fcf
@inline var"|S|S₂₃ᶜᶜᶜ"(i, j, k, grid, u, v, w) = √(ΣᵢⱼΣᵢⱼᶜᶠᶠ(i, j, k, grid, u, v, w)) * Σ₂₃(i, j, k, grid, u, v, w) # cff

@inline var"⟨|S|S₁₁⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) = ℱ²ᵟ(i, j, k, grid, var"|S|S₁₁ᶜᶜᶜ", u, v, w)
@inline var"⟨|S|S₂₂⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) = ℱ²ᵟ(i, j, k, grid, var"|S|S₂₂ᶜᶜᶜ", u, v, w)
@inline var"⟨|S|S₃₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) = ℱ²ᵟ(i, j, k, grid, var"|S|S₃₃ᶜᶜᶜ", u, v, w)

@inline var"⟨|S|S₁₂⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) = ℑxyᶜᶜᵃ(i, j, k, grid, ℱ²ᵟ, var"|S|S₁₂ᶜᶜᶜ", u, v, w)
@inline var"⟨|S|S₁₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) = ℑxzᶜᵃᶜ(i, j, k, grid, ℱ²ᵟ, var"|S|S₁₃ᶜᶜᶜ", u, v, w)
@inline var"⟨|S|S₂₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) = ℑyzᵃᶜᶜ(i, j, k, grid, ℱ²ᵟ, var"|S|S₂₃ᶜᶜᶜ", u, v, w)

@inline var"|S̄|S̄₁₁ᶜᶜᶜ"(i, j, k, grid, u, v, w) = √(Σ̄ᵢⱼΣ̄ᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)) * Σ̄₁₁(i, j, k, grid, u, v, w) # ccc
@inline var"|S̄|S̄₂₂ᶜᶜᶜ"(i, j, k, grid, u, v, w) = √(Σ̄ᵢⱼΣ̄ᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)) * Σ̄₂₂(i, j, k, grid, u, v, w) # ccc
@inline var"|S̄|S̄₃₃ᶜᶜᶜ"(i, j, k, grid, u, v, w) = √(Σ̄ᵢⱼΣ̄ᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)) * Σ̄₃₃(i, j, k, grid, u, v, w) # ccc

@inline var"|S̄|S̄₁₂ᶜᶜᶜ"(i, j, k, grid, u, v, w) = √(Σ̄ᵢⱼΣ̄ᵢⱼᶠᶠᶜ(i, j, k, grid, u, v, w)) * Σ̄₁₂(i, j, k, grid, u, v, w) # ffc
@inline var"|S̄|S̄₁₃ᶜᶜᶜ"(i, j, k, grid, u, v, w) = √(Σ̄ᵢⱼΣ̄ᵢⱼᶠᶜᶠ(i, j, k, grid, u, v, w)) * Σ̄₁₃(i, j, k, grid, u, v, w) # fcf
@inline var"|S̄|S̄₂₃ᶜᶜᶜ"(i, j, k, grid, u, v, w) = √(Σ̄ᵢⱼΣ̄ᵢⱼᶜᶠᶠ(i, j, k, grid, u, v, w)) * Σ̄₂₃(i, j, k, grid, u, v, w) # cff


@inline Δ(i, j, k, grid) = volume(i, j, k, grid, Center(), Center(), Center())
@inline M₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, α, β) = 2*Δ(i, j, k, grid)^2 * (var"⟨|S|S₁₁⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) - α^2*β * var"|S̄|S̄₁₁ᶜᶜᶜ"(i, j, k, grid, u, v, w))
@inline M₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, α, β) = 2*Δ(i, j, k, grid)^2 * (var"⟨|S|S₂₂⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) - α^2*β * var"|S̄|S̄₂₂ᶜᶜᶜ"(i, j, k, grid, u, v, w))
@inline M₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, α, β) = 2*Δ(i, j, k, grid)^2 * (var"⟨|S|S₃₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) - α^2*β * var"|S̄|S̄₃₃ᶜᶜᶜ"(i, j, k, grid, u, v, w))

@inline M₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w, α, β) = 2*Δ(i, j, k, grid)^2 * (var"⟨|S|S₁₂⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) - α^2*β * var"|S̄|S̄₁₂ᶜᶜᶜ"(i, j, k, grid, u, v, w))
@inline M₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w, α, β) = 2*Δ(i, j, k, grid)^2 * (var"⟨|S|S₁₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) - α^2*β * var"|S̄|S̄₁₃ᶜᶜᶜ"(i, j, k, grid, u, v, w))
@inline M₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w, α, β) = 2*Δ(i, j, k, grid)^2 * (var"⟨|S|S₂₃⟩ᶜᶜᶜ"(i, j, k, grid, u, v, w) - α^2*β * var"|S̄|S̄₂₃ᶜᶜᶜ"(i, j, k, grid, u, v, w))

function MᵢⱼMᵢⱼ_ccc(i, j, k, grid, u, v, w, p)
    return (      M₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1)^2
            +     M₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1)^2
            +     M₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1)^2
            + 2 * M₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1)^2
            + 2 * M₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1)^2
            + 2 * M₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1)^2)
end

L₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℱ²ᵟ(i, j, k, grid, u₁u₁, u, v, w) - ū₁ū₁ᶜᶜᶜ(i, j, k, grid, u, v, w)
L₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℱ²ᵟ(i, j, k, grid, u₂u₂, u, v, w) - ū₂ū₂ᶜᶜᶜ(i, j, k, grid, u, v, w)
L₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w) = ℱ²ᵟ(i, j, k, grid, u₃u₃, u, v, w) - ū₃ū₃ᶜᶜᶜ(i, j, k, grid, u, v, w)

function LᵢⱼMᵢⱼ_ccc(i, j, k, grid, u, v, w, p)
    S_abs = strain_rate_tensor_modulus_ccc(i, j, k, grid, u, v, w)
    S̄_abs = filtered_strain_rate_tensor_modulus_ccc(i, j, k, grid, u, v, w)
    a = M₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1)


    var"⟨|S|S₁₁⟩" = ℱ²ᵟ(i, j, k, grid, var"|S|S₁₁", u, v, w)
    var"⟨|S|S₂₂⟩" = ℱ²ᵟ(i, j, k, grid, var"|S|S₂₂", u, v, w)
    var"⟨|S|S₃₃⟩" = ℱ²ᵟ(i, j, k, grid, var"|S|S₃₃", u, v, w)

    var"⟨|S|S₁₂⟩" = ℑxyᶜᶜᵃ(i, j, k, grid, ℱ²ᵟ, var"|S|S₁₂", u, v, w)
    var"⟨|S|S₁₃⟩" = ℑxzᶜᵃᶜ(i, j, k, grid, ℱ²ᵟ, var"|S|S₁₃", u, v, w)
    var"⟨|S|S₂₃⟩" = ℑyzᵃᶜᶜ(i, j, k, grid, ℱ²ᵟ, var"|S|S₂₃", u, v, w)


    var"α²β|S̄|S̄₁₁" = p.α^2 * p.β * var"|S̄|S̄₁₁"(i, j, k, grid, u, v, w)
    var"α²β|S̄|S̄₂₂" = p.α^2 * p.β * var"|S̄|S̄₂₂"(i, j, k, grid, u, v, w)
    var"α²β|S̄|S̄₃₃" = p.α^2 * p.β * var"|S̄|S̄₃₃"(i, j, k, grid, u, v, w)

    var"α²β|S̄|S̄₁₂" = p.α^2 * p.β * ℑxyᶜᶜᵃ(i, j, k, grid, var"|S̄|S̄₁₂", u, v, w)
    var"α²β|S̄|S̄₁₃" = p.α^2 * p.β * ℑxzᶜᵃᶜ(i, j, k, grid, var"|S̄|S̄₁₃", u, v, w)
    var"α²β|S̄|S̄₂₃" = p.α^2 * p.β * ℑyzᵃᶜᶜ(i, j, k, grid, var"|S̄|S̄₂₃", u, v, w)

    LᵢⱼMᵢⱼ =   (var"⟨|S|S₁₁⟩" - var"α²β|S̄|S̄₁₁")^2
             + (var"⟨|S|S₂₂⟩" - var"α²β|S̄|S̄₂₂")^2
             + (var"⟨|S|S₃₃⟩" - var"α²β|S̄|S̄₃₃")^2
             + 2*(var"⟨|S|S₁₁⟩" - var"α²β|S̄|S̄₁₁")^2
             + 2*(var"⟨|S|S₂₂⟩" - var"α²β|S̄|S̄₂₂")^2
             + 2*(var"⟨|S|S₃₃⟩" - var"α²β|S̄|S̄₃₃")^2

    Δ = volume(i, j, k, grid, Center(), Center(), Center())
    return LᵢⱼMᵢⱼ
end

LᵢⱼMᵢⱼ = KernelFunctionOperation{Center, Center, Center}(LᵢⱼMᵢⱼ_ccc, model.grid, u, v, w, (; α = 2, β = 1))
@show compute!(Field(LᵢⱼMᵢⱼ))
pause

function MᵢⱼMᵢⱼ_ccc(i, j, k, grid, u, v, w, p)
    S_abs = strain_rate_tensor_modulus_ccc(i, j, k, grid, u, v, w)
    S̄_abs = filtered_strain_rate_tensor_modulus_ccc(i, j, k, grid, u, v, w)

    var"⟨|S|S₁₁⟩" = ℱ²ᵟ(i, j, k, grid, var"|S|S₁₁", u, v, w)
    var"⟨|S|S₂₂⟩" = ℱ²ᵟ(i, j, k, grid, var"|S|S₂₂", u, v, w)
    var"⟨|S|S₃₃⟩" = ℱ²ᵟ(i, j, k, grid, var"|S|S₃₃", u, v, w)

    var"⟨|S|S₁₂⟩" = ℑxyᶜᶜᵃ(i, j, k, grid, ℱ²ᵟ, var"|S|S₁₂", u, v, w)
    var"⟨|S|S₁₃⟩" = ℑxzᶜᵃᶜ(i, j, k, grid, ℱ²ᵟ, var"|S|S₁₃", u, v, w)
    var"⟨|S|S₂₃⟩" = ℑyzᵃᶜᶜ(i, j, k, grid, ℱ²ᵟ, var"|S|S₂₃", u, v, w)


    var"α²β|S̄|S̄₁₁" = p.α^2 * p.β * var"|S̄|S̄₁₁"(i, j, k, grid, u, v, w)
    var"α²β|S̄|S̄₂₂" = p.α^2 * p.β * var"|S̄|S̄₂₂"(i, j, k, grid, u, v, w)
    var"α²β|S̄|S̄₃₃" = p.α^2 * p.β * var"|S̄|S̄₃₃"(i, j, k, grid, u, v, w)

    var"α²β|S̄|S̄₁₂" = p.α^2 * p.β * ℑxyᶜᶜᵃ(i, j, k, grid, var"|S̄|S̄₁₂", u, v, w)
    var"α²β|S̄|S̄₁₃" = p.α^2 * p.β * ℑxzᶜᵃᶜ(i, j, k, grid, var"|S̄|S̄₁₃", u, v, w)
    var"α²β|S̄|S̄₂₃" = p.α^2 * p.β * ℑyzᵃᶜᶜ(i, j, k, grid, var"|S̄|S̄₂₃", u, v, w)

    Δ = volume(i, j, k, grid, Center(), Center(), Center())
    return 4 * Δ^4 * (  (var"⟨|S|S₁₁⟩" - var"α²β|S̄|S̄₁₁")^2
                      + (var"⟨|S|S₂₂⟩" - var"α²β|S̄|S̄₂₂")^2
                      + (var"⟨|S|S₃₃⟩" - var"α²β|S̄|S̄₃₃")^2
                      + 2*(var"⟨|S|S₁₁⟩" - var"α²β|S̄|S̄₁₁")^2
                      + 2*(var"⟨|S|S₂₂⟩" - var"α²β|S̄|S̄₂₂")^2
                      + 2*(var"⟨|S|S₃₃⟩" - var"α²β|S̄|S̄₃₃")^2)
end


MijMᵢⱼ = KernelFunctionOperation{Center, Center, Center}(MᵢⱼMᵢⱼ_ccc, model.grid, u, v, w, (; α = 2, β = 1))
@show compute!(Field(MijMᵢⱼ))
pause


#+++ Set up simulation
@info "Setting up simulation"
simulation = Simulation(model, Δt=0.2, stop_time=50)

wizard = TimeStepWizard(cfl=0.7, max_change=1.1, max_Δt=0.5)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))
add_callback!(simulation, BasicTimeMessenger(), IterationInterval(100))
#---

#+++ Writer and run!
@info "Setting up writer"
filename = "two_dimensional_turbulence"
simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u, v, w, ū, v̄, w̄, ω, ω̃, S, S̄, S̄2),
                                                      schedule = TimeInterval(0.6),
                                                      filename = filename * ".jld2",
                                                      overwrite_existing = true)
@info "Start running"
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
