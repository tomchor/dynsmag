using Oceananigans

grid = RectilinearGrid(size=(32, 32, 32), extent=(2π, 2π, 2π), topology=(Periodic, Periodic, Periodic))

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
@inline ℱxyz²ᵟ(i, j, k, grid, f, args...) = ℱz²ᵟ(i, j, k, grid, ℱy²ᵟ, ℱx²ᵟ, args...)

model = NonhydrostaticModel(; grid,
                            timestepper = :RungeKutta3,
                            advection = UpwindBiasedFifthOrder(),
                            closure = SmagorinskyLilly())

using Statistics

u, v, w = model.velocities

uᵢ = rand(size(u)...)
vᵢ = rand(size(v)...)

uᵢ .-= mean(uᵢ)
vᵢ .-= mean(vᵢ)

set!(model, u=uᵢ, v=vᵢ)


simulation = Simulation(model, Δt=0.2, stop_time=50)

wizard = TimeStepWizard(cfl=0.7, max_change=1.1, max_Δt=0.5)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

using Printf
function progress_message(sim)
    max_abs_u = maximum(abs, sim.model.velocities.u)
    walltime = prettytime(sim.run_wall_time)
    return @info @sprintf("Iteration: %04d, time: %1.3f, Δt: %.2e, max(|u|) = %.1e, wall time: %s\n", iteration(sim), time(sim), sim.Δt, max_abs_u, walltime)
end

add_callback!(simulation, progress_message, IterationInterval(100))

u, v, w = model.velocities


ω = ∂x(v) - ∂y(u)
s = sqrt(u^2 + v^2)
ω̃ = KernelFunctionOperation{Face, Face, Center}(ℱxy²ᵟ, grid, ω)
@show compute!(Field(ω̃))


# We pass these operations to an output writer below to calculate and output them during the simulation.
filename = "two_dimensional_turbulence"

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; ω, s, ω̃),
                                                      schedule = TimeInterval(0.6),
                                                      filename = filename * ".jld2",
                                                      overwrite_existing = true)

# ## Running the simulation
#
# Pretty much just

run!(simulation)

# ## Visualizing the results
#
# We load the output.

ω_timeseries = FieldTimeSeries(filename * ".jld2", "ω")
ω̃_timeseries = FieldTimeSeries(filename * ".jld2", "ω̃")
s_timeseries = FieldTimeSeries(filename * ".jld2", "s")

times = ω_timeseries.times

# Construct the ``x, y, z`` grid for plotting purposes,

xω, yω, zω = nodes(ω_timeseries)
xs, ys, zs = nodes(s_timeseries)
nothing #hide

# and animate the vorticity and fluid speed.

using CairoMakie
set_theme!(Theme(fontsize = 24))

fig = Figure(size = (800, 500))

axis_kwargs = (xlabel = "x",
               ylabel = "y",
               limits = ((0, 2π), (0, 2π)),
               aspect = AxisAspect(1))

ax_ω = Axis(fig[2, 1]; title = "Vorticity", axis_kwargs...)
ax_s = Axis(fig[2, 2]; title = "Speed", axis_kwargs...)
nothing #hide

# We use Makie's `Observable` to animate the data. To dive into how `Observable`s work we
# refer to [Makie.jl's Documentation](https://makie.juliaplots.org/stable/documentation/nodes/index.html).

n = Observable(1)

# Now let's plot the vorticity and speed.

ω = @lift interior(ω_timeseries[$n], :, :, 1)
ω̃ = @lift interior(ω̃_timeseries[$n], :, :, 1)
s = @lift interior(s_timeseries[$n], :, :, 1)

heatmap!(ax_ω, xω, yω, ω; colormap = :balance, colorrange = (-2, 2))
heatmap!(ax_s, xω, yω, ω̃; colormap = :balance, colorrange = (-2, 2))
#heatmap!(ax_s, xs, ys, s; colormap = :speed, colorrange = (0, 0.2))

title = @lift "t = " * string(round(times[$n], digits=2))
Label(fig[1, 1:2], title, fontsize=24, tellwidth=false)

current_figure() #hide
fig

# Finally, we record a movie.

frames = 1:length(times)

@info "Making a neat animation of vorticity and speed..."

record(fig, filename * ".mp4", frames, framerate=24) do i
    n[] = i
end
nothing #hide

# ![](two_dimensional_turbulence.mp4)
