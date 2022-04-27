# Credit: This script is a modified version of another script developed by Andre Souza (https://github.com/sandreza)

using Statistics
using Printf
using JLD2
using ColorSchemes
using GLMakie

filepath = "free_convection_19/3d.jld2"
file = jldopen(filepath)

n = time_index = 3  # time index for t = 2 days

Nx = file["grid/Nx"]
Ny = file["grid/Ny"]
Nz = file["grid/Nz"]
Nt = file["timeseries/t"] |> keys |> length

Lx = file["grid/Lx"]
Ly = file["grid/Ly"]
Lz = file["grid/Lz"]

iterations = keys(file["timeseries/t"])
T = cat([file["timeseries/T/$n"] for n in iterations]..., dims=4)
w = cat([file["timeseries/w/$n"] for n in iterations]..., dims=4)
wT = w[:, :, 1:end-1, :] .* T

xc = range(0, Lx, length=Nx)
yc = range(0, Ly, length=Ny)
zc = range(-Lz, 0, length=Nz)
zf = range(-Lz, 0, length=Nz+1)

T_west = T[1, :, :, n]
T_east = T[Nx, :, :, n]
T_south = T[:, 1, :, n]
T_north = T[:, Ny, :, n]
T_bottom = T[:, :, 1, n]
T_top = T[:, :, Nz, n]

ε = 1e-6
rands_xy = zeros(Nx, Ny) .+ ε * randn(Nx, Ny)
rands_xz = zeros(Nx, Nz) .+ ε * randn(Nx, Nz)
rands_yz = zeros(Ny, Nz) .+ ε * randn(Ny, Nz)

colormap = get(ColorSchemes.RdYlBu_11 |> reverse, [exp(4x) for x in range(0, 1, length=100)], :extrema)
colorrange = (19.6, 19.96)

fig = Figure(resolution = (1200, 1000))

aspect = :data

xlabel = "x (m)"
ylabel = "y (m)"
zlabel = "z (m)"

Δ = Lx/50
xlims = (-Δ,  Lx+Δ)
ylims = (-Δ,  Ly+Δ)
zlims = (-Lz-Δ, Δ)

xticks = range(0,  Lx, length=5)
yticks = range(0,  Ly, length=5)
zticks = range(-Lz, 0, length=5)

ax = Axis3(fig[1, 1]; aspect, xlabel, ylabel, zlabel, xticks, yticks, zticks, xlims, ylims, zlims)

surface!(ax, xc, yc, rands_xy, color=T_top; colormap, colorrange)
surface!(ax, xc, yc, -Lz .+ rands_xy, color=T_bottom; colormap, colorrange)
surface!(ax, xc, zc, rands_xz, color=T_south, transformation=(:xz,  0); colormap, colorrange)
surface!(ax, xc, zc, rands_xz, color=T_north, transformation=(:xz, Ly); colormap, colorrange)
surface!(ax, yc, zc, rands_yz, color=T_west,  transformation=(:yz,  0); colormap, colorrange)
surface!(ax, yc, zc, rands_yz, color=T_east,  transformation=(:yz, Lx); colormap, colorrange)

xlims!(ax, xlims)
ylims!(ax, ylims)
zlims!(ax, zlims)

Colorbar(fig[1, 2], label="°C", colormap=colormap, limits=colorrange)

T_profile = mean(T[:, :, :, n], dims=(1, 2))
ax = Axis(fig[1, 3], xlabel="Temperature (°C)", ylabel="z (m)", yticks=zticks, xgridvisible=false, ygridvisible=false)
lines!(ax, T_profile[:], zc, linewidth=4, color=GLMakie.Makie.wong_colors()[1])
ylims!(ax, -Lz, 0)
# hidedecorations!(ax, grid=true)

wT_west = wT[1, :, :, n]
wT_east = wT[Nx, :, :, n]
wT_south = wT[:, 1, :, n]
wT_north = wT[:, Ny, :, n]
wT_bottom = wT[:, :, 1, n]
wT_top = wT[:, :, Nz, n]

colormap = :balance
colorrange = (-0.2, 0.2)

ax = Axis3(fig[2, 1]; aspect, xlabel, ylabel, zlabel, xticks, yticks, zticks, xlims, ylims, zlims)

surface!(ax, xc, yc, rands_xy, color=wT_top; colormap, colorrange)
surface!(ax, xc, yc, -Lz .+ rands_xy, color=wT_bottom; colormap, colorrange)
surface!(ax, xc, zc, rands_xz, color=wT_south, transformation=(:xz,  0); colormap, colorrange)
surface!(ax, xc, zc, rands_xz, color=wT_north, transformation=(:xz, Ly); colormap, colorrange)
surface!(ax, yc, zc, rands_yz, color=wT_west,  transformation=(:yz,  0); colormap, colorrange)
surface!(ax, yc, zc, rands_yz, color=wT_east,  transformation=(:yz, Lx); colormap, colorrange)

xlims!(ax, xlims)
ylims!(ax, ylims)
zlims!(ax, zlims)

Colorbar(fig[2, 2], label="m/s K", colormap=colormap, limits=colorrange)

stats_filepath = "/home/alir/storage2/LESbrary.jl/data/free_convection_19/instantaneous_statistics.jld2"
stats_file = jldopen(stats_filepath)
iterations = keys(stats_file["timeseries/t"])

n′ = 289  # time index for t = 2 days
i′ = iterations[n′]
wT_profile = stats_file["timeseries/wT/$i′"][:] .- stats_file["timeseries/κₑ_∂z_T/$i′"][:]
wT_profile[Nz+1] = file["parameters/temperature_flux"]
wT_profile[Nz] = (wT_profile[Nz-1] + wT_profile[Nz+1]) / 2

xticks = ([0, 2e-6, 4e-6], ["0", "2×10⁻⁶", "4×10⁻⁶"])
ax = Axis(fig[2, 3], xlabel="Heat flux (m/s K)", ylabel="z (m)", xticks=xticks, yticks=zticks, xgridvisible=false, ygridvisible=false)
lines!(ax, wT_profile, zf, linewidth=4, color=GLMakie.Makie.wong_colors()[2])
ylims!(ax, -Lz, 0)

colsize!(fig.layout, 1, Relative(2/3))

# display(fig)
save("figure1_les_box.png", fig, px_per_unit=2)

close(file)
close(stats_file)
