# Credit: This script is a modified version of another script developed by Andre Souza (https://github.com/sandreza)

using Statistics
using JLD2
using GLMakie


filepath = "free_convection_19/3d.jld2"
file = jldopen(filepath)

n = time_index = 3

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

colormap = :blues
colorrange = (19.9, 20)

fig = Figure(resolution = (1920, 1080))

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

Colorbar(fig[1, 2], colormap=colormap, limits=colorrange)

T_profile = mean(T[:, :, :, n], dims=(1, 2))
lines(fig[1, 3], T_profile[:], zc)

wT_west = wT[1, :, :, n]
wT_east = wT[Nx, :, :, n]
wT_south = wT[:, 1, :, n]
wT_north = wT[:, Ny, :, n]
wT_bottom = wT[:, :, 1, n]
wT_top = wT[:, :, Nz, n]

colormap = :balance
colorrange = (-0.5, 0.5)

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

Colorbar(fig[2, 2], colormap=colormap, limits=colorrange)

wT_profile = mean(wT[:, :, :, n], dims=(1, 2))
lines(fig[2, 3], wT_profile[:], zc)

display(fig)

# save("figure1_les_box.png", fig, px_per_unit=2)
