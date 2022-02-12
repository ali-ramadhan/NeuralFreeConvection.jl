# Credit: This script is a modified version of another script developed by Andre Souza (https://github.com/sandreza)

using Statistics
using Printf

using GLMakie
using Oceananigans

filepath = "free_convection_19/3d_with_halos.jld2"
ds = FieldDataset(filepath)

n = time_index = 3

Nx, Ny, Nz, Nt = size(ds["T"])
Lx, Ly, Lz = length(ds["T"].grid)
T = interior(ds["T"])

# xc, yc, zc = nodes(ds["T"])
xc = range(0, Lx, length=Nx)
yc = range(0, Ly, length=Ny)
zc = range(-Lz, 0, length=Nz)

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
colorrange = (19.6, 20)

fig = Figure(resolution = (1920, 1080))

aspect = :data

xlabel = "x (m)"
ylabel = "y (m)"
zlabel = "z (m)"

xlims = (0,  Lx)
ylims = (0,  Ly)
zlims = (-Lz, 0)

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

display(fig)

save("figure1_les_box.png", fig, px_per_unit=2)
