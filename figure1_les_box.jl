using GLMakie
using ColorSchemes

f(x, y, z) = x^2 + y^2 + z^2

N = 10
Lx = Ly = Lz = 1
x = y = z = range(0, 1, length=N)
data = [f(ix, iy, iz) for ix in x, iy in y, iz in z]

one_coords = ones(N)
zero_coords = zeros(N)

rands = zeros(N, N) .+ 1e-6 * randn(N, N)

cmap = :solar
clims = (0, 3)

# Plot the bottom slice
fig, ax, _ = surface(x, y, rands, color=data[:, :, 1], colormap=cmap, colorrange=clims)

fig = Figure(resolution = (1920, 1080))
ax = Axis3(fig[1, 1], xticks=range(0, Lx, length=5))
surface!(ax, x, y, rands, color=data[:, :, 1], colormap=cmap, colorrange=clims)

surface!(ax, x, y, Lz .+ rands, color=data[:, :, N], colormap=cmap, colorrange=clims)
surface!(ax, x, z, rands, color=data[:, 1, :], transformation=(:xz,  0), colormap=cmap, colorrange=clims)
surface!(ax, x, z, rands, color=data[:, N, :], transformation=(:xz, Ly), colormap=cmap, colorrange=clims)
surface!(ax, y, z, rands, color=data[1, :, :], transformation=(:yz,  0), colormap=cmap, colorrange=clims)
surface!(ax, y, z, rands, color=data[N, :, :], transformation=(:yz, Lx), colormap=cmap, colorrange=clims)

display(fig)
