using Printf
using GLMakie
using Oceananigans
using FreeConvection

using GLMakie.Makie: wong_colors

GLMakie.activate!()

function plot_surfaces!(subfig, data, L, time_index)

    Nx, Ny, Nz, Nt = size(data)
    Lx, Ly, Lz = L

    xc = range(0, Lx, length=Nx)
    yc = range(0, Ly, length=Ny)
    zc = range(-Lz, 0, length=Nz)

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

    data_west = data[1, :, :, time_index]
    data_east = data[Nx, :, :, time_index]
    data_south = data[:, 1, :, time_index]
    data_north = data[:, Ny, :, time_index]
    data_bottom = data[:, :, 1, time_index]
    data_top = data[:, :, Nz, time_index]

    ε = 1e-6
    rands_xy = zeros(Nx, Ny) .+ ε * randn(Nx, Ny)
    rands_xz = zeros(Nx, Nz) .+ ε * randn(Nx, Nz)
    rands_yz = zeros(Ny, Nz) .+ ε * randn(Ny, Nz)

    ax = Axis3(subfig)

    colormap = :balance
    colorrange = (-0.2, 0.2)

    surface!(ax, xc, yc, rands_xy, color=data_top; colormap, colorrange)
    surface!(ax, xc, yc, -Lz .+ rands_xy, color=data_bottom; colormap, colorrange)
    surface!(ax, xc, zc, rands_xz, color=data_south, transformation=(:xz,  0); colormap, colorrange)
    surface!(ax, xc, zc, rands_xz, color=data_north, transformation=(:xz, Ly); colormap, colorrange)
    surface!(ax, yc, zc, rands_yz, color=data_west,  transformation=(:yz,  0); colormap, colorrange)
    surface!(ax, yc, zc, rands_yz, color=data_east,  transformation=(:yz, Lx); colormap, colorrange)

    xlims!(ax, xlims)
    ylims!(ax, ylims)
    zlims!(ax, zlims)

    hidedecorations!(ax)

    return nothing
end

Nz = 32

ids_train = 1:9
ids_test = setdiff(FreeConvection.SIMULATION_IDS, ids_train)

data = load_data(ids_train, ids_test, Nz)
datasets = data.coarse_datasets
ds = datasets[5]

n0 = 1    # time index for t = 0
n2 = 289  # time index for t = 2 days
n8 = 1153 # time index for t = 8 days

ns = [n0, n2, n8]

convective_adjustment_diffusivity = 2
ca_solution = oceananigans_convective_adjustment(ds, K=convective_adjustment_diffusivity)

T_param = ca_solution.T
wT_param = ca_solution.wT

wT_LES = interior(ds["wT"])[1, 1, :, :]
wT_missing = wT_LES .- wT_param

T = ds["T"]
wT = ds["wT"]

zc = znodes(T)
zf = znodes(wT)

colors = wong_colors()

fig = Figure(resolution = (1000, 1000))

## Top panel: w'T' surface plots

ds3d = FieldDataset("free_convection_19/3d_with_halos.jld2")

w3d = interior(ds3d["w"]) |> Array
T3d = interior(ds3d["T"]) |> Array
wT3d = w3d[:, :, 1:end-1, :] .* T3d
data_3d = wT3d
L = (512, 512, 256)
plot_surfaces!(fig[1, 1:2], data_3d, L, 1)
plot_surfaces!(fig[1, 3:4], data_3d, L, 2)
plot_surfaces!(fig[1, 5:6], data_3d, L, 3)

colgap!(fig.layout, 10)
rowgap!(fig.layout, 10)

## Bottom left panel: Temperature

ax = fig[2, 1:3] = Axis(fig, xlabel="Temperature (°C)", ylabel="z (m)")

for (i, n) in enumerate(ns)
    T_n = interior(T)[1, 1, :, n]
    lines!(ax, T_n, zc, linewidth=3, color=colors[i])

    T_param_n = T_param[:, n]
    lines!(ax, T_param_n, zc, linewidth=3, color=colors[i], linestyle=:dash)
end

ax.yticks = 0:-32:-128
ax.xgridvisible = false
ax.ygridvisible = false

xlims!(19.5, 20)
ylims!(-128, 0)


## Bottom right panel: heat flux

ax = fig[2, 4:6] = Axis(fig, xlabel="Heat flux (m/s K)")

for (i, n) in enumerate(ns)
    wT_n = interior(wT)[1, 1, :, n]
    lines!(ax, wT_n, zf, linewidth=3, color=colors[i])

    wT_param_n = wT_param[:, n]
    lines!(ax, wT_param_n, zf, linewidth=3, color=colors[i], linestyle=:dash)

    # Easier to infer wT_missing by eye so I'll leave it off.
    # if plot_missing_flux
    #     wT_missing_n = wT_missing[:, n]
    #     lines!(ax, wT_missing_n, zf, linewidth=3, color=colors[i], linestyle=:dot)
    # end
end

ax.yticks = 0:-32:-128
ax.ytickformat = ys -> ["" for y in ys]
ax.xgridvisible = false
ax.ygridvisible = false

ylims!(-128, 0)

entries = append!([LineElement(color=colors[l]) for l in 1:3], [LineElement(linestyle=s) for s in (:dash,)])
labels = ["t = 0", "t = 2 days", "t = 8 days", "convective adjustment"]
Legend(fig[end+1, :], entries, labels, framevisible=false, orientation=:horizontal, tellwidth=false, tellheight=true)

rowsize!(fig.layout, 1, Relative(1/4))

save("figure3_training_data_plus_3d.png", fig)
