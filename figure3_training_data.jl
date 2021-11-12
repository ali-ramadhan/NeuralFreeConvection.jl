using CairoMakie
using Oceananigans
using FreeConvection

using CairoMakie.Makie: wong_colors

Nz = 32

# data = load_data(ids_train, ids_test, Nz)
# datasets = data.coarse_datasets

id = 3  # Simulation to plot

n0  = 1   # time index for t = 0
n6  = 73  # time index for t = 12 hours
n96 = 577 # time index for t = 4 days

ns = [n0, n6, n96]

ds = datasets[id]

K = 0.2
add_convective_adjustment_flux!(ds, K)

T = ds["T"]
wT = ds["wT"]
wT_param = ds["wT_param"]
wT_missing = ds["wT_missing"]

zc = znodes(T)
zf = znodes(wT)

colors = wong_colors()

fig = Figure()

ax = fig[1, 1] = Axis(fig, xlabel="Temperature (°C)", ylabel="z (m)")

for (i, n) in enumerate(ns)
    T_n = interior(T)[1, 1, :, n]
    lines!(ax, T_n, zc, linewidth=3, color=colors[i])
end

ax.yticks = 0:-32:-128

xlims!(19.75, 20)
ylims!(-128, 0)

ax = fig[1, 2] = Axis(fig, xlabel="Heat flux (m/s K)")

for (i, n) in enumerate(ns)
    wT_n = interior(wT)[1, 1, :, n]
    lines!(ax, wT_n, zf, linewidth=3, color=colors[i])

    wT_param_n = interior(wT_param)[1, 1, :, n]
    lines!(ax, wT_param_n, zf, linewidth=3, color=colors[i], linestyle=:dash)

    wT_missing_n = interior(wT_missing)[1, 1, :, n]
    lines!(ax, wT_missing_n, zf, linewidth=3, color=colors[i], linestyle=:dot)
end

ax.yticks = 0:-32:-128
ax.ytickformat = ys -> ["" for y in ys]

ylims!(-128, 0)

entries = [LineElement(color=colors[l]) for l in 1:3]
labels = ["t = 0", "t = 12 hours", "t = 4 days"]
Legend(fig[0, :], entries, labels, framevisible=false, orientation=:horizontal, tellwidth=false, tellheight=true)

save("figure3_training_data.png", fig, px_per_unit=2)
save("figure3_training_data.pdf", fig, pt_per_unit=2)
