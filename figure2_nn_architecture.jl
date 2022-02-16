using Printf
using CairoMakie
using Oceananigans
using FreeConvection

using CairoMakie.Makie: wong_colors

function figure2_nn_architecture(ds; filepath_prefix, convective_adjustment_diffusivity=2)
    n4 = 577  # time index for t = 4 days

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

    fig = Figure(resolution=(1200, 500))

    ## Left panel: Temperature

    ax = fig[1, 1] = Axis(fig, xlabel="Temperature (°C)", ylabel="z (m)")

    T_param_n = T_param[:, n4]
    lines!(ax, T_param_n, zc, linewidth=3, color=colors[1])
    scatter!(ax, T_param_n, zc, color=colors[1])

    ax.yticks = 0:-64:-256
    ax.xgridvisible = false
    ax.ygridvisible = false

    # xlims!(19.5, 20)
    ylims!(-256, 0)

    ## Right panel: heat flux

    ax = fig[1, 5] = Axis(fig, xlabel="Missing heat flux (m/s K)")

    wT_missing_n = wT_missing[:, n4]
    lines!(ax, wT_missing_n, zf, linewidth=3, color=colors[2])
    scatter!(ax, wT_missing_n[2:end-1], zf[2:end-1], color=colors[2])

    ax.xticks = ([-2e-6, -1e-6, 0], ["-2×10⁻⁶", "-1×10⁻⁶", "0"])
    ax.yticks = 0:-64:-256
    ax.ytickformat = ys -> ["" for y in ys]
    ax.xgridvisible = false
    ax.ygridvisible = false

    ylims!(-256, 0)

    save("$filepath_prefix.png", fig, px_per_unit=2)
    save("$filepath_prefix.pdf", fig, pt_per_unit=2)
    save("$filepath_prefix.svg", fig, pt_per_unit=2)

    return nothing
end

Nz = 32

ids_train = 1:9
ids_test = setdiff(FreeConvection.SIMULATION_IDS, ids_train)

data = load_data(ids_train, ids_test, Nz)
datasets = data.coarse_datasets

figure2_nn_architecture(datasets[5], filepath_prefix="figure2_nn_architecture")
