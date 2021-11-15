using CairoMakie
using Oceananigans
using FreeConvection

using CairoMakie.Makie: wong_colors

Nz = 32

ids_train = 1:9
ids_test = setdiff(FreeConvection.SIMULATION_IDS, ids_train)

data = load_data(ids_train, ids_test, Nz)
datasets = data.coarse_datasets

function figure3_training_data(ds; filepath_prefix, plot_missing_flux=false, convective_adjustment_diffusivity=2)
    n0  = 1   # time index for t = 0
    n6  = 73  # time index for t = 12 hours
    n96 = 577 # time index for t = 4 days

    ns = [n0, n6, n96]

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

    fig = Figure()

    ## Left panel: Temperature

    ax = fig[1, 1] = Axis(fig, xlabel="Temperature (°C)", ylabel="z (m)")

    for (i, n) in enumerate(ns)
        T_n = interior(T)[1, 1, :, n]
        lines!(ax, T_n, zc, linewidth=3, color=colors[i])

        T_param_n = T_param[:, n]
        lines!(ax, T_param_n, zc, linewidth=3, color=colors[i], linestyle=:dash)
    end

    ax.yticks = 0:-32:-128
    ax.xgridvisible = false
    ax.ygridvisible = false

    xlims!(19.75, 20)
    ylims!(-128, 0)

    ## Right panel: heat flux

    ax = fig[1, 2] = Axis(fig, xlabel="Heat flux (m/s K)")

    for (i, n) in enumerate(ns)
        wT_n = interior(wT)[1, 1, :, n]
        lines!(ax, wT_n, zf, linewidth=3, color=colors[i])

        wT_param_n = wT_param[:, n]
        lines!(ax, wT_param_n, zf, linewidth=3, color=colors[i], linestyle=:dash)

        # Easier to infer wT_missing by eye so I'll leave it off.
        if plot_missing_flux
            wT_missing_n = wT_missing[:, n]
            lines!(ax, wT_missing_n, zf, linewidth=3, color=colors[i], linestyle=:dot)
        end
    end

    ax.yticks = 0:-32:-128
    ax.ytickformat = ys -> ["" for y in ys]
    ax.xgridvisible = false
    ax.ygridvisible = false

    ylims!(-128, 0)

    entries = append!([LineElement(color=colors[l]) for l in 1:3], [LineElement(linestyle=s) for s in (:dash,)])
    labels = ["t = 0", "t = 12 hours", "t = 4 days", "convective adjustment"]
    Legend(fig[0, :], entries, labels, framevisible=false, orientation=:horizontal, tellwidth=false, tellheight=true)

    save("$filepath_prefix.png", fig, px_per_unit=2)
    save("$filepath_prefix.pdf", fig, pt_per_unit=2)

    return nothing
end

for (id, ds) in datasets
    filepath_prefix = "figure3_training_data_simulation$id"
    @info "Plotting $filepath_prefix..."
    figure3_training_data(ds; filepath_prefix)
end
