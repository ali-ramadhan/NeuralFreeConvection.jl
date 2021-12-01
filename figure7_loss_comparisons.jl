using Statistics

using JLD2
using CairoMakie
using FreeConvection

using Flux.Losses: mse
using CairoMakie.Makie: wong_colors

using Oceananigans: interior
using Oceananigans.Units: days

function plot_figure7_loss_comparisons(datasets, nde_sols, kpp_sols, convective_adjustment_sols, T_scaling; filepath_prefix, rows=2, cols=3, alpha=0.3, ylims=(1e-6, 1e-1))

    loss(T, T̂) = mse(T_scaling.(T), T_scaling.(T̂))

    T = datasets[1]["T"]
    Nt = size(T, 4)
    times = T.times ./ days

    T_solution = Dict(id => [interior(ds["T"])[1, 1, :, n] for n in 1:Nt] for (id, ds) in datasets)

    loss_nde = Dict(id => [loss(T_solution[id][n], nde_sols[id].T[:, n]) for n in 1:Nt] for id in keys(datasets))
    loss_kpp = Dict(id => [loss(T_solution[id][n], kpp_sols[id].T[:, n]) for n in 1:Nt] for id in keys(datasets))
    loss_ca = Dict(id => [loss(T_solution[id][n], convective_adjustment_sols[id].T[:, n]) for n in 1:Nt] for id in keys(datasets))

    fig = Figure()

    colors = wong_colors()
    colors_alpha = wong_colors(alpha)

    # Ordered so that training subplot shows up at the bottom.
    simulation_ids = (10:12, 16:18, 1:9, 13:15, 19:21)

    for (N, sub_ids) in enumerate(simulation_ids)
        i = mod(N-1, cols) + 1
        j = div(N-1, cols) + 1
        ax = fig[i, j] = Axis(fig, title=simulation_label(sub_ids[1]), xlabel="Simulation time (days)", ylabel="Loss", yscale=log10)

        for (p, loss_param) in enumerate((loss_kpp, loss_ca, loss_nde))
            loss_param_min = [minimum([loss_param[id][n] for id in sub_ids]) for n in 1:Nt]
            loss_param_max = [maximum([loss_param[id][n] for id in sub_ids]) for n in 1:Nt]
            loss_param_mean = [mean([loss_param[id][n] for id in sub_ids]) for n in 1:Nt]

            for loss_param_stat in (loss_param_min, loss_param_max, loss_param_mean)
                replace!(x -> iszero(x) ? NaN : x, loss_param_stat)
            end

            band!(ax, times, loss_param_min, loss_param_max, color=colors_alpha[p])
            lines!(ax, times, loss_param_mean, color=colors[p])
        end

        xlims!(0, times[end])
        ylims!(ylims...)

        i != cols && hidexdecorations!(ax, grid=false)
        j != 1 && hideydecorations!(ax, grid=false)
    end

    entries = [PolyElement(color=c) for c in colors[1:3]]
    labels = ["KPP", "Convective adjustment", "NDE"]
    Legend(fig[3, 2], entries, labels, framevisible=false, tellwidth=false, tellheight=false)

    save(filepath_prefix * ".png", fig, px_per_unit=2)
    save(filepath_prefix * ".pdf", fig, pt_per_unit=2)

    return nothing
end

Nz = 32

ids_train = 1:9
ids_test = setdiff(FreeConvection.SIMULATION_IDS, ids_train)

data = load_data(ids_train, ids_test, Nz)
datasets = data.coarse_datasets

file = jldopen("solutions_and_history.jld2", "r")

nde_solutions = file["nde"]
kpp_solutions = file["kpp"]
convective_adjustment_solutions = file["convective_adjustment"]
T_scaling = file["T_scaling"]

filepath_prefix = "figure7_loss_comparisons"
plot_figure7_loss_comparisons(datasets, nde_solutions, kpp_solutions, convective_adjustment_solutions, T_scaling; filepath_prefix)

close(file)
