using Statistics

using JLD2
using CairoMakie
using FreeConvection

using Flux.Losses: mse
using CairoMakie.Makie: wong_colors

using Oceananigans: interior
using Oceananigans.Units: days

function figure6_initial_vs_final_loss_matrix_filled_curves(datasets, nde_sols, initial_nde_sols, T_scaling; filepath_prefix, alpha=0.3, ylims=(1e-6, 1e-1))

    loss(T, T̂) = mse(T_scaling.(T), T_scaling.(T̂))

    T = datasets[1]["T"]
    Nt = size(T, 4)
    times = T.times ./ days

    T_solution = Dict(id => [interior(ds["T"])[1, 1, :, n] for n in 1:Nt] for (id, ds) in datasets)
    loss_nde = Dict(id => [loss(T_solution[id][n], nde_sols[id].T[:, n]) for n in 1:Nt] for id in keys(datasets))
    loss_init = Dict(id => [loss(T_solution[id][n], initial_nde_sols[id].T[:, n]) for n in 1:Nt] for id in keys(datasets))

    loss_nde_min = [minimum([loss_nde[id][n] for id in keys(datasets)]) for n in 1:Nt]
    loss_nde_max = [maximum([loss_nde[id][n] for id in keys(datasets)]) for n in 1:Nt]
    loss_nde_mean = [mean([loss_nde[id][n] for id in keys(datasets)]) for n in 1:Nt]

    loss_init_min = [minimum([loss_init[id][n] for id in keys(datasets)]) for n in 1:Nt]
    loss_init_max = [maximum([loss_init[id][n] for id in keys(datasets)]) for n in 1:Nt]
    loss_init_mean = [mean([loss_init[id][n] for id in keys(datasets)]) for n in 1:Nt]

    for loss_param_stat in (loss_nde_min, loss_nde_max, loss_nde_mean, loss_init_min, loss_init_max, loss_init_mean)
        replace!(x -> iszero(x) ? NaN : x, loss_param_stat)
    end

    fig = Figure()

    colors = wong_colors()
    colors_alpha = wong_colors(alpha)

    ax = fig[1, 1] = Axis(fig, xlabel="Simulation time (days)", ylabel="Loss", yscale=log10)

    band!(ax, times, loss_nde_min, loss_nde_max, color=colors_alpha[1])
    lines!(ax, times, loss_nde_mean, color=colors[1])

    band!(ax, times, loss_init_min, loss_init_max, color=colors_alpha[2])
    lines!(ax, times, loss_init_mean, color=colors[2])

    CairoMakie.xlims!(0, times[end])
    CairoMakie.ylims!(ylims...)

    entries = [PolyElement(color=c) for c in colors[1:2]]
    labels = ["Trained on time series", "Trained on fluxes"]
    Legend(fig[0, 1], entries, labels, orientation=:horizontal, tellheight=true, framevisible=false)

    save(filepath_prefix * ".png", fig, px_per_unit=2)
    save(filepath_prefix * ".pdf", fig, pt_per_unit=2)

    return nothing
end

Nz = 32

ids_train = 1:9
ids_test = setdiff(FreeConvection.SIMULATION_IDS, ids_train)

data = load_data(ids_train, ids_test, Nz)
datasets = data.coarse_datasets

# Temporary until simulations 19-21 are regenerated.
delete!(datasets, 19)
delete!(datasets, 20)
delete!(datasets, 21)

file = jldopen("solutions_and_history.jld2", "r")

nde_solutions = file["nde"]
initial_nde_solutions = file["initial_nde"]
T_scaling = file["T_scaling"]

filepath_prefix = "figure6_online_vs_offline_loss"
figure6_initial_vs_final_loss_matrix_filled_curves(datasets, nde_solutions, initial_nde_solutions, T_scaling; filepath_prefix)

close(file)
