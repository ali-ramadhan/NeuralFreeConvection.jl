using Statistics

using ArgParse
using JLD2
using CairoMakie
using FreeConvection

using Flux.Losses: mse
using CairoMakie.Makie: wong_colors

function figureB_comparing_architectures(ids, nde_solutions, true_solutions, T_scalings, labels; title, filepath_prefix)
    fig = Figure()

    ax = fig[1, 1] = Axis(fig, xlabel="Epoch", ylabel="Mean squared error", title=title, yscale=log10)

    colors = wong_colors()

    for (nde_sols, true_sols, T_scaling, color, label) in zip(nde_solutions, true_solutions, T_scalings, colors, labels)
        epochs = length(nde_sols[first(ids)])

        loss_histories = Dict(
            id => [mse(T_scaling.(true_sols[id].T), T_scaling.(nde_sols[id][e].T)) for e in 1:epochs]
            for id in ids
        )

        mean_loss_training = [mean([loss_histories[id][e] for id in ids]) for e in 1:epochs]

        lines!(ax, 1:epochs, mean_loss_training; color, label)

        ax.xticks = 0:100:500
        xlims!(0, epochs)
    end

    fig[1, 2] = Legend(fig, ax, "NN architectures", framevisible=false)

    @info "Saving $filepath_prefix..."
    save(filepath_prefix * ".png", fig, px_per_unit=2)
    save(filepath_prefix * ".pdf", fig, pt_per_unit=2)

    return nothing
end

nn_archs = ["dense_default", "dense_wider", "dense_deeper", "conv_2", "conv_4"]
labels = ["dense (default)", "dense (wider)", "dense (deeper)", "conv (2)", "conv (4)"]

solutions_filepaths = [joinpath("free_convection_nde_$nn_arch", "solutions_and_history.jld2") for nn_arch in nn_archs]
files = [jldopen(fp, "r") for fp in solutions_filepaths]

true_solutions = [file["true"] for file in files]
nde_solutions = [file["nde_history"] for file in files]
T_scalings = [file["T_scaling"] for file in files]

[close(file) for file in files]

for ids in (1:9, 10:12, 13:15, 16:18, 19:21)
    title = simulation_label(first(ids))
    filepath_prefix="figureB_comparing_architectures_$title"
    @info "Plotting $filepath_prefix..."
    figureB_comparing_architectures(ids, nde_solutions, true_solutions, T_scalings, nn_archs; title, filepath_prefix)
end
