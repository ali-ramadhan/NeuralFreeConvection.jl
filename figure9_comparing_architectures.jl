using Statistics

using ArgParse
using JLD2
using CairoMakie
using FreeConvection

using Flux.Losses: mse
using CairoMakie.Makie: wong_colors

function figure9_comparing_architectures(nde_solutions, true_solutions, T_scalings, labels; filepath_prefix, rows=2, cols=3)
    fig = Figure()

    colors = wong_colors()[1:end-1] |> reverse

    # Ordered so that training subplot shows up at the bottom.
    simulation_ids = (10:12, 16:18, 1:9, 13:15, 19:21)

    for (N, ids) in enumerate(simulation_ids)
        i = mod(N-1, cols) + 1
        j = div(N-1, cols) + 1

        ax = fig[i, j] = Axis(fig, title=simulation_label(first(ids)), xlabel="Epochs", ylabel="Loss", yscale=log10)

        for (nde_sols, true_sols, T_scaling, color, label) in zip(nde_solutions, true_solutions, T_scalings, colors, labels)
            epochs = length(nde_sols[first(ids)])

            loss_histories = Dict(
                id => [mse(T_scaling.(true_sols[id].T), T_scaling.(nde_sols[id][e].T)) for e in 1:epochs]
                for id in ids
            )

            mean_loss_training = [mean([loss_histories[id][e] for id in ids]) for e in 1:epochs]

            lines!(ax, 1:epochs, mean_loss_training; color)

            ax.xgridvisible = false
            ax.ygridvisible = false

            ax.xticks = 0:100:500
            xlims!(0, epochs-1)
        end

        i != cols && hidexdecorations!(ax, grid=false)
        # j != 1 && hideydecorations!(ax, grid=false)
    end

    entries = [PolyElement(color=c) for c in colors[1:5]]
    Legend(fig[3, 2], entries, labels, framevisible=false, tellwidth=false, tellheight=false)

    @info "Saving $filepath_prefix..."
    save("$filepath_prefix.png", fig, px_per_unit=2)
    save("$filepath_prefix.pdf", fig, pt_per_unit=2)

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

filepath_prefix="figure9_comparing_architectures"
@info "Plotting $filepath_prefix..."
figure9_comparing_architectures(nde_solutions, true_solutions, T_scalings, labels; filepath_prefix)
