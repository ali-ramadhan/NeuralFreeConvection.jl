using Statistics
using ArgParse
using JLD2
using ColorSchemes
using CairoMakie
using FreeConvection

using Flux.Losses: mse
using CairoMakie.Makie: wong_colors

function figure9_comparing_architectures(solution_loss_histories, labels; filepath_prefix, rows=2, cols=3, colors)
    fig = Figure()

    # Ordered so that training subplot shows up at the bottom.
    simulation_ids = (10:12, 16:18, 1:9, 13:15, 19:21)

    for (N, sub_ids) in enumerate(simulation_ids)
        i = mod(N-1, cols) + 1
        j = div(N-1, cols) + 1

        xlabel = (i, j) == (3, 1) ? "Epoch" : ""
        ylabel = (i, j) == (2, 1) ? "Loss" : ""
        ax = fig[i, j] = Axis(fig, title=simulation_label(first(sub_ids)), yscale=log10, xlabel=xlabel, ylabel=ylabel,
                                   xgridvisible=false, ygridvisible=false, xticklabelsvisible=i == cols)

        for (solution_loss_history, color) in zip(solution_loss_histories, colors)
            epochs = size(solution_loss_history[first(sub_ids)], 1)

            mean_sol_loss = Dict(id => mean(solution_loss_history[id], dims=2)[:] for id in sub_ids)
            mean_sol_loss = [mean([mean_sol_loss[id][e] for id in sub_ids]) for e in 1:epochs]

            lines!(ax, 1:epochs, mean_sol_loss, linewidth=3; color)

            xlims!(ax, (0, epochs))
        end
    end

    entries = [LineElement(color=c) for c in colors[1:5]]
    Legend(fig[3, 2], entries, labels, framevisible=false, tellwidth=false, tellheight=false)

    @info "Saving $filepath_prefix..."
    save("$filepath_prefix.png", fig, px_per_unit=2)
    save("$filepath_prefix.pdf", fig, pt_per_unit=2)

    return fig
end

nn_archs = ["dense_default", "dense_wider", "dense_deeper", "conv_2", "conv_4"]
labels = ["dense (default)", "dense (wider)", "dense (deeper)", "convolutional (2)", "convolutional (4)"]

filepaths = [joinpath("trained_on_timeseries_$nn_arch", "neural_network_history_trained_on_timeseries.jld2") for nn_arch in nn_archs]
files = [jldopen(fp, "r") for fp in filepaths]

solution_loss_histories = [file["solution_loss_history"] for file in files]

[close(file) for file in files]

filepath_prefix="figure9_comparing_architectures"
@info "Plotting $filepath_prefix..."
figure9_comparing_architectures(solution_loss_histories, labels, colors=ColorSchemes.tab10; filepath_prefix)
