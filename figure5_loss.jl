using Statistics

using ArgParse
using JLD2
using CairoMakie
using FreeConvection

using Flux.Losses: mse

function parse_command_line_arguments()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--output-directory"
            help = "Output directory filepath."
            arg_type = String
    end

    return parse_args(settings)
end

function plot_epoch_loss_summary_filled_curves(ids, nde_solutions, true_solutions, T_scaling; filepath_prefix, normalize=false, alpha=0.3)
    epochs = length(nde_solutions[first(ids)])

    fig = Figure()
    ax1 = fig[1, 1] = Axis(fig, xlabel="Epoch", ylabel="Mean squared error", yscale=log10)

    loss_histories = Dict(
        id => [mse(T_scaling.(true_solutions[id].T), T_scaling.(nde_solutions[id][e].T)) for e in 1:epochs]
        for id in ids
    )

    for sub_ids in (1:9, 10:12, 13:15, 16:18, 19:21)
        min_loss_training = [minimum([loss_histories[id][e] for id in sub_ids]) for e in 1:epochs]
        max_loss_training = [maximum([loss_histories[id][e] for id in sub_ids]) for e in 1:epochs]
        mean_loss_training = [mean([loss_histories[id][e] for id in sub_ids]) for e in 1:epochs]

        if normalize
            normalization_factor = maximum(max_loss_training)
            min_loss_training ./= normalization_factor
            max_loss_training ./= normalization_factor
            mean_loss_training ./= normalization_factor
        end

        band!(ax1, 1:epochs, min_loss_training, max_loss_training, color=simulation_color(sub_ids[1]; alpha))
        lines!(ax1, 1:epochs, mean_loss_training, color=simulation_color(sub_ids[1]))
    end

    xlims!(0, epochs)

    entry_ids = (1, 10, 13, 16, 19)
    entries = [PolyElement(color=simulation_color(id)) for id in entry_ids]
    labels = [simulation_label(id) for id in entry_ids]

    Legend(fig[1, 2], entries, labels, framevisible=false)

    @info "Saving $filepath_prefix..."
    save(filepath_prefix * ".png", fig, px_per_unit=2)
    save(filepath_prefix * ".pdf", fig, pt_per_unit=2)

    return nothing
end

args = parse_command_line_arguments()
output_dir = args["output-directory"]

solutions_filepath = joinpath(output_dir, "solutions_and_history.jld2")
file = jldopen(solutions_filepath, "r")

true_solutions = file["true"]
nde_solutions = file["nde_history"]
T_scaling = file["T_scaling"]

ids = keys(nde_solutions) |> collect |> sort

plot_epoch_loss_summary_filled_curves(ids, nde_solutions, true_solutions, T_scaling, filepath_prefix=joinpath(output_dir, "figure5_loss"))
plot_epoch_loss_summary_filled_curves(ids, nde_solutions, true_solutions, T_scaling, filepath_prefix=joinpath(output_dir, "figure5_loss_normalized"), normalize=true)

close(file)
