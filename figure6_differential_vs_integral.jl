using Statistics
using ArgParse
using JLD2
using ColorTypes
using ColorSchemes
using CairoMakie
using FreeConvection

using Flux.Losses: mse
using CairoMakie.Makie: wong_colors
using Oceananigans: interior
using Oceananigans.Units: days

function parse_command_line_arguments()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--output-directory-fluxes"
            help = "Output directory for neural network trained on fluxes."
            arg_type = String

        "--output-directory-timeseries"
            help = "Output directory for neural differential equation trained on time series."
            arg_type = String
    end

    return parse_args(settings)
end

function figure6_differential_vs_integral_loss(ids, times, solution_loss_history_tof, solution_loss_history_tots; filepath_prefix, colors, rows=2, cols=3, alpha=0.3)
    _, Nt = size(solution_loss_history_tof[first(ids)])

    remove_zeros!(xs) = replace!(x -> iszero(x) ? NaN : x, xs)

    fig = Figure()

    # Ordered so that training subplot shows up at the bottom.
    simulation_ids = (10:12, 16:18, 1:9, 13:15, 19:21)

    for (N, sub_ids) in enumerate(simulation_ids)
        i = mod(N-1, cols) + 1
        j = div(N-1, cols) + 1

        xlabel = (i, j) == (3, 1) ? "Simulation time (days)" : ""
        ylabel = (i, j) == (2, 1) ? "Loss" : ""
        ax = Axis(fig[i, j], title=simulation_label(sub_ids[1]), xlabel=xlabel, ylabel=ylabel, yscale=log10,
                             xgridvisible=false, ygridvisible=false, xticklabelsvisible=i == cols, yticklabelsvisible=j == 1)

        loss_nde_tof_min = [minimum([solution_loss_history_tof[id][end, n] for id in sub_ids]) for n in 1:Nt] |> remove_zeros!
        loss_nde_tof_max = [maximum([solution_loss_history_tof[id][end, n] for id in sub_ids]) for n in 1:Nt] |> remove_zeros!
        loss_nde_tof_mean = [mean([solution_loss_history_tof[id][end, n] for id in sub_ids]) for n in 1:Nt] |> remove_zeros!

        loss_nde_tots_min = [minimum([solution_loss_history_tots[id][end, n] for id in sub_ids]) for n in 1:Nt] |> remove_zeros!
        loss_nde_tots_max = [maximum([solution_loss_history_tots[id][end, n] for id in sub_ids]) for n in 1:Nt] |> remove_zeros!
        loss_nde_tots_mean = [mean([solution_loss_history_tots[id][end, n] for id in sub_ids]) for n in 1:Nt] |> remove_zeros!

        band!(ax, times, loss_nde_tof_min, loss_nde_tof_max, color=RGBA(colors[1], alpha))
        lines!(ax, times, loss_nde_tof_mean, linewidth=3, color=colors[1])

        band!(ax, times, loss_nde_tots_min, loss_nde_tots_max, color=RGBA(colors[2], alpha))
        lines!(ax, times, loss_nde_tots_mean, linewidth=3, color=colors[2])

        xlims!(ax, (0, times[end]))
        ylims!(ax, (1e-6, 1e-2))
    end

    entries = [PolyElement(color=c) for c in colors[1:2]]
    labels = ["Trained on fluxes", "Trained on time series"]
    Legend(fig[3, 2], entries, labels, framevisible=false, tellwidth=false, tellheight=false)

    save(filepath_prefix * ".png", fig, px_per_unit=2)
    save(filepath_prefix * ".pdf", fig, pt_per_unit=2)

    return fig
end

args = parse_command_line_arguments()
output_dir_tof = args["output-directory-fluxes"]
output_dir_tots = args["output-directory-timeseries"]

Nz = 32

ids_train = 1:9
ids_test = setdiff(FreeConvection.SIMULATION_IDS, ids_train)

data = load_data(ids_train, ids_test, Nz)

history_filepath_tof = joinpath(output_dir_tof, "neural_network_history_trained_on_fluxes.jld2")
history_filepath_tots = joinpath(output_dir_tots, "neural_network_history_trained_on_timeseries.jld2")

file_tof = jldopen(history_filepath_tof)
file_tots = jldopen(history_filepath_tots)

solution_loss_history_tof = file_tof["solution_loss_history"]
solution_loss_history_tots = file_tots["solution_loss_history"]

ids = keys(data.coarse_datasets) |> collect |> sort
times = data.coarse_datasets[1]["T"].times ./ days
colors = wong_colors()[6:-1:5] .|> RGB
filepath_prefix = "figure6_differential_vs_integral_loss"
figure6_differential_vs_integral_loss(ids, times, solution_loss_history_tof, solution_loss_history_tots; filepath_prefix, colors)
