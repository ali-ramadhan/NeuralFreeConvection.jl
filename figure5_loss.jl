using Statistics

using ArgParse
using JLD2
using CairoMakie
using FreeConvection

using Flux.Losses: mse
using OrdinaryDiffEq: ROCK4
using Oceananigans: interior

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

function clean_flux_loss_history(lh; max_loss=1)
    lh = copy(lh)
    outliers = lh .> max_loss
    lh[outliers] .= eps()  # To avoid plotting zeros on a log scale.
    return lh
end

function figure5_losses(ids, flux_loss_history_tof, flux_loss_history_tots, solution_loss_history_tof, solution_loss_history_tots; filepath_prefix)
    fig = Figure(resolution=(800, 600), figure_padding=(10, 40, 0, 0))

    epochs_tof = size(flux_loss_history_tof[first(ids)], 1)
    epochs_tots = size(flux_loss_history_tots[first(ids)], 1)

    mean_flux_loss_tof = Dict(id => mean(clean_flux_loss_history(flux_loss_history_tof[id]), dims=2)[:] for id in ids)
    mean_flux_loss_tots = Dict(id => mean(clean_flux_loss_history(flux_loss_history_tots[id]), dims=2)[:] for id in ids)

    mean_sol_loss_tof = Dict(id => mean(solution_loss_history_tof[id], dims=2)[:] for id in ids)
    mean_sol_loss_tots = Dict(id => mean(solution_loss_history_tots[id], dims=2)[:] for id in ids)

    ax11 = Axis(fig[1, 1], title="Trained on ℒ₁ (fluxes)", ylabel="ℒ₁ (fluxes)", xgridvisible=false, ygridvisible=false, xticklabelsvisible=false)
    ax12 = Axis(fig[1, 2], title="Trained on ℒ₂ (time series)", xgridvisible=false, ygridvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
    ax21 = Axis(fig[2, 1], xlabel="Epoch", ylabel="ℒ₂ (time series)", yscale=log10, xgridvisible=false, ygridvisible=false)
    ax22 = Axis(fig[2, 2], xlabel="Epoch", yscale=log10, xgridvisible=false, ygridvisible=false, yticklabelsvisible=false)

    simulation_sub_ids = (10:12, 13:15, 16:18, 19:21, 1:9)

    for sub_ids in simulation_sub_ids
        mean_flux_loss_tof_sub = [mean([mean_flux_loss_tof[id][e] for id in sub_ids]) for e in 1:epochs_tof]
        lines!(ax11, 1:epochs_tof, mean_flux_loss_tof_sub, linewidth=3, color=simulation_color(sub_ids[1]))

        mean_flux_loss_tots_sub = [mean([mean_flux_loss_tots[id][e] for id in sub_ids]) for e in 1:epochs_tots]
        lines!(ax12, 1:epochs_tots, mean_flux_loss_tots_sub, linewidth=3, color=simulation_color(sub_ids[1]))

        mean_sol_loss_tof_sub = [mean([mean_sol_loss_tof[id][e] for id in sub_ids]) for e in 1:epochs_tof]
        lines!(ax21, 1:epochs_tof, mean_sol_loss_tof_sub, linewidth=3, color=simulation_color(sub_ids[1]))

        mean_sol_loss_tots_sub = [mean([mean_sol_loss_tots[id][e] for id in sub_ids]) for e in 1:epochs_tots]
        lines!(ax22, 1:epochs_tots, mean_sol_loss_tots_sub, linewidth=3, color=simulation_color(sub_ids[1]))
    end

    xlims!(ax11, (0, epochs_tof))
    xlims!(ax12, (0, epochs_tots))
    xlims!(ax21, (0, epochs_tof))
    xlims!(ax22, (0, epochs_tots))

    ylims!(ax11, (0, 3e-1))
    ylims!(ax12, (0, 3e-1))
    ylims!(ax21, (5e-6, 2e-3))
    ylims!(ax22, (5e-6, 2e-3))

    ax21.yticks = ([1e-5, 1e-4, 1e-3], ["10⁻⁵", "10⁻⁴", "10⁻³"])
    ax22.yticks = ([1e-5, 1e-4, 1e-3], ["10⁻⁵", "10⁻⁴", "10⁻³"])

    simulation_sub_ids = (1:9, 10:12, 13:15, 16:18, 19:21)
    entries = [LineElement(color=simulation_color(sub_ids[1])) for sub_ids in simulation_sub_ids]
    labels = [simulation_label(sub_ids[1]) for sub_ids in simulation_sub_ids]
    Legend(fig[0, :], entries, labels, framevisible=false, orientation=:horizontal, tellwidth=true, tellheight=true)

    @info "Saving $filepath_prefix..."
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
datasets = data.coarse_datasets

history_filepath_tof = joinpath(output_dir_tof, "neural_network_history_trained_on_fluxes.jld2")
history_filepath_tots = joinpath(output_dir_tots, "neural_network_history_trained_on_timeseries.jld2")

file_tof = jldopen(history_filepath_tof)
file_tots = jldopen(history_filepath_tots)

flux_loss_history_tof = file_tof["flux_loss_history"]
flux_loss_history_tots = file_tots["flux_loss_history"]

solution_loss_history_tof = file_tof["solution_loss_history"]
solution_loss_history_tots = file_tots["solution_loss_history"]

close(file_tof)
close(file_tots)

ids = keys(datasets) |> collect |> sort
figure5_losses(ids, flux_loss_history_tof, flux_loss_history_tots, solution_loss_history_tof, solution_loss_history_tots, filepath_prefix="figure5_loss_4panels")
