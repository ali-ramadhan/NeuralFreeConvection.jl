using Printf
using ArgParse
using JLD2
using ColorSchemes
using CairoMakie
using OceanTurb
using FreeConvection

using Flux.Losses: mse
using Oceananigans: interior, znodes
using Oceananigans.Units: days

function parse_command_line_arguments()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--output-directory"
            help = "Output directory for neural network trained on time series."
            arg_type = String
    end

    return parse_args(settings)
end

function figure8_comparing_profiles(ds, nde_sol, kpp_sol, convective_adjustment_sol, T_scaling; filepath_prefix, time_index, colors,
                                    xticks_T=nothing, xticks_wT=nothing, xlims_T=nothing, xlims_wT=nothing)
    T = ds["T"]
    wT = ds["wT"]
    κₑ_∂z_T = ds["κₑ_∂z_T"]
    Nt = size(T, 4)
    zc = znodes(T)
    zf = znodes(wT)
    times = T.times ./ days

    loss(T, T̂) = mse(T_scaling.(T), T_scaling.(T̂))

    remove_zeros!(xs) = replace!(x -> iszero(x) ? NaN : x, xs)

    loss_nde = [loss(interior(T)[1, 1, :, n], nde_sol.T[:, n]) for n in 1:Nt] |> remove_zeros!
    loss_kpp = [loss(interior(T)[1, 1, :, n], kpp_sol.T[:, n]) for n in 1:Nt] |> remove_zeros!
    loss_ca  = [loss(interior(T)[1, 1, :, n], convective_adjustment_sol.T[:, n]) for n in 1:Nt] |> remove_zeros!

    fig = Figure()

    # Left panel: temperatures

    ax = fig[1, 1] = Axis(fig, xlabel="Temperature (°C)", ylabel="z (m)", xgridvisible=false, ygridvisible=false)

    lines!(ax, interior(T)[1, 1, :, time_index], zc, linewidth=3, color=colors[1])
    lines!(ax, convective_adjustment_sol.T[:, time_index], zc, linewidth=3, color=colors[2])
    lines!(ax, kpp_sol.T[:, time_index], zc, linewidth=3, color=colors[3])
    lines!(ax, nde_sol.T[:, time_index], zc, linewidth=3, color=colors[4])

    ax.yticks = 0:-32:-128
    ylims!(ax, (-128, 0))

    !isnothing(xticks_T) && (ax.xticks = xticks_T)
    !isnothing(xlims_T) && xlims!(ax, xlims_T)

    # Middle panel: heat fluxes

    ax = fig[1, 2] = Axis(fig, xlabel="Heat flux (m/s K)", xgridvisible=false, ygridvisible=false)

    wT_total = interior(wT)[1, 1, :, time_index] .- interior(κₑ_∂z_T)[1, 1, :, time_index]
    lines!(ax, wT_total, zf, linewidth=3, color=colors[1])
    lines!(ax, convective_adjustment_sol.wT[:, time_index], zf, linewidth=3, color=colors[2])
    lines!(ax, kpp_sol.wT[:, time_index], zf, linewidth=3, color=colors[3])
    lines!(ax, nde_sol.wT[:, time_index], zf, linewidth=3, color=colors[4])

    ax.yticks = 0:-32:-128
    ylims!(ax, (-128, 0))

    !isnothing(xticks_wT) && (ax.xticks = xticks_wT)
    !isnothing(xlims_wT) && xlims!(ax, xlims_wT)

    # Right panel: losses

    ax = fig[1, 3] = Axis(fig, xlabel="Simulation time (days)", ylabel="Loss", yscale=log10, xgridvisible=false, ygridvisible=false)

    lines!(ax, times, loss_ca, linewidth=3, color=colors[2])
    lines!(ax, times, loss_kpp, linewidth=3, color=colors[3])
    lines!(ax, times, loss_nde, linewidth=3, color=colors[4])

    xlims!(ax, extrema(times)...)
    ylims!(ax, (1e-5, 1e-2))

    # Legend

    entries = [LineElement(color=colors[l]) for l in 1:4]
    labels = ["Large eddy simulation", "Convective adjustment", "K-Profile Parameterization", "Neural differential equation"]
    Legend(fig[0, :], entries, labels, nbanks=2, orientation=:horizontal, framevisible=false, tellwidth=false, tellheight=true)

    save("$filepath_prefix.png", fig, px_per_unit=2)
    save("$filepath_prefix.pdf", fig, pt_per_unit=2)

    return fig
end

args = parse_command_line_arguments()
output_dir = args["output-directory"]

Nz = 32

ids_train = 1:9
ids_test = setdiff(FreeConvection.SIMULATION_IDS, ids_train)

data = load_data(ids_train, ids_test, Nz)
datasets = data.coarse_datasets

solutions_filepath = joinpath(output_dir, "solutions_and_history.jld2")
file = jldopen(solutions_filepath, "r")

nde_solutions = file["nde"]
convective_adjustment_solutions = file["convective_adjustment"]
T_scaling = file["T_scaling"]

close(file)

# Generate KPP solutions using optimal parameters from optimize_kpp.jl
kpp_parameters = OceanTurb.KPP.Parameters(CSL=2/3, CNL=5.0, Cb_T=0.16, CKE=8.0)
kpp_solutions = Dict(id => free_convection_kpp(ds, parameters=kpp_parameters) for (id, ds) in data.coarse_datasets)

# time_index = 577 # t = 4 days
time_index = 1153 # t = 8 days

# for id in keys(datasets)
#     filepath_prefix = "figure8_comparing_profiles_simulation" * @sprintf("%02d", id)
#     @info "Plotting $filepath_prefix..."
#     figure8_comparing_profiles(datasets[id], nde_solutions[id], kpp_solutions[id], convective_adjustment_solutions[id], T_scaling; filepath_prefix, time_index)
# end

xticks_T = [19.5, 19.7, 19.9]
xticks_wT = ([-0.5e-5, 0, 1e-5], ["-5×10⁻⁶", "0", "1×10⁻⁵"])
xlims_T = (19.5, 19.9)
xlims_wT = (-5e-6, 1e-5)
colors = circshift(ColorSchemes.julia.colors, 1)
filepath_prefix = "figure8_comparing_profiles_simulation11_pretty"
figure8_comparing_profiles(datasets[11], nde_solutions[11], kpp_solutions[11], convective_adjustment_solutions[11], T_scaling;
                           filepath_prefix, time_index, colors, xticks_T, xticks_wT, xlims_T, xlims_wT)
