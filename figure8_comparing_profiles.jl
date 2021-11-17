using Printf

using JLD2
using CairoMakie
using FreeConvection

using Flux.Losses: mse
using CairoMakie.Makie: wong_colors

using Oceananigans: interior, znodes
using Oceananigans.Units: days

function figure8_comparing_profiles(ds, nde_sol, kpp_sol, convective_adjustment_sol, T_scaling; filepath_prefix, time_index)
    T = ds["T"]
    wT = ds["wT"]
    Nt = size(T, 4)
    zc = znodes(T)
    zf = znodes(wT)
    times = T.times ./ days

    T_lims = interior(T) |> extrema
    wT_lims = interior(wT) |> extrema

    loss(T, T̂) = mse(T_scaling.(T), T_scaling.(T̂))

    loss_nde = [loss(interior(T)[1, 1, :, n], nde_sol.T[:, n]) for n in 1:Nt]
    loss_kpp = [loss(interior(T)[1, 1, :, n], kpp_sol.T[:, n]) for n in 1:Nt]
    loss_ca  = [loss(interior(T)[1, 1, :, n], convective_adjustment_sol.T[:, n]) for n in 1:Nt]

    colors = wong_colors()

    fig = Figure()

    ## Left panel: heat fluxes

    ax = fig[1, 1] = Axis(fig, xlabel="Heat flux (m/s K)", ylabel="z (m)")

    ylims!(-128, 0)

    ax.yticks = 0:-32:-128
    ax.xgridvisible = false
    ax.ygridvisible = false

    lines!(ax, interior(wT)[1, 1, :, time_index], zf, linewidth=3, color=colors[1])
    lines!(ax, convective_adjustment_sol.wT[:, time_index], zf, linewidth=3, color=colors[2])
    lines!(ax, kpp_sol.wT[:, time_index], zf, linewidth=3, color=colors[3])
    lines!(ax, nde_sol.wT[:, time_index], zf, linewidth=3, color=colors[4])

    ## Middle panel: temperatures

    ax = fig[1, 2] = Axis(fig, xlabel="Temperature (°C)", ylabel="z (m)")

    ylims!(-128, 0)

    ax.yticks = 0:-32:-128
    ax.xgridvisible = false
    ax.ygridvisible = false

    lines!(ax, interior(T)[1, 1, :, time_index], zc, linewidth=3, color=colors[1])
    lines!(ax, convective_adjustment_sol.T[:, time_index], zc, linewidth=3, color=colors[2])
    lines!(ax, kpp_sol.T[:, time_index], zc, linewidth=3, color=colors[3])
    lines!(ax, nde_sol.T[:, time_index], zc, linewidth=3, color=colors[4])

    ## Right panel: losses

    ax = fig[1, 3] = Axis(fig, xlabel="Simulation time (days)", ylabel="Loss", yscale=log10)

    xlims!(extrema(times)...)
    ylims!(1e-6, 1e-2)

    ax.xgridvisible = false
    ax.ygridvisible = false

    lines!(ax, times, loss_ca, linewidth=3, color=colors[2])
    lines!(ax, times, loss_kpp, linewidth=3, color=colors[3])
    lines!(ax, times, loss_nde, linewidth=3, color=colors[4])

    entries = [LineElement(color=colors[l]) for l in 1:4]
    labels = ["true (LES)", "Convective adjustment", "KPP", "NDE"]
    Legend(fig[0, :], entries, labels, framevisible=false, orientation=:horizontal, tellwidth=false, tellheight=true)

    save("$filepath_prefix.png", fig, px_per_unit=2)
    save("$filepath_prefix.pdf", fig, pt_per_unit=2)

    return nothing
end

Nz = 32

ids_train = 1:9
ids_test = setdiff(FreeConvection.SIMULATION_IDS, ids_train)

# data = load_data(ids_train, ids_test, Nz)
# datasets = data.coarse_datasets

file = jldopen("solutions_and_history.jld2", "r")

nde_solutions = file["nde"]
kpp_solutions = file["kpp"]
convective_adjustment_solutions = file["convective_adjustment"]
T_scaling = file["T_scaling"]

# time_index = 577 # t = 4 days
time_index = 1153 # t = 8 days

for id in keys(datasets)
    filepath_prefix = "figure8_comparing_profiles_simulation" * @sprintf("%02d", id)
    @info "Plotting $filepath_prefix..."
    figure8_comparing_profiles(datasets[id], nde_solutions[id], kpp_solutions[id], convective_adjustment_solutions[id], T_scaling; filepath_prefix, time_index)
end

close(file)
