using Printf

using JLD2
using CairoMakie
using FreeConvection

using Flux.Losses: mse
using CairoMakie.Makie: wong_colors

using Oceananigans: interior, znodes
using Oceananigans.Units: days

function movie2_comparing_profiles(ds, nde_sol, kpp_sol, convective_adjustment_sol, T_scaling; filepath_prefix, resolution=(1280, 720), fps=30)
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

    fig = Figure(; resolution)

    time_index = Node(1)

    ## Left panel: heat fluxes

    ax = fig[1, 1] = Axis(fig, xlabel="Heat flux (m/s K)", ylabel="z (m)")

    ylims!(-128, 0)
    xlims!(extrema(nde_sol.wT)...)

    ax.yticks = 0:-32:-128
    ax.xgridvisible = false
    ax.ygridvisible = false

    wT_les = @lift interior(wT)[1, 1, :, $time_index]
    wT_ca = @lift convective_adjustment_sol.wT[:, $time_index]
    wT_kpp = @lift kpp_sol.wT[:, $time_index]
    wT_nde = @lift nde_sol.wT[:, $time_index]

    lines!(ax, wT_les, zf, linewidth=3, color=colors[1])
    lines!(ax, wT_ca, zf, linewidth=3, color=colors[2])
    lines!(ax, wT_kpp, zf, linewidth=3, color=colors[3])
    lines!(ax, wT_nde, zf, linewidth=3, color=colors[4])

    ## Middle panel: temperatures

    ax = fig[1, 2] = Axis(fig, xlabel="Temperature (°C)", ylabel="z (m)")

    ylims!(-128, 0)

    ax.yticks = 0:-32:-128
    ax.xgridvisible = false
    ax.ygridvisible = false

    T_les = @lift interior(T)[1, 1, :, $time_index]
    T_ca = @lift convective_adjustment_sol.T[:, $time_index]
    T_kpp = @lift kpp_sol.T[:, $time_index]
    T_nde = @lift nde_sol.T[:, $time_index]

    lines!(ax, T_les, zc, linewidth=3, color=colors[1])
    lines!(ax, T_ca, zc, linewidth=3, color=colors[2])
    lines!(ax, T_kpp, zc, linewidth=3, color=colors[3])
    lines!(ax, T_nde, zc, linewidth=3, color=colors[4])

    ## Right panel: losses

    ax = fig[1, 3] = Axis(fig, xlabel="Simulation time (days)", ylabel="Loss", yscale=log10)

    xlims!(extrema(times)...)
    ylims!(1e-6, 1e-2)

    ax.xgridvisible = false
    ax.ygridvisible = false

    # times_evolution = @lift times[1:$time_index]
    # loss_ca_evolution = @lift loss_ca[1:$time_index]
    # loss_kpp_evolution = @lift loss_kpp[1:$time_index]
    # loss_nde_evolution = @lift loss_nde[1:$time_index]

    lines!(ax, times, loss_ca, linewidth=3, color=colors[2])
    lines!(ax, times, loss_kpp, linewidth=3, color=colors[3])
    lines!(ax, times, loss_nde, linewidth=3, color=colors[4])

    entries = [LineElement(color=colors[l]) for l in 1:4]
    labels = ["true (LES)", "Convective adjustment", "KPP", "NDE"]

    # entries = [LineElement(color=colors[l]) for l in (1, 2, 4)]
    # labels = ["true (LES)", "Convective adjustment", "NDE"]

    Legend(fig[0, :], entries, labels, framevisible=false, orientation=:horizontal, tellwidth=false, tellheight=true)

    Nt = length(times)
    record(fig, "$filepath_prefix.mp4", 1:Nt; framerate=fps) do n
        @info "Animating $filepath_prefix frame $n/$Nt..."
        time_index[] = n
    end

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

for id in [15] # keys(datasets)
    filepath_prefix = "movie2_comparing_profiles_simulation" * @sprintf("%02d", id)
    @info "Animating $filepath_prefix..."
    movie2_comparing_profiles(datasets[id], nde_solutions[id], kpp_solutions[id], convective_adjustment_solutions[id], T_scaling; filepath_prefix)
end

close(file)
