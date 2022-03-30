using Printf
using JLD2
using ColorSchemes
using CairoMakie
using OceanTurb
using FreeConvection

using Flux.Losses: mse
using Oceananigans: interior, znodes
using Oceananigans.Units: days

function movie2_comparing_profiles(ds, nde_sol, kpp_sol, convective_adjustment_sol, T_scaling; filepath_prefix, colors, resolution=(1280, 720), fps=30)
    T = ds["T"]
    wT = ds["wT"]
    Nt = size(T, 4)
    zc = znodes(T)
    zf = znodes(wT)
    times = T.times ./ days

    loss(T, T̂) = mse(T_scaling.(T), T_scaling.(T̂))

    loss_nde = [loss(interior(T)[1, 1, :, n], nde_sol.T[:, n]) for n in 1:Nt]
    loss_kpp = [loss(interior(T)[1, 1, :, n], kpp_sol.T[:, n]) for n in 1:Nt]
    loss_ca  = [loss(interior(T)[1, 1, :, n], convective_adjustment_sol.T[:, n]) for n in 1:Nt]

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

    lines!(ax, wT_les, zf, linewidth=4, color=colors[1])
    lines!(ax, wT_ca, zf, linewidth=4, color=colors[2])
    lines!(ax, wT_kpp, zf, linewidth=4, color=colors[3])
    lines!(ax, wT_nde, zf, linewidth=4, color=colors[4])

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

    lines!(ax, T_les, zc, linewidth=4, color=colors[1])
    lines!(ax, T_ca, zc, linewidth=4, color=colors[2])
    lines!(ax, T_kpp, zc, linewidth=4, color=colors[3])
    lines!(ax, T_nde, zc, linewidth=4, color=colors[4])

    ## Right panel: losses

    ax = fig[1, 3] = Axis(fig, xlabel="Simulation time (days)", ylabel="Loss", yscale=log10)

    xlims!(extrema(times)...)
    ylims!(1e-5, 1e-2)

    ax.xgridvisible = false
    ax.ygridvisible = false

    lp_ca = lines!(ax, times, loss_ca, linewidth=4, color=colors[2])
    lp_kpp = lines!(ax, times, loss_kpp, linewidth=4, color=colors[3])
    lp_nde = lines!(ax, times, loss_nde, linewidth=4, color=colors[4])

    entries = [LineElement(color=colors[l], linewidth=4) for l in 1:4]
    labels = ["Large eddy simulation", "Convective adjustment", "K-Profile Parameterization", "Neural differential equation"]
    Legend(fig[0, :], entries, labels, nbanks=2, orientation=:horizontal, framevisible=false, tellwidth=false, tellheight=true)

    Nt = length(times)
    record(fig, "$filepath_prefix.mp4", 1:Nt; framerate=fps) do n
        @info "Animating $filepath_prefix frame $n/$Nt..."
        time_index[] = n

        lp_ca[1] = vcat(times[1:n], fill(NaN, Nt-n))
        lp_ca[2] = vcat(loss_ca[1:n], fill(NaN, Nt-n))

        lp_kpp[1] = vcat(times[1:n], fill(NaN, Nt-n))
        lp_kpp[2] = vcat(loss_kpp[1:n], fill(NaN, Nt-n))

        lp_nde[1] = vcat(times[1:n], fill(NaN, Nt-n))
        lp_nde[2] = vcat(loss_nde[1:n], fill(NaN, Nt-n))
    end

    return nothing
end

Nz = 32

ids_train = 1:9
ids_test = setdiff(FreeConvection.SIMULATION_IDS, ids_train)

# data = load_data(ids_train, ids_test, Nz)
datasets = data.coarse_datasets

file = jldopen("solutions_and_history.jld2", "r")

nde_solutions = file["nde"]
convective_adjustment_solutions = file["convective_adjustment"]
T_scaling = file["T_scaling"]

# kpp_solutions = file["kpp"]
kpp_parameters = OceanTurb.KPP.Parameters(CSL=2/3, CNL=5.0, Cb_T=0.16, CKE=8.0)
kpp_solutions = Dict(id => free_convection_kpp(ds, parameters=kpp_parameters) for (id, ds) in data.coarse_datasets)

close(file)

colors = ColorSchemes.julia.colors
permute!(colors, [4, 2, 1, 3])

for id in keys(datasets)
    filepath_prefix = "movie2_comparing_profiles_simulation" * @sprintf("%02d", id)
    @info "Animating $filepath_prefix..."
    movie2_comparing_profiles(datasets[id], nde_solutions[id], kpp_solutions[id], convective_adjustment_solutions[id], T_scaling; filepath_prefix, colors)
end
