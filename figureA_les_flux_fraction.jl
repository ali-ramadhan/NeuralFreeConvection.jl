using Printf
using CairoMakie
using Oceananigans
using FreeConvection

using Oceananigans.Units: day

function figureA_plot_les_heat_flux_fraction(datasets; filepath_prefix)
    t = datasets[1]["T"].times ./ day

    fig = Figure()

    ax = fig[1, 1] = Axis(fig, xlabel="Simulation time (days)", ylabel="|κₑ∂zT| / ( |w'T'| + |κₑ∂zT| )")

    for id in FreeConvection.SIMULATION_IDS
        ds = datasets[id]
        advective_heat_flux = sum(ds["wT"].data[1, 1, :, :] .|> abs, dims=1)[:]
        diffusive_heat_flux = sum(ds["κₑ_∂z_T"].data[1, 1, :, :] .|> abs, dims=1)[:]
        total_heat_flux = advective_heat_flux .+ diffusive_heat_flux
        les_flux_fraction = diffusive_heat_flux ./ total_heat_flux

        linestyle = 1 <= id <= 9 ? :solid : :dash
        lines!(ax, t, les_flux_fraction, linewidth=2, label="simulation $id"; linestyle)
    end

    xlims!(extrema(t)...)

    fig[1, 2] = Legend(fig, ax, framevisible = false)

    save("$filepath_prefix.png", fig, px_per_unit=2)
    save("$filepath_prefix.pdf", fig, pt_per_unit=2)

    return nothing
end

Nz = 32

ids_train = 1:9
ids_test = setdiff(FreeConvection.SIMULATION_IDS, ids_train)

data = load_data(ids_train, ids_test, Nz)
datasets = data.coarse_datasets

figureA_plot_les_heat_flux_fraction(datasets, filepath_prefix="figureA_les_flux_fraction")
