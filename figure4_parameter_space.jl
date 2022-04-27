using Statistics
using ColorTypes
using ColorSchemes
using CairoMakie
using FreeConvection

Nz = 32

ids_train = 1:9
ids_test = setdiff(FreeConvection.SIMULATION_IDS, ids_train)

data = load_data(ids_train, ids_test, Nz)
datasets = data.datasets

ρ₀ = 1027
cₚ = 4000
α = 2e-4
g = 9.81
toWm⁻² = ρ₀ * cₚ / (α * g)

Qbs = 1e-8 .* [
    1, 3,   5,
    1, 3,   5,
    1, 3,   5,
    4, 2,   4,
    6, 0.5, 6,
    1, 3,   5,
    1, 3,   5
]

function label(id)
     1 <= id <= 9  && return "training"
    10 <= id <= 12 && return "Qb interpolation"
    13 <= id <= 15 && return "Qb extrapolation"
    16 <= id <= 18 && return "N² interpolation"
    19 <= id <= 21 && return "N² extrapolation"
    error("Invalid ID: $id")
end

let
    fig = Figure()
    ax = fig[1, 1] = Axis(fig, xlabel="Surface buoyancy flux (m²/s³) ", ylabel="Stratification (1/s²)")

    Qb_training_min = Inf
    Qb_training_max = -Inf
    N²_training_min = Inf
    N²_training_max = -Inf

    for (n, Qb) in enumerate(Qbs)
        T = datasets[n]["T"]
        Nz = T.grid.Nz
        Lz = T.grid.Lz
        Δz = Lz / Nz
        T₀ = T[1, 1, 1:Nz, 1]

        N² = diff(T₀) / Δz * (α * g)
        N²_mean = mean(N²) # Use N²_surface instead?
        N²_min, N²_max = extrema(N²)
        N²_midpoint = (N²_min + N²_max) / 2
        N²_half_interval = N²_max - N²_midpoint

        scatter!(ax, [Qb], [N²_midpoint], color=simulation_color(n))
        text!(ax, " " * string(n), position=(Qb, N²_midpoint), color=simulation_color(n), align=(:left, :center))

        errorbars!([Qb], [N²_midpoint], [N²_half_interval], [N²_half_interval], color=simulation_color(n), whiskerwidth=10)

        if n <= 9
            Qb < Qb_training_min && (Qb_training_min = Qb)
            Qb > Qb_training_max && (Qb_training_max = Qb)
            N²_min < N²_training_min && (N²_training_min = N²_min)
            N²_max > N²_training_max && (N²_training_max = N²_max)
        end
    end

    band!(ax, [Qb_training_min, Qb_training_max], N²_training_min, N²_training_max, color=simulation_color(1, alpha=0.25))

    entry_ids = (1, 10, 13, 16, 19)
    entries = [MarkerElement(marker=:circle, color=simulation_color(id), strokecolor=:transparent, markersize=15) for id in entry_ids]
    labels = [label(id) for id in entry_ids]
    Legend(fig[1, 2], entries, labels, framevisible=false)

    ax.xticks = ([0, 2e-8, 4e-8, 6e-8], ["0", "2×10⁻⁸", "4×10⁻⁸", "6×10⁻⁸"])
    ax.yticks = ([0, 1e-5, 2e-5, 3e-5], ["0", "1×10⁻⁵", "2×10⁻⁵", "3×10⁻⁵"])

    xlims!(ax, 0, 7e-8)
    ylims!(ax, 0, 3.5e-5)

    save("figure4_parameter_space.png", fig, px_per_unit=2)
    save("figure4_parameter_space.pdf", fig, pt_per_unit=2)

    return fig
end

let
    fig = Figure()

    xticksvisible = false
    yticksvisible = false
    xticklabelsvisible = false
    yticklabelsvisible = false
    xgridvisible = false
    ygridvisible = false
    leftspinevisible = false
    rightspinevisible = false
    topspinevisible = false
    bottomspinevisible = false
    ax_kwargs = (; xticksvisible, yticksvisible, xticklabelsvisible, yticklabelsvisible, xgridvisible, ygridvisible,
                   leftspinevisible, rightspinevisible, topspinevisible, bottomspinevisible)
    ax_kwargs_last_row = (; yticksvisible, yticklabelsvisible, xgridvisible, ygridvisible,
                            leftspinevisible, rightspinevisible, topspinevisible)

    xaxisposition = :top
    ax1_kwargs_first_row = (; xaxisposition, yticksvisible, yticklabelsvisible, xgridvisible, ygridvisible,
                              leftspinevisible, rightspinevisible, bottomspinevisible)

    for (n, Qb) in enumerate(Qbs)
        T = datasets[n]["T"]
        Nz = T.grid.Nz
        Lz = T.grid.Lz
        Δz = Lz / Nz
        T₀ = T[1, 1, Int(Nz/2):Nz, 1]
        N² = diff(T₀) / Δz * (α * g)

        if n == 1
            ax1_kwargs_row = ax1_kwargs_first_row
            ax2_kwargs_row = ax_kwargs
            xlabel_ax1 = "Surface cooling (W/m²)"
        else
            ax1_kwargs_row = ax2_kwargs_row = n < length(Qbs) ? ax_kwargs : ax_kwargs_last_row
            xlabel_ax1 = n < length(Qbs) ? "" : "Surface buoyancy flux (m²/s³)"
        end

        xlabel_ax2 = n < length(Qbs) ? "" : "Stratification (1/s²)"

        ax1 = fig[n, 1] = Axis(fig, xlabel=xlabel_ax1; ax1_kwargs_row...)

        if n == 1
            scatter!(ax1, [Qb * toWm⁻²], [1], color=simulation_color(n))
            text!(ax1, string(n), position=(-0.2e-8 * toWm⁻², 1), align=(:center, :center), color=simulation_color(n))
            vlines!(ax1, [1e-8, 3e-8, 5e-8] .* toWm⁻², color=simulation_color(1, alpha=0.5), linestyle=:dash)
            xlims!(ax1, [-0.5e-8, 6.5e-8] .* toWm⁻²)
        else
            scatter!(ax1, [Qb], [1], color=simulation_color(n))
            text!(ax1, string(n), position=(-0.2e-8, 1), align=(:center, :center), color=simulation_color(n))
            vlines!(ax1, [1e-8, 3e-8, 5e-8], color=simulation_color(1, alpha=0.5), linestyle=:dash)
            xlims!(ax1, (-0.5e-8, 6.5e-8))
        end

        ax2 = fig[n, 2] = Axis(fig, xlabel=xlabel_ax2; ax2_kwargs_row...)
        bins = range(0, 3e-5, length=26)
        # hist!(ax2, N², color=simulation_color(n); bins)
        density!(ax2, N², bandwidth=1e-6, color=simulation_color(n))
        xlims!(ax2, (0, 3.5e-5))

        if n == 1
            ax1.xticks = ([20, 60, 100], ["20", "60", "100"])
        elseif n == length(Qbs)
            ax1.xticks = ([1e-8, 3e-8, 5e-8], ["1×10⁻⁸", "3×10⁻⁸", "5×10⁻⁸"])
            ax2.xticks = ([0, 1e-5, 2e-5, 3e-5], ["0", "1×10⁻⁵", "2×10⁻⁵", "3×10⁻⁵"])
        end
    end

    rowgap!(fig.layout, 0)

    entry_ids = (1, 10, 13, 16, 19)
    entries = [PolyElement(color=simulation_color(id)) for id in entry_ids]
    labels = [label(id) for id in entry_ids]
    Legend(fig[0, :], entries, labels, orientation=:horizontal, framevisible=false)

    save("figure4_parameter_space_v2.png", fig, px_per_unit=2)
    save("figure4_parameter_space_v2.pdf", fig, pt_per_unit=2)

    return fig
end
