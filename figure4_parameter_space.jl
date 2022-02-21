using Statistics
using CairoMakie
using FreeConvection

using CairoMakie.Makie: wong_colors

Nz = 32

ids_train = 1:9
ids_test = setdiff(FreeConvection.SIMULATION_IDS, ids_train)

data = load_data(ids_train, ids_test, Nz)
datasets = data.datasets

α = 2e-4
g = 9.81

Qbs = 1e-8 .* [
    1, 3,   5,
    1, 3,   5,
    1, 3,   5,
    4, 2,   4,
    6, 0.5, 6,
    1, 3,   5,
    1, 3,   5
]

function color(id; alpha=1.0)
    colors = CairoMakie.Makie.wong_colors(alpha)
     1 <= id <= 9  && return colors[1]
    10 <= id <= 12 && return colors[2]
    13 <= id <= 15 && return colors[3]
    16 <= id <= 18 && return colors[4]
    19 <= id <= 21 && return colors[5]
    error("Invalid ID: $id")
end

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

        scatter!(ax, [Qb], [N²_midpoint], color=color(n))
        text!(ax, " " * string(n), position=(Qb, N²_midpoint), color=color(n), align=(:left, :center))

        errorbars!([Qb], [N²_midpoint], [N²_half_interval], [N²_half_interval], color=color(n), whiskerwidth=10)

        if n <= 9
            Qb < Qb_training_min && (Qb_training_min = Qb)
            Qb > Qb_training_max && (Qb_training_max = Qb)
            N²_min < N²_training_min && (N²_training_min = N²_min)
            N²_max > N²_training_max && (N²_training_max = N²_max)
        end
    end

    band!(ax, [Qb_training_min, Qb_training_max], N²_training_min, N²_training_max, color=color(1, alpha=0.25))

    entry_ids = (1, 10, 13, 16, 19)
    entries = [MarkerElement(marker=:circle, color=color(id), strokecolor=:transparent, markersize=15) for id in entry_ids]
    labels = [label(id) for id in entry_ids]
    Legend(fig[1, 2], entries, labels, framevisible=false)

    ax.xticks = ([0, 2e-8, 4e-8, 6e-8], ["0", "2×10⁻⁸", "4×10⁻⁸", "6×10⁻⁸"])
    ax.yticks = ([0, 1e-5, 2e-5, 3e-5], ["0", "1×10⁻⁵", "2×10⁻⁵", "3×10⁻⁵"])

    xlims!(ax, 0, 7e-8)
    ylims!(ax, 0, 3.5e-5)

    save("figure4_parameter_space.png", fig, px_per_unit=2)
    save("figure4_parameter_space.pdf", fig, pt_per_unit=2)
end
