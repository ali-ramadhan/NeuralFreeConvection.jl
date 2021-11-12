function simulation_color(id; alpha=1.0)
    colors = CairoMakie.Makie.wong_colors(alpha)
     1 <= id <= 9  && return colors[1]
    10 <= id <= 12 && return colors[2]
    13 <= id <= 15 && return colors[3]
    16 <= id <= 18 && return colors[4]
    19 <= id <= 21 && return colors[5]
    error("Invalid ID: $id")
end

function simulation_label(id)
     1 <= id <= 9  && return "training"
    10 <= id <= 12 && return "Qb interpolation"
    13 <= id <= 15 && return "Qb extrapolation"
    16 <= id <= 18 && return "N² interpolation"
    19 <= id <= 21 && return "N² extrapolation"
    error("Invalid ID: $id")
end
