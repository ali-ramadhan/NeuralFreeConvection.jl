using ColorTypes
using ColorSchemes

minimum_nonzero(xs...) = min([minimum(filter(!iszero, x)) for x in xs]...)
maximum_nonzero(xs...) = max([maximum(filter(!iszero, x)) for x in xs]...)

function simulation_color(id, colors=circshift(ColorSchemes.Set1_5.colors, -1); alpha=1.0)
    1 <= id <= 9  && return RGBA(colors[1], alpha)
   10 <= id <= 12 && return RGBA(colors[2], alpha)
   13 <= id <= 15 && return RGBA(colors[3], alpha)
   16 <= id <= 18 && return RGBA(colors[4], alpha)
   19 <= id <= 21 && return RGBA(colors[5], alpha)
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
