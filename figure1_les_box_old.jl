# Credit: This script is a modified version of another script developed by Andre Souza (https://github.com/sandreza)

using Statistics
using Printf

using GLMakie
using Oceananigans

filepath = "free_convection_19/3d_with_halos.jld2"
ds = FieldDataset(filepath)

n = time_index = 3

fig = Figure()

ax = fig[1, 1] = LScene(fig)

clims = (19.6, 20)
cmap = :blues

Nx, Ny, Nz, Nt = size(ds["T"])
Lx, Ly, Lz = length(ds["T"].grid)
# xc, yc, zc = nodes(ds["T"])
T = interior(ds["T"])

xc = range(0, Lx, length=Nx)
yc = range(0, Ly, length=Ny)
zc = range(-Lz, 0, length=Nz)

T_west = T[1, :, :, n]
T_east = T[Nx, :, :, n]
T_south = T[:, 1, :, n]
T_north = T[:, Ny, :, n]
T_bottom = T[:, :, 1, n]
T_top = T[:, :, Nz, n]

surface!(ax, xc, zc, T_south,  transformation=(:xz,   0)), axis=(type=Axis3,)#, colorrange=clims, colormap=cmap)
surface!(ax, xc, zc, T_north,  transformation=(:xz,  Ly)), axis=(type=Axis3,)#, colorrange=clims, colormap=cmap)
surface!(ax, yc, zc, T_west,   transformation=(:yz,   0)), axis=(type=Axis3,)#, colorrange=clims, colormap=cmap)
surface!(ax, yc, zc, T_east,   transformation=(:yz,  Lx)), axis=(type=Axis3,)#, colorrange=clims, colormap=cmap)
surface!(ax, xc, yc, T_bottom, transformation=(:xy, -Lz)), axis=(type=Axis3,)#, colorrange=clims, colormap=cmap)
surface!(ax, xc, yc, T_top,    transformation=(:xy,   0)), axis=(type=Axis3,)#, colorrange=clims, colormap=cmap)

save("figure1_les_box.png", fig, px_per_unit=2)
# save("figure1_les_box.pdf", fig, pt_per_unit=2)

error("Barrier!")

#########################################################

catke_jl_file = jldopen("catke_state.jld2", "r+")
catke_t_keys = keys(catke_jl_file["timeseries"]["b"])

# get slices:
# bottom
jl_file_bottom = jldopen("processed_highres_bottom.jld2", "r")
t_keys_bottom = keys(jl_file_bottom["timeseries"]["b"])
# top
jl_file_top = jldopen("processed_highres_top.jld2", "r")
t_keys_top = keys(jl_file_top["timeseries"]["b"])
# east
jl_file_east = jldopen("processed_highres_east.jld2", "r")
t_keys_east = keys(jl_file_east["timeseries"]["b"])
# west
jl_file_west = jldopen("processed_highres_west.jld2", "r")
t_keys_west = keys(jl_file_west["timeseries"]["b"])
# north
jl_file_north = jldopen("processed_highres_north.jld2", "r")
t_keys_north = keys(jl_file_north["timeseries"]["b"])
# south
jl_file_south = jldopen("processed_highres_south.jld2", "r")
t_keys_south = keys(jl_file_south["timeseries"]["b"])

# get statistics
jl_file_averages = jldopen("processed_highres_averaged_state.jld2", "r")
t_keys_averages = keys(jl_file_averages["timeseries"]["b"])
t_keys = keys(jl_file_averages["timeseries"]["b"])

time_node=Node(10)

b = @lift(jl_file_averages["timeseries"]["b"][t_keys[$time_node]][1,1,:])
u = @lift(jl_file_averages["timeseries"]["u"][t_keys[$time_node]][1,1,:])
v = @lift(jl_file_averages["timeseries"]["v"][t_keys[$time_node]][1,1,:])

# wb = @lift(jl_file_averages["timeseries"]["wb"][t_keys[$time_node]][1,1,:])
wb = @lift(jl_file_averages["timeseries"]["wb"][t_keys[$time_node]][1,1,:])

uu = @lift(jl_file_averages["timeseries"]["uu"][t_keys[$time_node]][1,1,:])
vv = @lift(jl_file_averages["timeseries"]["vv"][t_keys[$time_node]][1,1,:])
ww = @lift(jl_file_averages["timeseries"]["ww"][t_keys[$time_node]][1,1,:])
full_e = @lift($uu + $vv + 0.5*($ww[1:end-1]+$ww[2:end]) - $u .* $u - $v .* $v)

catke_b = @lift(catke_jl_file["timeseries"]["b"][catke_t_keys[$time_node]][1,1,:])
catke_e = @lift(catke_jl_file["timeseries"]["e"][catke_t_keys[$time_node]][1,1,:])
catke_kc = @lift(catke_jl_file["timeseries"]["Kc"][catke_t_keys[$time_node]][1,1,:])
catke_Nz = catke_jl_file["grid"]["Nz"]
catke_Lz = catke_jl_file["grid"]["Lz"]
ghost = 1
catke_z = catke_jl_file["grid"]["zC"][ghost+1:catke_Nz+ghost]

# get out slices
fieldstring = "w"
ϕ_top = @lift(jl_file_top["timeseries"][fieldstring][t_keys_averages[$time_node]][:,:,1])
ϕ_bottom = @lift(jl_file_bottom["timeseries"][fieldstring][t_keys_averages[$time_node]][:,:,1])
ϕ_east = @lift(jl_file_east["timeseries"][fieldstring][t_keys_averages[$time_node]][1,:,:])
ϕ_west = @lift(jl_file_west["timeseries"][fieldstring][t_keys_averages[$time_node]][1,:,:])
ϕ_north = @lift(jl_file_north["timeseries"][fieldstring][t_keys_averages[$time_node]][:,1,:])
ϕ_south = @lift(jl_file_south["timeseries"][fieldstring][t_keys_averages[$time_node]][:,1,:])

Nx = jl_file_averages["grid"]["Nx"]
Ny = jl_file_averages["grid"]["Ny"]
Nz = jl_file_averages["grid"]["Nz"]

Lx = jl_file_averages["grid"]["Lx"]
Ly = jl_file_averages["grid"]["Ly"]
Lz = jl_file_averages["grid"]["Lz"]

xsurf = range(0, Lx,  length = Nx)
ysurf = range(0, Ly,  length = Ny)
zsurf = range(-Lz, 0, length = Nz)
zsurf2 = range(-Lz, 0, length = Nz+1)


clims = (-5e-3,5e-3)
zscale = 1
fig = Figure(resolution = (1920, 1080))
ax = fig[1:3,1:3] = LScene(fig, title= "Cooling")

# https://docs.juliaplots.org/latest/generated/colorschemes/
# colormap = :sun # :afmhot # :balance # Reverse(:bone_1)
colormap = :blues
# colormap = :balance
# edge 1
ϕedge1 = ϕ_south
GLMakie.surface!(ax, xsurf, zsurf2 .* zscale, ϕedge1, transformation = (:xz, 0),  colorrange = clims, colormap = colormap, show_axis=false)

# edge 2
ϕedge2 = ϕ_north
GLMakie.surface!(ax, xsurf, zsurf2 .* zscale, ϕedge2, transformation = (:xz, Ly),  colorrange = clims, colormap = colormap)

# edge 3
ϕedge3 = ϕ_west
GLMakie.surface!(ax, ysurf, zsurf2 .* zscale, ϕedge3, transformation = (:yz, 0),  colorrange = clims, colormap = colormap)

# edge 4
ϕedge4 = ϕ_east
GLMakie.surface!(ax, ysurf, zsurf2 .* zscale, ϕedge4, transformation = (:yz, Lx),  colorrange = clims, colormap = colormap)

# edge 5
ϕedge5 = ϕ_bottom
GLMakie.surface!(ax, xsurf, ysurf, ϕedge5, transformation = (:xy, -Lz *  zscale), colorrange = clims, colormap = colormap)

# edge 6
ϕedge6 = ϕ_top
GLMakie.surface!(ax, xsurf, ysurf, ϕedge6, transformation = (:xy, 0 *  zscale), colorrange = clims, colormap = colormap)


b_start = jl_file_averages["timeseries"]["b"][t_keys[2]]
xlims_b = extrema(b_start)

ax2 = fig[1,4] = Axis(fig, title= "⟨b⟩")
haverage = b
GLMakie.lines!(ax2, haverage, collect(zsurf), color = :green)
haverage2 = catke_b
GLMakie.scatter!(ax2, haverage2, catke_z)
xlims!(ax2, xlims_b)

ax2.xlabel = "Buoyancy [m/s²]"
ax2.xlabelsize = 25

ax3 = fig[2,4] = Axis(fig, title= "⟨e⟩")
haverage4 = catke_e
haverage5 = @lift(0.5 * $full_e)
GLMakie.lines!(ax3, haverage5, collect(zsurf), color = :green)
GLMakie.scatter!(ax3, haverage4, catke_z)
xlims!(ax3, (0, 2.5e-5))

ax3.xlabel = "Turbulent Kinetic Energy [m²/s²]"
ax3.xlabelsize = 25


ax4 = fig[3,4] = Axis(fig, title= "⟨w'b'⟩")
haverage3 = wb


wb_catke = @lift( -($catke_b[2:end] .* $catke_kc[2:end] - $catke_b[1:end-1] .* $catke_kc[1:end-1] ) ./ 4 )
z_avg =  0.5 * (catke_z[2:end] + catke_z[1:end-1])

GLMakie.lines!(ax4, haverage3, collect(zsurf2), color = :green)
GLMakie.scatter!(ax4, wb_catke, z_avg)

xlims!(ax4, (-0.2e-8, 1.5e-8))
ax4.xlabel = "Buoyancy Flux [m²/s³]"
ax4.xlabelsize = 25

for ax in [ax2, ax3, ax4]
    ax.titlesize = 40
    ax.ylabel = "Depth [m]"
    ax.ylabelsize = 25
end
#
Label(fig[4,2], @lift( "vertical velocity at day " * @sprintf("%1.1f ", jl_file_averages["timeseries"]["t"][t_keys[$time_node]] /86400  )), textsize = 50)
Label(fig[4,4], "horizontal averages", textsize = 50)
#=
ax.title = title
ax.titlesize = 40
ax.xlabel = "Latitude"
ax.ylabel = "Stretched Height"
ax.xlabelsize = 25
ax.ylabelsize = 25
ax.xticks = ([-80, -60, -30, 0, 30, 60, 80], ["80S", "60S", "30S", "0", "30N", "60N", "80N"])
=#
rotate_cam!(fig.scene.children[1], (0, π/4, 0))
display(fig)


iterations = 10:2840
# iterations = [collect(3:108)..., collect(110:570)...] # 109 data corrupted
record(fig, "preliminary_convection.mp4", iterations, framerate=60) do i
    time_node[] = i
    θ = 2π/iterations[end]/2
    rotate_cam!(fig.scene.children[1], (0, θ, 0))
    println("finishing ", i)
end

