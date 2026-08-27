# SpeedyWeather physics-context probe for the wtune autotuner.
#
# Extends speedy_hotspot.jl: instead of only accumulated convective rain, runs a
# fast global PrimitiveWetModel forecast over the case window and extracts the
# fields that map to the tuner's knob/hotspot decisions, aggregated over the
# case domain bbox. Writes a compact JSON of DOMAIN AGGREGATES (mean/p90/max),
# which the stdlib-only tuner reads to refine radt cap, microphysics regime, etc.
#
# Usage: julia --project=jlenv speedy_probe.jl --hours 6 --lat0 22 --lat1 51 \
#              --lon0 -118 --lon1 -67 --out probe.json
#
# FIDELITY: idealized IC (not this case's ERA5) -> dynamically-plausible SHAPE/
# MAGNITUDE of the physics context, not the exact storm. Use for regime + scale.

using SpeedyWeather
using Statistics

argval(f, d) = (i = findfirst(==(f), ARGS); (i === nothing || i == length(ARGS)) ? d : ARGS[i+1])
fnum(f, d) = parse(Float64, argval(f, string(d)))

trunc_   = parse(Int, argval("--truncation", argval("--trunc", "31")))
nlayers  = parse(Int, argval("--nlayers", "8"))
hours    = parse(Int, argval("--hours", "6"))
lat0, lat1 = fnum("--lat0", -90), fnum("--lat1", 90)
lon0, lon1 = fnum("--lon0", -180), fnum("--lon1", 180)
outpath  = argval("--out", "speedy_probe.json")

sg = SpectralGrid(truncation=trunc_, nlayers=nlayers)
model = PrimitiveWetModel(sg)
sim = initialize!(model)
run!(sim, period=Hour(hours))

g = sim.variables.grid
pz = sim.variables.parameterizations
grid = g.u.grid
londs, latds = RingGrids.get_londlatds(grid)
londs = [lo > 180 ? lo - 360 : lo for lo in londs]
inbox = [(lat0 <= latds[i] <= lat1) && (lon0 <= londs[i] <= lon1) for i in eachindex(londs)]
midk = max(1, nlayers ÷ 2)

# helper: pull a single-layer, latest-time grid field's masked values.
# SpeedyWeather grid fields are XYZT (gridpoint, layer, time) 3D; parameterization
# fields are XY (1D) or XYT (2D, gridpoint x time, leapfrog keeps 2 time levels).
function masked1(field)
    a = try Array(field) catch; field.data end
    nd = ndims(a)
    v = nd == 1 ? a :
        nd == 2 ? a[:, end] :            # (gridpoint, time) -> latest time
                  a[:, midk, end]         # (gridpoint, layer, time) -> mid layer, latest time
    Float64.(v[inbox])
end
agg(v) = isempty(v) ? Dict("mean"=>nothing) :
    Dict("mean"=>round(mean(v),digits=4), "p90"=>round(quantile(v,0.9),digits=4),
         "max"=>round(maximum(v),digits=4), "min"=>round(minimum(v),digits=4))

# mid-level wind speed
uu = masked1(g.u); vv = masked1(g.v)
wspd = sqrt.(uu.^2 .+ vv.^2)

conv = masked1(pz.rain_convection)         # accumulated conv rain [m]
ls   = masked1(pz.rain_large_scale)        # accumulated large-scale rain [m]
totp = conv .+ ls
conv_frac = sum(totp) > 0 ? sum(conv) / sum(totp) : 0.0

result = Dict{String,Any}(
    "provider" => "speedyweather-probe",
    "version" => string(pkgversion(SpeedyWeather)),
    "hours" => hours, "trunc" => trunc_, "nlayers" => nlayers,
    "bbox" => [lat0, lat1, lon0, lon1],
    "npoints_in_box" => count(inbox),
    "midlevel" => midk,
    "note" => "idealized IC; regime + scale, not exact storm. domain aggregates over bbox.",
    "wind_midlevel_ms" => agg(wspd),
    "convective_fraction" => round(conv_frac, digits=3),
    "rain_convection_m" => agg(conv),
    "rain_large_scale_m" => agg(ls),
)
# best-effort optional fields (skip any that error on this version)
for (key, field) in [
        ("cloud_top", :cloud_top), ("sw_down_surface", :surface_shortwave_down),
        ("lw_down_surface", :surface_longwave_down), ("outgoing_longwave", :outgoing_longwave),
        ("cos_zenith", :cos_zenith), ("pbl_height", :boundary_layer_height),
        ("sensible_heat_flux", :sensible_heat_flux)]
    try
        result[key] = agg(masked1(getproperty(pz, field)))
    catch e
        result[key] = Dict("error" => string(e))
    end
end

open(outpath, "w") do io
    print(io, "{")
    ks = collect(keys(result)); n = length(ks)
    for (i,k) in enumerate(ks)
        print(io, "\"", k, "\":")
        v = result[k]
        if v isa Dict
            print(io, "{"); kk = collect(keys(v))
            for (j,k2) in enumerate(kk)
                print(io, "\"", k2, "\":", v[k2] === nothing ? "null" : v[k2]); j < length(kk) && print(io, ",")
            end
            print(io, "}")
        elseif v isa AbstractVector
            print(io, "[", join(v, ","), "]")
        elseif v isa AbstractString
            print(io, "\"", v, "\"")
        else
            print(io, v)
        end
        i < n && print(io, ",")
    end
    print(io, "}")
end
println("wrote ", outpath, " (", count(inbox), " pts in box, conv_frac=", round(conv_frac,digits=3),
        ", wind_mean=", round(mean(wspd),digits=1), " m/s)")
