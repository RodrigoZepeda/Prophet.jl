@doc raw"""
`get_valid_index()`

Auxiliary function to filter a `DataFrame` and get the index of the rows with 
    invalid values (missing/nothing/nan/infinity) in columns `col`.
...
# Arguments
- `df::DataFrame`: DataFrame to filter
- `col::Union{Symbol, String, Vector{Symbol},Vector{String}}` Column or column vector of columns to filter
- `negate::Bool`: if `negate = false` (defualt) it filters invalid values; if `negate = true` it keeps invalid values
...
"""
function get_valid_index(df::DataFrame, col::Union{Symbol, String, Vector{Symbol},Vector{String}}; negate::Bool = false)

    if !(typeof(col) <: Vector)
        col = [col]
    end

    if (negate)
        #Return invalid inputs
        temp = any.(
                eachrow(
                    mapcols(z -> isnothing.(z) .|| ismissing.(z) .|| isnan.(z) .|| isinf.(z), 
                            df[:,col])))
    else
        temp = all.(
                eachrow(
                    mapcols(z -> .!isnothing.(z) .&& .!ismissing.(z) .&& .!isnan.(z) .&& .!isinf.(z), 
                            df[:,col])))
    end
    return temp
end

@doc raw"
`drop_invalid()`, `drop_invalid!()`, `keep_invalid()`, `keep_invalid!()`

Function to filter a `DataFrame` and drop (`drop_invalid`) or keep (`keep_invalid`) 
rows with invalid values (`missing`/`nothing`/`NaN`/`Inf`) in columns `col`.

...
# Arguments
- `df::DataFrame`: DataFrame to filter
- `col::Union{Symbol, String, Vector{Symbol},Vector{String}}` Column or column vector of columns to filter
- `negate::Bool`: if `negate = false` (defualt) it filters invalid values; if `negate = true` it keeps invalid values
...

Examples
```julia-repl
julia> df = DataFrame(x = [missing, 1, 2, Inf, NaN, nothing, 3], y = 1:7)
julia> drop_invalid(df, [:x,:y])
3×2 DataFrame
 Row │ x        y     
     │ Union…?  Int64 
─────┼────────────────
   1 │ 1.0          2
   2 │ 2.0          3
   3 │ 3.0          7
julia> keep_invalid(df, [:x,:y])   
4×2 DataFrame
 Row │ x        y     
     │ Union…?  Int64 
─────┼────────────────
   1 │ missing      1
   2 │ Inf          4
   3 │ NaN          5
   4 │              6
julia> drop_invalid(df, [:x,:y]; negate = true) #same thing as keep
4×2 DataFrame
 Row │ x        y     
     │ Union…?  Int64 
─────┼────────────────
   1 │ missing      1
   2 │ Inf          4
   3 │ NaN          5
   4 │              6
julia> drop_invalid(df, :y) 
7×2 DataFrame
 Row │ x        y     
     │ Union…?  Int64 
─────┼────────────────
   1 │ missing      1
   2 │ 1.0          2
   3 │ 2.0          3
   4 │ Inf          4
   5 │ NaN          5
   6 │              6
   7 │ 3.0          7
julia> keep_invalid(df, :y) 
0×2 DataFrame
 Row │ x        y     
     │ Union…?  Int64 
─────┴──────────────── 
julia> df = DataFrame(x = [missing, 1, 2, Inf, NaN, nothing, 3], y = 1:7)
julia> drop_invalid!(df, :x)
julia> df
3×2 DataFrame
Row │ x        y     
    │ Union…?  Int64 
─────┼────────────────
    1 │ 1.0          2
    2 │ 2.0          3
    3 │ 3.0          7
julia> df = DataFrame(x = [missing, 1, 2, Inf, NaN, nothing, 3], y = 1:7)
julia> keep_invalid!(df, :x)
julia> df
4×2 DataFrame
 Row │ x        y     
     │ Union…?  Int64 
─────┼────────────────
   1 │ missing      1
   2 │ Inf          4
   3 │ NaN          5
   4 │              6
```
"
function drop_invalid(df::DataFrame, col::Union{Symbol, String, Vector{Symbol},Vector{String}}; negate::Bool = false) 
    temp = get_valid_index(df, col; negate = negate)
    return df[temp,:]
end
export drop_invalid

@doc (@doc drop_invalid)
function drop_invalid!(df::DataFrame, col::Union{Symbol, String, Vector{Symbol},Vector{String}}; negate::Bool = false) 
    temp = get_valid_index(df, col; negate = negate)
    keepat!(df, temp)
    return nothing
end
export drop_invalid!

@doc (@doc drop_invalid)
function keep_invalid(df::DataFrame, col::Union{Symbol, String, Vector{Symbol},Vector{String}})
    drop_invalid(df, col; negate=true)
end
export keep_invalid

@doc (@doc drop_invalid)
function keep_invalid!(df::DataFrame, col::Union{Symbol, String, Vector{Symbol},Vector{String}})
    drop_invalid!(df, col; negate=true)
end
export keep_invalid!

@doc raw"""
`set_date()`

Function to attempt to format string or vector entries as `TimeType` (preferably `DateTime`)

...
# Arguments
- `ds::Union{T, AbstractArray{T}, OrdinalRange{T}}`: Dates to attempt to parse as `TimeType`
where `T <: Union{TimeType, Real, String}` 
...

Examples
```julia-repl
julia> set_date("2020-01-01")
2020-01-01T00:00:00
julia> set_date(Dates.Date(1980,1,1))
1980-01-01
julia> set_date(Dates.Date(2014):Month(1):Dates.Date(2015))
Date("2014-01-01"):Month(1):Date("2015-01-01")
julia> set_date(Dates.DateTime(2014):Month(1):Dates.DateTime(2015))
DateTime("2014-01-01T00:00:00"):Month(1):DateTime("2015-01-01T00:00:00")
julia> set_date(Dates.DateTime("2014-08-11T10:09:12"))
set_date(Dates.DateTime("2014-08-11T10:09:12"))
julia> set_date(1:10)
1:10
julia> set_date(["2020-01-01","2020-02-02"])
2-element Vector{DateTime}:
 2020-01-01T00:00:00
 2020-02-02T00:00:00
```
"""
function set_date(ds::Union{T, AbstractArray{T}, OrdinalRange{T}}) where {T <: Union{TimeType, Real, String}}

    if (typeof(ds) <: AbstractArray || typeof(ds) <: OrdinalRange) && length(ds) < 1
        @warn "ds is of length $(length(ds))."
        return nothing
    end

    if typeof(ds) <: Union{<: TimeType, AbstractArray{<: TimeType}, OrdinalRange{<: TimeType}}
        return ds
    elseif (typeof(ds) <: Union{<: Real, AbstractArray{Real}, OrdinalRange{<: Real}})
        @warn "`ds` is not of type Date. Consider using `DateTime` or `TimeZones.ZonedDateTime` format"
        return ds
    else
        return tryparse.(DateTime, ds);
    end
end
export set_date

@doc raw"""
`time_diff()`

Function to estimate difference as `date1 - date2`.
The function attempts to `convert` the difference to `units` via
    `convert(units, (date1 - date2)).value`.
If the units don't return an integer a second attempt is made for the
following units:
    - `Dates.Millisecond`
    - `Dates.Second`
    - `Dates.Minute`
    - `Dates.Hour`
    - `Dates.Day`
    - `Dates.Month`
    - `Dates.Quarter`
    - `Dates.Year`
by considering `30.5` day months and `365` day years as well as `4*30.5` day quarters.

...
# Arguments
- `date1::Union{<: Real, <: TimeType}`: First date for the difference `date1 - date2`.  
- `date2::Union{<: Real, <: TimeType}`: Second date for the difference `date1 - date2`.
- `units::DataType`: Units in which to convert the difference `date1 - date2` to.
...

Examples
```julia-repl
julia> time_diff(3,1)
┌ Warning: Date type is real. Assuming it represents days. Use `DateTime` to better specify dates and avoid this warning
└ @ Main ~/Prophet/src/aux.jl:106
-172800
julia> time_diff(Date("2020-01-02"),Date("2020-01-01"); units = Dates.Second)
86400
julia> time_diff(DateTime("2020-01-04"),DateTime("2020-01-02"))
172800
julia> time_diff(DateTime("2020-05-01T00:00:01"),DateTime("2020-01-02"))
10368001
julia> time_diff(DateTime("2020-05-01T00:00:01"),DateTime("2020-01-02"); units = Dates.Month)
┌ Warning: Considering 30.5 day months. Use different units instead.
└ @ Main ~/Prophet/src/aux.jl:122
3.934426608986035
julia> time_diff(ZonedDateTime(2014, 5, 30, 21, tz"UTC-4"),ZonedDateTime(2014, 5, 30, 21, tz"UTC"))
14400
```
"""
function time_diff(date1::T, date2::T; units::DataType = Dates.Second) where {T<:Union{<: Real, <: TimeType}}
    if (typeof(date1) <: Real)
        @warn "Date type is real. Assuming it represents days. Use `DateTime` to better specify dates and avoid this warning"
        datedif = (date2 - date1)*86_400 #If real assume days and convert to seconds
    else 
        try
            datedif = convert(units, (date1 - date2)).value
        catch e
            datedif = convert(Dates.Millisecond, Dates.CompoundPeriod(date1 - date2)).value
            if (units == Dates.Second)
                datedif = datedif / 1_000
            elseif (units == Dates.Minute)
                datedif = datedif / (60*1_000)
            elseif (units == Dates.Hour)
                datedif = datedif / (60*60*1_000)
            elseif (units == Dates.Day)
                datedif = datedif / (24*60*60*1_000)
            elseif (units == Dates.Month)
                @warn "Considering 30.5 day months. Use different units instead."
                datedif = datedif / (30.5*24*60*60*1_000)
            elseif (units == Dates.Quarter)
                datedif = datedif / (4*30.5*24*60*60*1_000)
            elseif (units == Dates.Year)
                @warn "Considering 365 day Years. Use different units instead."
                datedif = datedif / (365*24*60*60*1_000)
            else
                error("Invalid date unable to get date difference")
            end
        end
    end
    return datedif
end
export time_diff