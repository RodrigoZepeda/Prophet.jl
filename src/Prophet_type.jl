@doc raw"""
`ProphetModel()`

Contructs a `ProphetModel` mutable type of object. 

...
# Arguments (optional)

- `growth::String`: "linear", "logistic" or "flat" specifying the trend. *Default:* `"linear"`.

- `changepoints::Union{Vector{<: TimeType}, Vector{<: Real}, Nothing}`: Vector of dates at which to include 
    potential changepoints. If not specified,  potential changepoints are selected automatically.
    *Default:* `nothing`.

- `n_changepoints::Integer`: Number of potential changepoints to include. Not used if input `changepoints` is supplied. 
    If `changepoints` is not supplied, then `n.changepoints` potential changepoints are selected 
    uniformly from the first `changepoint.range` proportion of `df[:,:ds]`. *Default:* `25`.

- `changepoint_range::Float64`: Proportion of `history` in which trend changepoints will be estimated. 
    Defaults to `0.8` for the first 80%. Not used if `changepoints` is specified. *Default:* `0.8`.

- `yearly_seasonality::Union{String,Integer,Bool}`: Fit yearly seasonality. Can be "auto", `true`, `false`,
    or a number of Fourier terms to generate. *Default:* `"auto"`.

- `weekly_seasonality::Union{String,Integer,Bool}`: Fit weekly seasonality. Can be "auto", `true`, `false`,
    or a number of Fourier terms to generate. *Default:* `"auto"`.

- `daily_seasonality::Union{String,Integer,Bool}`: Fit daily seasonality. Can be "auto", `true`, `false`,
    or a number of Fourier terms to generate. *Default:* `"auto"`.
    
- `holidays::Union{DataFrame,Nothing}`: holidays `DataFrame` with columns `:holiday` (`String`) and 
    `ds` (any `TimeType` preferably: `DateTime` or `ZonedDateTime` or `Date`) and optionally columns 
    `lower_window` and `upper_window` which specify a range of days around the date to be 
    included as holiday. As an example, `lower_window=-2` will include 2 days prior to the date 
    as holidays. Also optionally can have a column `prior_scale` specifying the prior scale for each holiday.
    *Default:* `nothing`.

- `seasonality_mode::String`: "additive" or "multiplicative". *Default:* `"additive"`.

- `seasonality_prior_scale::Union{Vector{Float64}, Float64}`: Parameter modulating the strength of the
    seasonality model. Larger values allow the model to fit larger seasonal 
    fluctuations, smaller values dampen the seasonality. Can be specified for 
    individual seasonalities using `add_seasonality`. *Default:* `10.0`.

TODO: Create add seasonalty function

- `holidays_prior_scale::Float64`: Parameter modulating the strength of the holiday 
    components model, unless overridden in the holidays input. *Default:* `10.0`.

- `changepoint_prior_scale::Float64`: Parameter modulating the flexibility of the 
    automatic changepoint selection. Large values will allow many changepoints, 
    small values will allow few changepoints. *Default:* `0.05`.

- `mcmc_samples::Integer`: Integer, if greater than `0`, will do full Bayesian 
    inference with the specified number of MCMC samples. If `0`, will do MAP estimation.
    *Default:* `0`.

- `interval_width::Float64`: Numeric, width of the uncertainty intervals provided 
    for the forecast. If `mcmc.samples = 0`, this will be only the uncertainty in 
    the trend using the MAP estimate of the extrapolated generative model. If 
    `mcmc.samples > 0`, this will be integrated over all model parameters, which 
    will include uncertainty in seasonality. *Default:* `0.80`.

- `uncertainty_samples::Union{Integer, Bool}`: Number of simulated draws used to estimate 
    uncertainty intervals. Settings this value to `0` or `false` will disable 
    uncertainty estimation and speed up the calculation. *Default:* `1_000`.

# Arguments (internal don't use unless you know what you are doing)

- `specified_changepoints::Bool`: Flag indicating whether the user specified `changepoints`. 
    *Default:* `!isnothing(changepoints)`.

- `start::Union{Nothing,<: TimeType, <: Real}`: Initial time of the model with DataFrame `history`. 
    Equivalent to `min(history.ds)`. *Default:* `nothing`.

- `y_scale::Union{Nothing,<: Real}`: If a floor is given to the logistic model (`floor = history.floor`), 
    the scale corresponds  to `max(abs.(history.y - floor))` if no floor is given `floor = 0` is assumed.
    *Default:* `nothing`.    

- `logistic_floor::Union{Bool}`: Flag indicating whether a floor was provided to the logistic model 
    via column `:floor` in the `DataFrame` `history`. *Default:* `false`.    

- `t_scale::Union{Nothing,<: Real}`: Difference in seconds between the initial time of `DataFrame` 
    `history` and the final time: `t_scale = max(history.ds) - min(history.ds)` in seconds.
    *Default:* `nothing`.

TODO: Some are missing
- `country_holidays::Union{Nothing, String}`: Two letter code for the country of which to consider the holiday. 
    Country codes for the holidays follow the Python's holiday package convention (see https://pypi.org/project/holidays/)

- `history::Union{Nothing,DataFrame}`: Set during fitting, `DataFrame` of the input time series. Requires
    at least two columns: `:y`, a column with the time series and `:ds` a column of the time units.
    *Default:* `nothing`.

- `history_dates::Union{Nothing,Vector{<: TimeType},Vector{<: Real}}`: Column `:ds` of `history` containing
    the time units. *Default:* `nothing`.
    
...
"""
mutable struct ProphetModel
    # String 'linear', 'logistic', or 'flat' to specify a linear,
    # logistic or flat trend.
    growth::String

    # Vector of dates at which to include potential changepoints. If not specified, 
    # potential changepoints are selected automatically.
    changepoints::Union{Vector{<: TimeType}, Vector{<: Real}, Nothing}

    # Number of potential changepoints to include. Not used
    # if input `changepoints` is supplied. If `changepoints` is not supplied,
    # then n.changepoints potential changepoints are selected uniformly from the
    # first `changepoint.range` proportion of df[:,:ds].
    n_changepoints::Integer

    # Proportion of history in which trend changepoints
    # will be estimated. Defaults to 0.8 for the first 80%. Not used if
    # `changepoints` is specified.
    changepoint_range::Float64

    # Fit yearly seasonality. Can be 'auto', true, false,
    # or a number of Fourier terms to generate.
    yearly_seasonality::Union{String,Integer,Bool}

    # Fit weekly seasonality. Can be 'auto', TRUE, FALSE,
    # or a number of Fourier terms to generate.
    weekly_seasonality::Union{String,Integer,Bool}

    # Fit daily seasonality. Can be 'auto', TRUE, FALSE,
    # or a number of Fourier terms to generate.
    daily_seasonality::Union{String,Integer,Bool}
    
    # holidays DataFrame with columns holiday (string) and ds (DateTime or ZonedDateTime or Date)
    # and optionally columns lower_window and upper_window which specify a
    # range of days around the date to be included as holidays. lower_window=-2
    # will include 2 days prior to the date as holidays. Also optionally can have
    # a column prior_scale specifying the prior scale for each holiday.
    holidays::Union{DataFrame,Nothing}

    #  seasonality.mode 'additive' (default) or 'multiplicative'.
    seasonality_mode::String

    # Parameter modulating the strength of the
    # seasonality model. Larger values allow the model to fit larger seasonal
    # fluctuations, smaller values dampen the seasonality. Can be specified for
    # individual seasonalities using add_seasonality.
    seasonality_prior_scale::Union{Vector{Float64}, Float64}

    # Parameter modulating the strength of the holiday
    # components model, unless overridden in the holidays input.
    holidays_prior_scale::Float64

    # Parameter modulating the flexibility of the
    # automatic changepoint selection. Large values will allow many changepoints,
    # small values will allow few changepoints.
    changepoint_prior_scale::Float64

    # Integer, if greater than 0, will do full Bayesian
    # inference with the specified number of MCMC samples. If 0, will do MAP
    # estimation.
    mcmc_samples::Integer

    # Numeric, width of the uncertainty intervals provided
    # for the forecast. If mcmc.samples=0, this will be only the uncertainty in
    # the trend using the MAP estimate of the extrapolated generative model. If
    # mcmc.samples>0, this will be integrated over all model parameters, which
    # will include uncertainty in seasonality.
    interval_width::Float64

    # Number of simulated draws used to estimate
    # uncertainty intervals. Settings this value to 0 or False will disable
    # uncertainty estimation and speed up the calculation.
    uncertainty_samples::Union{Integer, Bool}

    #Flag indicating whether the user specified `changepoints`.
    specified_changepoints::Bool

    #Initial time of the model with DataFrame `df`. Equivalent to `min(df.df)`
    start::Union{Nothing,<: TimeType, <: Real}

    #If a floor is given to the logistic model, the scale corresponds to max(abs.(df.y - floor))
    #if no floor is given floor = 0 is assumed.
    y_scale::Union{Nothing,<: Real}

    #Flag indicating whether a floor was provided to the logistic model via column
    #`:floor` in the `DataFrame`
    logistic_floor::Union{Bool}

    #Difference in seconds between the initial time of DataFrame `df` and the final time. 
    #t_scale = max(df.ds) - min(df.ds) in seconds
    t_scale::Union{Nothing,<: Real}

    changepoints_t::Union{Nothing, Vector{<: Real}}

    seasonalities::AbstractDict

    extra_regressors::AbstractDict

    #Country for which to consider the holidays
    country_holidays::Union{Nothing, String}

    params::Union{Nothing,AbstractDict}

    #Set during fitting: DataFrame of the input time series
    history::Union{Nothing,DataFrame}

    #Set during fitting: vector of dates at which the history was measured
    history_dates::Union{Nothing,Vector{<: TimeType},Vector{<: Real}}

    train_holiday_names::Union{Nothing}

    train_component_cols::Union{Nothing, DataFrame}

    component_modes::Union{Nothing, Dict}

    #Additional arguments for fit
    fit_kwargs::Array

    #Default constructor
    function ProphetModel(;
        growth::String="linear",
        changepoints::Union{Vector{<:TimeType},Vector{<: Real}, Nothing}=nothing,
        n_changepoints::Integer=25,
        changepoint_range::Float64=0.8,
        yearly_seasonality::Union{String,Integer,Bool}="auto",
        weekly_seasonality::Union{String,Integer,Bool}="auto",
        daily_seasonality::Union{String,Integer,Bool}="auto",  
        holidays::Union{DataFrame,Nothing}=nothing,
        seasonality_mode::String="additive",
        seasonality_prior_scale::Union{Vector{Float64}, Float64}=10.0,
        holidays_prior_scale::Float64=10.0,
        changepoint_prior_scale::Float64=0.05,
        mcmc_samples::Integer=0,
        interval_width::Float64=0.80,
        uncertainty_samples::Union{Integer, Bool}=1_000,
        specified_changepoints::Bool = !isnothing(changepoints),
        start::Union{Nothing,<: TimeType, <: Real}=nothing,
        y_scale::Union{Nothing,Real}=nothing,
        logistic_floor::Union{Bool}=false,
        t_scale::Union{Nothing,Real}=nothing,
        changepoints_t::Union{Nothing, Vector{<: Real}}=nothing,
        seasonalities::AbstractDict=Dict(),
        extra_regressors::AbstractDict=Dict(),
        country_holidays::Union{Nothing, String}=nothing,
        params::Union{Nothing,AbstractDict}=nothing,
        history::Union{Nothing,DataFrame}=nothing,
        history_dates::Union{Nothing,Vector{<: TimeType},Vector{<: Real}}=nothing,
        train_holiday_names::Union{Nothing}=nothing,
        train_component_cols::Union{Nothing, DataFrame}=nothing,
        component_modes::Union{Nothing, Dict}=nothing,
        fit_kwargs::Array=Array{Any}(undef,0))
        return new(
            growth, 
            changepoints, 
            n_changepoints, 
            changepoint_range, 
            yearly_seasonality,
            weekly_seasonality, 
            daily_seasonality, 
            holidays, 
            seasonality_mode, 
            seasonality_prior_scale,
            changepoint_prior_scale, 
            holidays_prior_scale, 
            mcmc_samples, 
            interval_width, 
            uncertainty_samples, 
            specified_changepoints,
            start, 
            y_scale, 
            logistic_floor, 
            t_scale,
            changepoints_t,
            seasonalities,
            extra_regressors,
            country_holidays,
            params,
            history,
            history_dates,
            train_holiday_names,
            train_component_cols,
            component_modes,
            fit_kwargs
        )
    end
end
export ProphetModel

@doc raw"
`show()`

Render a `ProphetModel` object to an I/O stream by showing all the parameters and their values
...
# Arguments

- `io::IO`: The I/O stream to which `ProphetModel` will be printed.

- `m::ProphetModel`: A `ProphetModel`
...
"
function Base.show(io::IO, m::ProphetModel)
    stack = CrayonStack()
    print(io, push!(stack, Crayon(foreground = 208, bold = true)))
    print(io, stack, "PROPHET model with the following specification:\n")
    pop!(stack)
    for name in fieldnames(ProphetModel)
        mfield = getfield(m,name)
        print(io, push!(stack, Crayon(foreground = :blue, italics = true, faint = false, bold = false)))
        print(io, stack, "-" * string(name))
        pop!(stack)
        print(io, push!(stack, Crayon(foreground = :green, faint = true, bold = false)))
        print(io, "{",typeof(mfield), "}", ": ")
        pop!(stack)
        print(io, stack, mfield,"\n")
    end
end

@doc raw"
`validate_inputs()`

Validates the inputs of a `ProphetModel`
...
# Arguments
- `m::ProphetModel`: A `ProphetModel`
...
"
function validate_inputs(m::ProphetModel)

    if !(m.growth in ["linear" "logistic" "flat"])
        error("Invalid parameter growth = $(m.growth). Select 'linear', 'logistic' or 'flat'")
    end

    if (m.changepoint_range < 0) | (m.changepoint_range > 1)
        error("Parameter changepoint_range must be between 0 and 1 (i.e. the interval [0,1])")
    end

    if !(m.seasonality_mode in ["additive" "multiplicative"])
        error("seasonality_mode must be 'additive' or 'multiplicative'")
    end

    #Check holiday DataFrame
    if (!isnothing(m.holidays))
        if (typeof(holidays) != DataFrame)
            error("`holidays` must be a `DataFrame`")
        end

        if !("holiday" in names(holidays))
            error("`holidays` must have a :holiday column")
        end

        if !("ds" in names(holidays))
            error("`holidays` must have a :ds column")
        end

        #Set as date
        m.holidays.ds = set_date(m.holidays.ds)

        #Stop if missing
        if (any(isnothing.(m.holidays.ds)))
            error("Unable to parse date format in column `ds` of `holidays`. Convert to `DateTime` or `TimeZones.ZonedDateTime` "*
                  "format %Y-%m-%d or %Y-%m-%dT%H:%M:%S as in '2020-01-01' or '2020-01-01T22:00:12' "*
                  "and check that there are no missing values.")
        end

        has_lower = ("lower_window" in names(holidays))
        has_upper = ("upper_window" in names(holidays))

        if (has_lower + has_upper == 1)
            error("`holidays` must have both `:lower_window` and `:upper_window` or neither")
        end

        if (has_lower)
            if !(typeof(holidays.lower_window[1]) <: Real) || !(typeof(holidays.upper_window[1]) <: Real)
                error(":lower_window and :upper_window must be numeric (i.e subtypes of Real)")
            end

            max_lower = max(skipmissing(holidays.lower_window)...)
            min_upper = min(skipmissing(holidays.upper_window)...)

            if (max_lower > 0)
                error("Holiday :lower_window should be <= 0")
            end

            if (min_upper < 0)
                error("Holiday :upper_window should be >= 0")
            end
        end

        validate_column_name(m, unique(m.holidays.holiday); check_holidays = false)

    end
end
export validate_inputs

@doc raw"

`validate_column_name()`

Validates the name of seasonalities, holidays or regressors

...
# Arguments

- `m::ProphetModel`: A `ProphetModel`.

- `h::Union{String,Vector{String}}`: `String` or `Vector` of strings containing the names
    of the columns to validate.

- `check_holidays::Bool`: Whether to check holidays for the name (default `true`).

- `check_seasonalities::Bool`: Whether to check seasonality variables for the name (default `true`).

- `check_regressors::Bool`: Whether to check regressor variables for the name (default `true`).
...
"
function validate_column_name(m::ProphetModel, 
        h::Union{String,Vector{String}}; 
        check_holidays::Bool = true, 
        check_seasonalities::Bool = true, 
        check_regressors::Bool = true)

    if (typeof(h) == String)
        h = [h]
    end

    if (any(contains.(h, "_delim_")))
        error("Holiday name cannot contain `_delim_`")
    end

    reserved_names = ["trend", "additive_terms", "daily", "weekly", "yearly", "holidays", 
                      "zeros", "extra_regressors_additive", "yhat", "extra_regressors_multiplicative", 
                      "multiplicative_terms"]
    lower_reserved = reserved_names .* "_lower"
    upper_reserved = reserved_names .* "_upper"
    reserved_names = vcat(reserved_names, lower_reserved, upper_reserved, ["ds","y","cap","floor","y_scaled","cap_scaled"])

    if (any(in.(h, Ref(reserved_names))))
        error("Cannot use the following reserved name: $(h[in.(h, Ref(reserved_names))])")
    end

    if (check_holidays && !isnothing(m.holidays))
        holiday_names = unique(m.holidays.holiday)
        if (any(in.(h, Ref(holiday_names))))
            error("Name(s) $(h[in.(h, Ref(holiday_names))]) already used for a holiday")
        end
    end

    if (check_holidays && !isnothing(m.country_holidays))
        holiday_names = get_holiday_names(m.country_holidays)
        if (any(in.(h, Ref(holiday_names))))
            error("Name(s) $(h[in.(h, Ref(holiday_names))]) already used for a holiday in $(m.country_holidays).")
        end
    end

    if (check_seasonalities && length(m.seasonalities) > 0)
        seasonality_names = keys(m.seasonalities)
        if (any(in.(Symbol.(h), Ref(seasonality_names))))
            error("Name(s) $(h[in.(Symbol.(h), Ref(seasonality_names))]) already used for a seasonality")
        end
    end

    #FIXME: confront line 245 in prophet.R
    if (check_regressors && length(m.extra_regressors) > 0)
        regressor_names = keys(m.extra_regressors)
        if (any(in.(Symbol.(h), Ref(regressor_names))))
            error("Name(s) $(h[in.(Symbol.(h), Ref(regressor_names))]) already used for a regressor")
        end
    end

end
export validate_column_name

function prophet(df = nothing; 
    growth::String = "linear",
    changepoints::Union{Vector{<: TimeType},Vector{<: Real}, Nothing} = nothing,
    n_changepoints::Integer = 25,
    changepoint_range::Float64 = 0.8,
    yearly_seasonality::Union{String,Integer,Bool} = "auto",
    weekly_seasonality::Union{String,Integer,Bool} = "auto",
    daily_seasonality::Union{String,Integer,Bool}  = "auto",
    holidays::Union{DataFrame,Nothing} = nothing,
    seasonality_mode::String = "additive",
    seasonality_prior_scale::Union{Vector{Float64}, Float64} = 10.0,
    holidays_prior_scale::Float64 = 10.0,
    changepoint_prior_scale::Float64 = 0.05,
    mcmc_samples::Integer = 0,
    interval_width::Float64 = 0.80,
    uncertainty_samples::Union{Integer, Bool} = 1_000,
    fit::Bool = true,
    kwargs...)

    if (!isnothing(changepoints))
        n_changepoints = length(changepoints)
    end

    m = ProphetModel(;
                growth = growth, 
                changepoints = changepoints, 
                n_changepoints = n_changepoints, 
                changepoint_range = changepoint_range, 
                yearly_seasonality = yearly_seasonality,
                weekly_seasonality = weekly_seasonality, 
                daily_seasonality = daily_seasonality, 
                holidays = holidays, 
                seasonality_mode = seasonality_mode, 
                seasonality_prior_scale = seasonality_prior_scale,
                changepoint_prior_scale = changepoint_prior_scale, 
                holidays_prior_scale = holidays_prior_scale, 
                mcmc_samples = mcmc_samples, 
                interval_width = interval_width, 
                uncertainty_samples = uncertainty_samples)

    #Validate the inputs
    validate_inputs(m)

    if fit && !isnothing(df)
        m = fit_prophet(m, df; kwargs...)
    end
end
export prophet

