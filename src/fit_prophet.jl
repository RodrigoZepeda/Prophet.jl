#Sets model_model scaling factors using df
@doc raw"""
Sets model_model scaling factors for normalizing the inputs and regressors as
    well as the time scale of the model_model.
"""
function initialize_scales_fn!(m::ProphetModel, initialize_scales::Bool, df::DataFrame)
    if (initialize_scales)
        
        if (m.growth == "logistic") && ("floor" in names(df))
            m.logistic_floor = true
            model_model_floor = df.floor
        else
            model_model_floor = zeros(nrow(df))
        end

        m.y_scale = maximum(abs.(df.y - model_model_floor))
        if (m.y_scale == 0)
            m.y_scale = 1.0
        end

        m.start   = minimum(df.ds)
        m.t_scale = time_diff(maximum(df.ds), m.start; units = Dates.Second)

        
        for name in keys(m.extra_regressors)
            standardize = m.extra_regressors[name]["standardize"];
            n_vals      = length(unique(df[!,name]))

            #Don't standardize if unable to estimate variance
            if (n_vals < 2)
                standardize = false
            end

            #Check if binary variables
            is_binary = (sort(unique(df[!,name])) == [0,1])
            if (standardize == "auto")
                if (n_vals == 2 && is_binary)
                    standardize = false
                else
                    standardize = true
                end
            end

            if (standardize)               
                m.extra_regressors[name]["μ"] = mean(df[!,name])
                m.extra_regressors[name]["σ"] = std(df[!,name])
            end
        end
    end
end
export initialize_scales_fn!


function flat_growth_init(df::DataFrame)
    #Initialize the rate
    k = 0
    #And the offset
    m = mean(df[!,:y_scaled])
    return k, m
end
export flat_growth_init

function linear_growth_init(df::DataFrame)
    i0 = argmin(df[!,:ds])
    i1 = argmax(df[!,:ds])
    TDIF = df[i1,:t] - df[i0,:t]

    #Initialize the rate
    k = (df[i1,:y:scaled] - df[i0,:y:scaled]) / TDIF;

    #And the offset
    m = df[i0,:y:scaled] - k*df[i0,:t]

    return k, m

end
export linear_growth_init

function logistic_growth_init(df::DataFrame)
    i0 = argmin(df[!,:ds])
    i1 = argmax(df[!,:ds])

    TDIF = df[i1,:t] - df[i0,:t]

    #Force valid values in case y > cap or y < 0
    C0 = df[i0, :cap_scaled]
    C1 = df[i1, :cap_scaled]

    y0 = max(0.01*C0, min(0.99*C0, df[i0, :y_scaled]))
    y1 = max(0.01*C1, min(0.99*C1, df[i1, :y_scaled]))

    r0 = C0 / y0
    r1 = C1 / y1

    if (abs(r0 - r1) <= 0.01)
        r0 = 1.05*r0
    end

    L0 = log(r0 - 1)
    L1 = log(r1 - 1)

    #Initialize the offset
    m = L0*TDIF / (L0 - L1)

    #And the rate
    k = (L0 - L1) / TDIF

    return k, m

end
export logistic_growth_init

function setup_dataframe!(m::ProphetModel, df::DataFrame; initialize_scales::Bool = false)

    #Check numeric
    if "y" in names(df)
        df.y   = convert.(Number, df.y)
        invals = map(x -> any(f -> f(x), (ismissing, isnothing, isnan, isinf)), df.y)
        if any(invals)
            error("Found invalid value: $(df.y[invals]) at row(s) $((1:length(invals))[invals .> 0])")
        end
    else 
        error(":y missing from DataFrame `df`")
    end

    #Build dates
    df.ds = set_date(df.ds)
    if (any(isnothing.(df.ds)))
        error("Unable to parse date format in column `ds` of `df`. Convert to `DateTime` or `TimeZones.ZonedDateTime` "*
              "format %Y-%m-%d or %Y-%m-%dT%H:%M:%S as in '2020-01-01' or '2020-01-01T22:00:12' "*
              "and check that there are no missing values.")
    end

    #Check regressors in DataFrame
    if length(m.extra_regressors) > 0
        regressor_names = keys(m.extra_regressors)

        #Check regressors exist
        if !(all(in.(regressor_names, Ref(propertynames(df)))))
            error("Regressor(s) $(regressor_names[collect(in.(regressor_names, Ref(propertynames(df))))]) "* 
                    "not found in DataFrame df")
        end

        #Transform regressors to numeric
        df[:, collect(regressor_names)] = convert.(Number, df[:, collect(regressor_names)])
        invalid_cols = keep_invalid(df, collect(regressor_names))
        if (nrow(invalid_cols) > 0)
            error("The following rows couldn't be transformed to numeric:\n$invalid_cols")
        end
    end

    if (length(m.seasonalities) > 0)
        for name in keys(m.seasonalities)
            condition_name = m.seasonalities[name]["condition_name"]
            if (!isnothing(condition_name))
                if !(condition_name in names(df))
                    error("Condition $name missing from DataFrame")
                end
                if !(eltype(df[!,condition_name]) == Bool)
                    error("Found non Bool column in $name")
                end
                df[!,condition_name] = Bool.(df[!,condition_name])
            end
        end
    end

    #Arrange dataframe
    sort!(df, :ds)

    #Initialize scaling values (for normalization and time)
    initialize_scales_fn!(m, initialize_scales, df)

    if (m.logistic_floor)
        if !("floor" in names(df))
            error("Expected column 'floor' in DataFrame")
        end
    else
        df[!,:floor] .= 0
    end
        
    if (m.growth == "logistic")
        if !("cap" in names(df))
            error("Capacities column 'cap' must be supplied for logistic growth in DataFrame")
        end

        if (any(df[!,:cap] .<= df[!,:floor]))
            error("cap must be greater than floor (which defaults to 0)")
        end

        df[!,:cap_scaled] = (df[!,:cap] - df[!,:floor]) ./ m.y_scale;

    end

    df[!,:t] = time_diff(df[!,:ds], m.start; units = Dates.Second) ./ m.t_scale

    if ("y" in names(df))
        df[!,:y_scaled] = (df[!,:y] - df[!,:floor]) ./ m.y_scale;
    end

    if (length(m.extra_regressors) > 0)
        for name in keys(m.extra_regressors)
            df[!,name] = (df[!,name] .- m.extra_regressors[name]["μ"]) ./ m.extra_regressors[name]["σ"]
        end
    end

    return df

end
export setup_dataframe!

function set_auto_seasonalities!(m::ProphetModel)
    first_t  = minimum(m.history[!,:ds])
    last_t   = maximum(m.history[!,:ds])
    dt       = diff(time_diff(m.history[!,:ds], m.start; units = Dates.Day))
    min_dt   = minimum(dt[dt .> 0])

    #Disable yearly seasonality if < 2 yrs or data is yearly
    yearly_disable = time_diff(last_t, first_t; units = Dates.Day) < (365*2) || min_dt >= 365
    fourier_order  = parse_seasonality_args(m, "yearly", m.yearly_seasonality, yearly_disable, 10)
    if fourier_order > 0
        m.seasonalities["yearly"] = Dict("period" => 365.25, 
                                         "fourier_order" => fourier_order, 
                                         "prior_scale" => m.seasonality_prior_scale,
                                         "model_mode" => m.seasonality_mode,
                                         "condition_name" => nothing)
    end

    #Disable weekly seasonality if < 2 weeks or data is weekly
    weekly_disable = time_diff(last_t, first_t; units = Dates.Day) < 14 || min_dt >= 7
    fourier_order  = parse_seasonality_args(m, "weekly", m.weekly_seasonality, weekly_disable, 3)
    if fourier_order > 0
        m.seasonalities["weekly"] = Dict("period" => 7, 
                                         "fourier_order" => fourier_order, 
                                         "prior_scale" => m.seasonality_prior_scale,
                                         "model_mode" => m.seasonality_mode,
                                         "condition_name" => nothing)
    end

    #Disable daily seasonality if < 2 days or data is daily
    daily_disable = time_diff(last_t, first_t; units = Dates.Day) < 2 || min_dt >= 1
    fourier_order  = parse_seasonality_args(m, "daily", m.daily_seasonality, daily_disable, 4)
    if fourier_order > 0
        m.seasonalities["weekly"] = Dict("period" => 1, 
                                         "fourier_order" => fourier_order, 
                                         "prior_scale" => m.seasonality_prior_scale,
                                         "model_mode" => m.seasonality_mode,
                                         "condition_name" => nothing)
    end
end
export set_auto_seasonalities!

@doc raw"""
Get number of Fourier components for built in seasonality

"""
function parse_seasonality_args(m::ProphetModel, name::String, arg::Union{String,Integer,Bool},
    auto_disable::Union{Bool}, default_order::Integer)

    if arg == "auto"
        fourier_order = 0
        if (length(m.seasonalities) > 0 && name in keys(m.seasonalities))
            @info "Found custom seasonality named $name. Disabling built-in $name seasonality."
        elseif auto_disable
            @info "Disabling $name seasonality. Run prophet with `$(name)_seasonality = true` to override this"
        else 
            fourier_order = default_order
        end
    elseif typeof(arg) == Bool && arg
        fourier_order = default_order
    elseif typeof(arg) == Bool && !arg
        fourier_order = 0
    else
        fourier_order = arg
    end

    return fourier_order
end
export parse_seasonality_args

function fourier_series(dates::Union{Vector{<: TimeType},Vector{<: Real}}, period::Real,
    series_order::Integer)
    tp       = time_diff(dates, convert(typeof(dates[1]), DateTime(1970,01,01,00,00,00)); units = Dates.Day)
    features = zeros(length(tp), 2*series_order)
    for i in 1:series_order
        x = 2*pi*i*tp / period
        features[:,i*2 - 1] = sin.(x)
        features[:,i*2]     = cos.(x)
    end
    return features
end
export fourier_series

function make_seasonality_features(dates::Union{Vector{<: TimeType},Vector{<: Real}}, period::Real,
    series_order::Integer, prefix::String)
    features   = fourier_series(dates, period, series_order)
    columnames = map(*, repeat([prefix * "_delim_"], size(features,2)), string.(1:size(features,2)))
    return DataFrame(features, columnames)
end
export make_seasonality_features

function counstruct_holiday_dataframe(m::ProphetModel, dates::Union{Vector{<: TimeType},Vector{<: Real}})

    if !isnothing(m.holidays)
        all_holidays = m.holidays
    else
        all_holidays = DataFrame()
    end

    if !isnothing(m.country_holidays)
        if (typeof(dates) <: Union{TimeType,Vector{<:TimeType}})
            year_list = getfield.(unique(Dates.Year.(dates)),:value)
            country_holidays_df = make_holidays_df(year_list, m.country_holidays)
            all_holidays = vcat(all_holidays, country_holidays_df)
        else
            error("Unable to get country holidays for years of type $(typeof(dates))")
        end
    end

    #If the model_model has already been fit with a certain set of holidays
    #make sure we are using those same ones (dates/names)
    if (!isnothing(m.train_holiday_names))
        row_to_keep  = in.(all_holidays[!,:holiday], Ref(m.train_holiday_names))
        all_holidays = all_holidays[row_to_keep,:]
        holidays_to_add = DataFrame(
            holiday = setdiff(m.train_holiday_names, all_holidays[!,:holiday])
        )
        all_holidays = vcat(all_holidays, country_holidays_df)
    end

    return all_holidays

end
export counstruct_holiday_dataframe

function add_seasonality!(m::ProphetModel, name::String, period::Real, fourier_order::Integer;
    prior_scale::Union{Nothing,Real} = nothing, model_mode::Union{Nothing,String} = nothing, condition_name::Union{Nothing,String} = nothing)

    if (!isnothing(m.history))
        error("Seasonality must be added before fitting")
    end

    if (!(name in ["daily","weekly","yearly"]))
        if (Base.isidentifier(name))
            error("You have provided a name that is already in use in Julia: $name")
        end
        validate_column_name(m, name; check_seasonalities = false)
    end

    if (isnothing(prior_scale))
        prior_scale = m.seasonality_prior_scale
    end

    if (prior_scale <= 0)
        error("Prior scale must be > 0. Provided value was: $prior_scale")
    end

    if (fourier_order <= 0)
        error("Fourier order must be > 0. Value provided was $fourier_order")
    end

    if (!(model_mode in ["additive","multiplicative"]))
        error("""`model_mode` must be "additive" or "multiplicative" """)
    end

    if (!isnothing(condition_name))
        validate_column_name(m, condition_name)
    end

    m.seasonalities[name] = Dict("period" => period, 
                                 "fourier_order" => fourier_order,
                                 "prior_scale" => prior_scale,
                                 "model_mode" => model_mode,
                                 "condition_name" => condition_name)

end
export add_seasonality!

function add_regressor!(m::ProphetModel, name::String; 
    prior_scale::Union{Nothing,Real} = nothing, 
    standardize::Union{Bool,String} = "auto",
    model_mode::Union{Nothing,String} = nothing)

    if (!isnothing(m.history))
        error("Regressors must be added prior to model_model fitting")
    end

    if (Base.isidentifier(name))
        error("You have provided a name that is already in use in Julia: $name")
    end

    validate_column_name(m, name; check_regressors = false)

    if (isnothing(prior_scale))
        prior_scale = m.holidays_prior_scale
    end

    if (isnothing(model_mode))
        model_mode = m.seasonality_mode
    end

    if (prior_scale <= 0)
        error("Prior scale must be > 0. Provided value was: $prior_scale")
    end

    if (!(model_mode in ["additive","multiplicative"]))
        error("""`model_mode` must be "additive" or "multiplicative" """)
    end

    m.extra_regressors[name] = Dict("prior_scale" => prior_scale, "standardize" => standardize, "μ" => 0, "σ" => 1.0, "model_mode" => model_mode)

end
export add_regressor!

function add_country_holidays!(m::ProphetModel, country_name::String; force=true)
    if (!isnothing(m.history) && !force)
        error("Country holidays must be added before model_model fitting")
    end

    if !(country_name in generated_holidays()[!,:country])
        error("Holidays in $country_name are not currently supported")
    end

    for name in get_holiday_names(country_name)
        validate_column_name(m, name; check_holidays = false)
    end

    if (!isnothing(m.country_holidays))
        @info "Changing country holidays from $(m.country_holidays) to $(country_name)"
    end

    m.country_holidays = country_name
end
export add_country_holidays!

function make_holiday_features(m::ProphetModel, dates::Union{Vector{<: TimeType},Vector{<:Real}}, holidays::DataFrame)
    dates = set_date(dates) #FIXME: I don't think this is required
    #TODO: rellenar
end
export make_holiday_features

function add_group_component(components::DataFrame, name::String, gp::Vector)
    new_comp = filter(:component => comp -> comp in gp, components)
    gp_cols  = unique(new_comp[!,:col])
    if length(gp_cols) > 0
        new_comp   = DataFrame(col = gp_cols, component = name)
        components = vcat(components, new_comp)
    end
    return components
end
export add_group_component

function regressor_column_matrix(m::ProphetModel, seasonal_features::DataFrame, model_modes::Dict)
    components = DataFrame(
        component = names(seasonal_features),
        col       = 1:ncol(seasonal_features))
    transform!(components,:, :component => ByRow(x -> split.(x,"_delim_")) => [:component,:part])
    select!(components, [:col,:component])

    #Add total for holidays
    if (!isnothing(m.train_holiday_names))
        components = add_group_component(components, "holidays", unique(m.train_holiday_names))
    end

    #Add total for additive and multiplicative components ands regressors
    for model_mode in ["additive","multiplicative"]
        components = add_group_component(components, model_mode * "_terms", model_modes[model_mode])
        regressors_by_model_mode = String[]
        if (length(m.extra_regressors) > 0)
            for name in keys(m.extra_regressors)
                props = m.extra_regressors[name]
                if (props["model_mode"] == model_mode)
                    regressors_by_model_mode = hcat(regressors_by_model_mode, name)
                end
            end
            components  = add_group_component(components, "extra_regressors_" * model_mode, regressors_by_model_mode)
            model_modes[model_mode] = vcat(model_modes[model_mode], model_mode * "_terms")
            model_modes[model_mode] = vcat(model_modes[model_mode], model_mode * "extra_regressors_")
        end
    end
    #After all of the additive/multiplicative groups have been added
    model_modes[m.seasonality_mode] = vcat(model_modes[m.seasonality_mode], ["holidays"])

    #Convert to a binary matrix
    component_cols = select(components, [:component => ByRow(isequal(v)) => Symbol(v) for v in unique(components.component)])
    #TODO: check if additional order needed

    #Add columns for additive and multiplicative terms in missing
    for name in ["additive_terms","multiplicative_terms"]
        if !(name in names(component_cols))
            component_cols[!,name] .= 0
        end
    end

    filter!(:component => component -> component != "zeros", components)

    if maximum(component_cols[!,:additive_terms] + component_cols[!,:multiplicative_terms]) > 1
        error("A bug ocurred in seasonal components.")
    end

    if (!isnothing(m.train_component_cols))
        component_cols = component_cols[!,names(m.train_component_cols)]
        if (!all(component_cols == m.train_component_cols))
            error("A bug ocurred in constructing regressors")
        end
    end

    return component_cols, model_modes
    
end
export regressor_column_matrix

function make_all_seasonality_features!(m::ProphetModel, df::DataFrame)
    prior_scales = []
    seasonal_features = DataFrame()
    model_modes = Dict("additive" => [], "multiplicative" => [])

    #Seasonality features
    for name in keys(m.seasonalities)
        props    = m.seasonalities[name]
        features = make_seasonality_features(df[!,:ds], props["period"], props["fourier_order"], name)
        if !isnothing(props["condition_name"])
            @warn "This has not been checked yet" #TODO: Check 
            features[.!Bool.(df[!,props["condition_name"]]),:] .= 0
        end
        seasonal_features    = hcat(seasonal_features, features)
        prior_scales         = vcat(prior_scales, repeat([props["prior_scale"]], size(features, 2)))
        model_modes[props["model_mode"]] = vcat(model_modes[props["model_mode"]], name)
    end

    #Holiday features
    holidays = counstruct_holiday_dataframe(m, df[!,:ds])
    if (nrow(holidays) > 0)
        holiday_features, prior_scales, holiday_names = make_holiday_features(m, df[!,:ds], holidays)
    end

    #Additional regressors
    for name in keys(m.extra_regressors)
        props                   = m.extra_regressors[name]
        seasonal_features[name] = df[!,name]
        prior_scales            = vcat(prior_scales, props["prior_scale"])
        model_modes[props["model_mode"]]    = vcat(model_modes[props["model_mode"]], name)
    end

    if (ncol(seasonal_features) == 0)
        seasonal_features = DataFrame(zeros = zeros(nrow(df)))
        prior_scales = [1.0]
    end

    component_cols, model_modes = regressor_column_matrix(m, seasonal_features, model_modes)

    return seasonal_features, prior_scales, component_cols, model_modes

end
export make_all_seasonality_features!

function set_changepoints!(m::ProphetModel)
    if (!isnothing(m.changepoints))
        if (length(m.changepoints) > 0)
            m.changepoints = set_date(m.changepoints)
            if (minimum(m.changepoints) < minimum(m.history[!,:ds]) ||
                    maximum(m.changepoints) > maximum(m.history[!,:ds]))
                error("Changepoints must fall within training data")
            end
        end
    else
        #Place potential changepoints evenly through the first
        #changepoint_range proportion of the history
        hist_size = floor(nrow(m.history) * m.changepoint_range)
        if (m.n_changepoints + 1 > hist_size)
            m.n_changepoints = hist_size - 1
            @info "n_changepoints greater than number of observations. Using $(m.n_changepoints)"
        end

        if (m.n_changepoints > 0)
            cp_indexes = Integer.(round.(range(1, 
                stop = hist_size, length = (m.n_changepoints + 1))))[2:end]
            m.changepoints = m.history[cp_indexes,:ds]
        else
            m.changepoints = []
        end
    end
    if (length(m.changepoints) > 0)
        m.changepoints_t = time_diff(m.changepoints, m.start; units = Dates.Second) ./ m.t_scale
        sort!(m.changepoints_t)
    else
        m.changepoints_t = [0] #dummy changepoint
    end
end
export set_changepoints!

function fit_prophet!(m::ProphetModel, df::DataFrame; kwargs...)

    if (!isnothing(m.history))
        error("ProphetModel object can only be fit once. Instantiate a new object.")
    end

    if (!("ds" in names(df)) || !("y" in names(df)))
        error("DataFrame must have columns 'ds' and 'y' with dates and values respectively")
    end

    history = drop_invalid(df, :y)
    if (nrow(history) < 2)
        error("DataFrame has less than 2 valid rows")
    end

    m.history_dates = sort(unique(set_date(df.ds)));
    history = setup_dataframe!(m, history; initialize_scales = true)
    m.history = history

    set_auto_seasonalities!(m)

    seasonal_features, prior_scales, component_cols, model_modes = make_all_seasonality_features!(m, history);
    m.train_component_cols = component_cols
    m.component_modes = model_modes

    set_changepoints!(m)
    

end
export fit_prophet!
