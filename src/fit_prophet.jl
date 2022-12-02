#Sets model scaling factors using df
function initialize_scales_fn(m::ProphetModel, initialize_scales::Bool, df::DataFrame)
    if (!initialize_scales)
        return m
    end

    if (m.growth == "logistic") && ("floor" in names(df))
        m.logistic_floor = true
        model_floor = df.floor
    else
        model_floor = zeros(nrow(df))
    end

    m.y_scale = maximum(abs.(df.y - model_floor))
    if (m.y_scale == 0)
        m.y_scale = 1.0
    end

    m.start   = minimum(df.ds)
    m.t_scale = time_diff(maximum(df.ds), m.start; units = Dates.Second)

    

end


function setup_dataframe(m::ProphetModel, df::DataFrame; initialize_scales::Bool = false)

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
    if !isnothing(m.extra_regressors)
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

    #TODO: Add here the seasonalities stuff
    if (!isnothing(m.seasonalities))
        print("add")
    end

    #Arrange dataframe
    sort!(df, :ds)

    m = initialize_scales_fn(m, initialize_scales, df)
        
        
    end



end

function fit_prophet(m::ProphetModel, df::DataFrame; kwargs...)

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


end
