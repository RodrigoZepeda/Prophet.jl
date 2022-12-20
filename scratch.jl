using Pkg
Pkg.build()
Pkg.precompile()
using StatsFuns: logistic
using LinearAlgebra: Diagonal
using DataFrames, Turing, Distributions, Random, AbstractMCMC, Dates, TimeZones, Crayons, Prophet

df = DataFrame(ds = Dates.Date(1999,1,1):Dates.Month(1):Dates.Date(2019,1,1), 
            y = 1:241, regressor1 = (1:241).^2, regressor2 =  vcat(1, repeat([0,1], 120)))
m  = ProphetModel()
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