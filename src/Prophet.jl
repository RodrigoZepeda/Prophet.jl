module Prophet

    using StatsFuns: logistic
    using LinearAlgebra: Diagonal
    using HTTP: get
    using DataFrames, Turing, Distributions, Random, AbstractMCMC, Dates, TimeZones, Crayons, CSV

    #Prophet type and auxiliary functions
    include("Prophet_type.jl")
    include("prophet_model.jl")
    include("make_holidays.jl")
    include("aux.jl")

    #Prophet model
    #include("fit_prophet.jl")


end
