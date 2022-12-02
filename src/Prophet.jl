module Prophet

    using StatsFuns: logistic
    using LinearAlgebra: Diagonal
    using DataFrames, Turing, Distributions, Random, AbstractMCMC, Dates, TimeZones, Crayons

    #Prophet type and auxiliary functions
    include("Prophet_type.jl")
    include("prophet_model.jl")
    include("aux.jl")

    #Prophet model
    #include("fit_prophet.jl")


end
