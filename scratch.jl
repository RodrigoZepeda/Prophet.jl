using Pkg
Pkg.build()
Pkg.precompile()
using StatsFuns: logistic
using LinearAlgebra: Diagonal
using DataFrames, Turing, Distributions, Random, AbstractMCMC, Dates, TimeZones, Crayons, Prophet

df = DataFrame(ds = Date.(["2020-01-01","2020-02-01","2020-03-01"]), y = 1:3)
m  = ProphetModel()

