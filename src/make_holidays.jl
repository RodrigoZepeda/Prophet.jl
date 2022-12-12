#Inspired by https://github.com/JuliaStats/RDatasets.jl
@doc raw"""
`generated_holidays()`

Holiday names for countries as given by FB prophet
https://github.com/facebook/prophet/blob/main/R/data-raw/generated_holidays.csv

# Example
```julia-repl
julia> generated_holidays()
64332×4 DataFrame
   Row │ ds          holiday                            country  year  
       │ Date        String                             String   Int32 
───────┼───────────────────────────────────────────────────────────────
     1 │ 1995-01-01  New Year's Day                     AE        1995
     2 │ 1995-03-02  Eid al-Fitr* (*estimated)          AE        1995
     3 │ 1995-03-03  Eid al-Fitr Holiday* (*estimated)  AE        1995
     4 │ 1995-03-04  Eid al-Fitr Holiday* (*estimated)  AE        1995
   ⋮   │     ⋮                       ⋮                     ⋮       ⋮
 64330 │ 2044-12-25  Christmas Day                      ZW        2044
 64331 │ 2044-12-26  Boxing Day                         ZW        2044
 64332 │ 2044-12-27  Christmas Day (Observed)           ZW        2044
                                                     64325 rows omitted
```
"""
function generated_holidays()
    try
        #Read file from facebook's github
        csvfile = get("https://raw.githubusercontent.com/facebook/prophet/main/R/data-raw/generated_holidays.csv")
        return CSV.read(csvfile.body, DataFrame, header=1, types=[Date, String, String, Int32]);
    catch e
        #Read local file
        csvfile = joinpath(@__DIR__, "..", "data", "generated_holidays.csv")
        if isfile(csvfile)
            return CSV.read(csvfile, DataFrame, header=1, types=[Date, String, String, Int32]);
        else
            error("Unable to locate dataset file generated_holidays.csv. In addition " * e)
        end
    end
end

export generated_holidays

@doc raw"""
`get_holiday_names()`

Function that returns the names of the holidays for a given country `country_name`. Current supported country codes: 
    AE, AO, AR, AT, AU, AW, AZ, BD, BE, BI, BR, BW, CA, CH, CL, CN, CO, CW, CZ, DE, DJ, DK, DO, EE, EG, ES, 
    ET, FI, FR, GB, GE, GR, HK, HN, HR, HU, ID, IE, IL, IM, IN, IS, IT, JM, KE, KZ, LS, LT, LU, LV, MA, MK, 
    MW, MX, MY, MZ, NA, NG, NI, NL, NO, NZ, PE, PH, PK, PL, PT, PY, RO, RU, SA, SE, SG, SI, SK, SZ, TH, TN, 
    TR, TU, TW, UK, US, UY, UZ, VE, VN, ZA, ZM, ZW    

Holidays come from FB Prophet: https://github.com/facebook/prophet/blob/main/R/data-raw/generated_holidays.csv

...
# Arguments
- `country_name::Union{String,Vector{String}}`: Name(s) of the country(s) to get holidays from
...

# Example
```julia-repl
julia> get_holiday_names("MX")
15-element Vector{String}:
 "Ano Nuevo [New Year's Day]"
 "Ano Nuevo [New Year's Day] (Observed)"
 "Dia de la Constitucion [Constitution Day]"
 "Natalicio de Benito Juarez [Benito Juarez's birthday]"
 "Dia del Trabajo [Labour Day]"
 "Dia de la Independencia [Independence Day] (Observed)"
 ⋮
 "Transmision del Poder Ejecutivo" ⋯ 19 bytes ⋯ " Federal Government] (Observed)"
 "Transmision del Poder Ejecutivo Federal [Change of Federal Government]"
 "Dia de la Constitucion [Constitution Day] (Observed)"
 "Natalicio de Benito Juarez [Benito Juarez's birthday] (Observed)"
 "Dia de la Revolucion [Revolution Day] (Observed)"
julia> get_holiday_names(["MX","US"])
31-element Vector{String}:
 "Ano Nuevo [New Year's Day]"
 "Ano Nuevo [New Year's Day] (Observed)"
 "Dia de la Constitucion [Constitution Day]"
 "Natalicio de Benito Juarez [Benito Juarez's birthday]"
 "Dia del Trabajo [Labour Day]"
 ⋮
 "Christmas Day"
 "Independence Day (Observed)"
 "Christmas Day (Observed)"
 "Juneteenth National Independence Day (Observed)"
 "Juneteenth National Independence Day" 
```
"""
function get_holiday_names(country_name::Union{String,Vector{String}})
    if (typeof(country_name) <: Vector)
        holidays = filter(:country => x -> x in country_name, generated_holidays())
    else    
        holidays = filter(:country => x -> x == country_name, generated_holidays())
    end
    holidays = unique(holidays[!,:holiday])

    if (length(holidays) == 0)
        @warn "Country $country_name not currently supported."
    end

    return holidays
end

export get_holiday_names

@doc raw"""
`make_holidays_df()`

Function that returns a `DataFrame` for given years and countries.

Current supported country codes: 
AE, AO, AR, AT, AU, AW, AZ, BD, BE, BI, BR, BW, CA, CH, CL, CN, CO, CW, CZ, DE, DJ, DK, DO, EE, EG, ES, 
ET, FI, FR, GB, GE, GR, HK, HN, HR, HU, ID, IE, IL, IM, IN, IS, IT, JM, KE, KZ, LS, LT, LU, LV, MA, MK, 
MW, MX, MY, MZ, NA, NG, NI, NL, NO, NZ, PE, PH, PK, PL, PT, PY, RO, RU, SA, SE, SG, SI, SK, SZ, TH, TN, 
TR, TU, TW, UK, US, UY, UZ, VE, VN, ZA, ZM, ZW  

Holidays come from FB Prophet: https://github.com/facebook/prophet/blob/main/R/data-raw/generated_holidays.csv

...
# Arguments
- `years::Union{Integer, UnitRange{<: Integer}, Vector{<: Integer}}` Range of years to get holidays from. 
- `country_name::String`: Name of the country to get holidays from.
...

*Note* If years are outside the information in `generated_holidays()` will throw a warning.

# Example
```julia-repl
julia> make_holidays_df(1998, "AE")
13×2 DataFrame
 Row │ ds          holiday                           
     │ Date        String                            
─────┼───────────────────────────────────────────────
   1 │ 1998-01-01  New Year's Day
   2 │ 1998-01-29  Eid al-Fitr* (*estimated)
   3 │ 1998-01-30  Eid al-Fitr Holiday* (*estimated)
  ⋮  │     ⋮                       ⋮
  11 │ 1998-11-16  Leilat al-Miraj - The Prophet's …
  12 │ 1998-12-02  National Day
  13 │ 1998-12-03  National Day Holiday
                                       7 rows omitted
julia> make_holidays_df(1998:2010, "BR")
169×2 DataFrame
 Row │ ds          holiday                           
     │ Date        String                            
─────┼───────────────────────────────────────────────
   1 │ 1998-01-01  Ano novo
   2 │ 1998-02-24  Carnaval
   3 │ 1998-02-25  Quarta-feira de cinzas (Inicio d…
  ⋮  │     ⋮                       ⋮
 167 │ 2010-11-02  Finados
 168 │ 2010-11-15  Proclamacao da Republica
 169 │ 2010-12-25  Natal
                                     163 rows omitted
julia> make_holidays_df([1996, 2010], "VE")
30×2 DataFrame
 Row │ ds          holiday                           
     │ Date        String                            
─────┼───────────────────────────────────────────────
   1 │ 1996-01-01  Ano Nuevo [New Year's Day]
   2 │ 1996-02-19  Lunes de Carnaval
   3 │ 1996-02-20  Martes de Carnaval
   ⋮│     ⋮                       ⋮
  28 │ 2010-12-24  Nochebuena
  29 │ 2010-12-25  Dia de Navidad
  30 │ 2010-12-31  Fiesta de Fin de Ano
                                    24 rows omitted
julia> make_holidays_df([1996, 2010], "AA")
┌ Warning: Country AA not currently supported.
└ @ Prophet ~/Prophet/src/make_holidays.jl:177
0×2 DataFrame
 Row │ ds    holiday 
     │ Date  String  
─────┴───────────────    
julia> make_holidays_df(1991, "NG")
┌ Warning: Holidays for NG are only supported from 1995 to 2044
└ @ Prophet ~/Prophet/src/make_holidays.jl:185
0×2 DataFrame
 Row │ ds    holiday 
     │ Date  String  
─────┴───────────────                                
```
"""
function make_holidays_df(years::Union{Integer, UnitRange{<: Integer}, Vector{<: Integer}}, country_name::String)
    country_holidays = filter(:country => x -> x == country_name, generated_holidays())

    if (nrow(country_holidays) == 0)
        @warn "Country $country_name not currently supported."
        return  DataFrame(ds = Date[], holiday = String[])
    end

    max_year = maximum(country_holidays[!,:year])
    min_year = minimum(country_holidays[!,:year])
    
    if (maximum(years) > max_year || minimum(years) < min_year)
        @warn "Holidays for $country_name are only supported from $min_year to $max_year"
    end

    if (typeof(years) == Integer)
        years = [years]
    end

    holidays_df = filter(:year => x -> x in years, country_holidays) 

    return  holidays_df[!,[:ds,:holiday]]
end
export make_holidays_df