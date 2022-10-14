using DataFrames
using CSV
using MLJ
using DecisionTree

df = CSV.read("./data/zomato.csv", 
	     DataFrame,
	     normalizenames=true)

list_names = [ "Country_Code", "City", "Longitude", "Latitude",
	     "Cuisines", "Average_Cost_for_two", "Currency", "Has_Table_booking",
	     "Has_Online_delivery", "Is_delivering_now", "Switch_to_order_menu",
	     "Price_range", "Votes", "Aggregate_rating"]

df = df[:,list_names]


#Assign Multiclass type
df = coerce(df,
	   #:Restaurant_ID => Multiclass,
	   :Country_Code => Continuous,
	   :City => Multiclass,
	   :Cuisines => Multiclass,
	   :Currency => Multiclass,
	   :Has_Table_booking => Multiclass,
	   :Has_Online_delivery => Multiclass,
	   :Is_delivering_now => Multiclass,
	   )

y, X = unpack(df, ==(:Aggregate_rating); rng=123)

train, test = partition(eachindex(y), 0.7, shuffle=true)

encoder = ContinuousEncoder()
encMach = machine(encoder, X) |> fit!
X_encoded = MLJ.transform(encMach, X)

dt = @load RandomForestRegressor pkg=DecisionTree

dtMachine = machine(dt(), X_encoded, y)
fit!(dtMachine, rows=train, verbosity=0)
evaluate!(dtMachine, rows = train, resampling=cv, measures=[mae, rms, rsquared],
	  verbosity=0)
