using DataFrames
using CSV
using MLJ
using MLJDecisionTreeInterface
using Plots
using FileIO

function get_cleaned_data()
    df = CSV.read("./data/zomato.csv", 
	     DataFrame,
	     normalizenames=true)
    list_names = [ "Country_Code", "City", "Longitude", "Latitude",
	     "Cuisines", "Average_Cost_for_two", "Currency", "Has_Table_booking",
	     "Has_Online_delivery", "Is_delivering_now", "Switch_to_order_menu",
	     "Price_range", "Votes", "Aggregate_rating"]
    df = df[:,list_names]
    return df

end

df = get_cleaned_data()
#Assign Multiclass type


df = coerce(df,
	   #:Restaurant_ID => Multiclass,
	   :Country_Code => OrderedFactor,
	   :City =>  OrderedFactor,
	   :Cuisines =>OrderedFactor,
	   :Currency => OrderedFactor,
	   :Has_Table_booking => OrderedFactor,
	   :Has_Online_delivery => OrderedFactor,
	   :Is_delivering_now => OrderedFactor,
	   )

y, X = unpack(df, ==(:Aggregate_rating); rng=123)

train, test = partition(eachindex(y), 0.7, shuffle=true)   

encoder = ContinuousEncoder()
encMach = machine(encoder, X) |> fit!
X_encoded = MLJ.transform(encMach, X)

rf = @load RandomForestRegressor pkg=DecisionTree

max_depth_range = range( rf(),  :max_depth, lower = 1, upper = 30)
n_trees_range = range(rf(), :n_trees, lower=9, upper=15)

lmTuneModel = TunedModel(model=rf(),
                          resampling = CV(nfolds=6, shuffle=true),
                          tuning = Grid(resolution=25),
                          range = [max_depth_range, n_trees_range],
                          measures=[mae, rms, rsquared]);

dtMachine = machine(lmTuneModel, X_encoded, y)
fit!(dtMachine, rows=train, verbosity=0)
evaluate!(dtMachine, rows = train,
	  verbosity=0)


plot(dtMachine)

y_pred = predict(dtMachine, rows= test)

plot(
	histogram((y_pred- y[test,:]), label = "residuals"),
	scatter((y_pred- y[test,:]), label = "residuals"),

)

dtMachine

MLJ.save("./models/rf_regressor.jls", dtMachine)
