using DataFrames
using CSV 
using Plots
#using UnicodePlots

df = CSV.read("./data/zomato.csv",DataFrame, normalizenames=true)


histogram(df."Aggregate_rating" .* .1,  ) #Histogram var of study

line(df."Aggregate_rating")

scatter(df."Latitude", df."Longitude")

sort!(combine(groupby(df, [:"Cuisines"]), nrow => :count), :count, rev=true)

bar(first(sort!(combine(groupby(df, [:"Cuisines"]), nrow => :count), :count, rev=true)."Cuisines",15),
    first(sort!(combine(groupby(df, [:"Cuisines"]), nrow => :count), :count, rev=true)."count",15)
)
names(df)

describe(df) #Describe df

size(df) #get shape


