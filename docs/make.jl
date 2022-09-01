# make.jl
using Documenter, GaussianProcess

makedocs(sitename = "GaussianProcess.jl",
         modules = [GaussianProcess],
         pages = ["Home" => "index.md",
                  "API" => "api.md"])
deploydocs(
    repo = "github.com/JaimeRZP/GaussianProcess.jl"
)