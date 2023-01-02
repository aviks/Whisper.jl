using Whisper
using Documenter

DocMeta.setdocmeta!(Whisper, :DocTestSetup, :(using Whisper); recursive=true)

makedocs(;
    modules=[Whisper],
    authors="Avik Sengupta <avik@sengupta.net> and contributors",
    repo="https://github.com/aviks/Whisper.jl/blob/{commit}{path}#{line}",
    sitename="Whisper.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
