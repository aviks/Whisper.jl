module Whisper
using DataDeps


# Write your package code here.

include("LibWhisper.jl")
using Whisper.LibWhisper


include("models.jl")


function __init__()
    ENV["DATADEPS_ALWAYS_ACCEPT"]="true"
    register_datadeps()
end

end
