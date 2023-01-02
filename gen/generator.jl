using Clang
using Clang.Generators

using whisper_cpp_jll

d = pwd()
cd(@__DIR__)

options = load_options(joinpath(@__DIR__, "generator.toml"))


include_dir = joinpath(@__DIR__, "include")
headers = joinpath(include_dir, "whisper.h")

args = get_default_args() 
push!(args, "-I$include_dir")

ctx = create_context(headers, args, options)
build!(ctx)

cd(d)