using Clang
using Clang.Generators

# Regenerate src/LibWhisper.jl from the whisper.cpp headers.
#
# By default the headers come from whisper_cpp_jll. To regenerate against a local
# whisper.cpp build instead -- e.g. when bumping to a whisper.cpp version whose JLL
# has not been released yet -- point WHISPER_CPP_PREFIX at its install prefix:
#
#     WHISPER_CPP_PREFIX=/path/to/prefix julia --project=gen gen/generator.jl
#
# The prefix must contain include/whisper.h and the ggml headers it pulls in.

include_dir = if haskey(ENV, "WHISPER_CPP_PREFIX")
    joinpath(ENV["WHISPER_CPP_PREFIX"], "include")
else
    using whisper_cpp_jll
    joinpath(whisper_cpp_jll.find_artifact_dir(), "include")
end
isfile(joinpath(include_dir, "whisper.h")) || error("whisper.h not found in $include_dir")

d = pwd()
cd(@__DIR__)

options = load_options(joinpath(@__DIR__, "generator.toml"))

# whisper.h includes ggml.h / ggml-cpu.h (and through them ggml-backend.h /
# ggml-alloc.h), and its public API refers to ggml types such as
# ggml_log_callback and ggml_backend_sched_eval_callback. Those headers are
# wrapped too so the types exist; only whisper_* / WHISPER_* are exported.
headers = [joinpath(include_dir, h) for h in
           ("whisper.h", "ggml.h", "ggml-alloc.h", "ggml-backend.h", "ggml-cpu.h")]
filter!(isfile, headers)

args = get_default_args()
push!(args, "-I$include_dir")

ctx = create_context(headers, args, options)
build!(ctx)

cd(d)
