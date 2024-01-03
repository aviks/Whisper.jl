

function register_datadeps()
    register(DataDep(
        "whisper-ggml-tiny.en",
        "Whisper tiny.en model from HuggingFace (~390 MB)",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
        (sha1, "c78c86eb1a8faa21b369bcd33207cc90d64ae9df"),
        ))

    register(DataDep(
        "whisper-ggml-tiny",
        "Whisper tiny model from HuggingFace (~390 MB)",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
        (sha1, "bd577a113a864445d4c299885e0cb97d4ba92b5f"),
        ))

    register(DataDep(
        "whisper-ggml-base.en",
        "Whisper base.en model from HuggingFace (~500 MB)",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
        (sha1, "137c40403d78fd54d454da0f9bd998f78703390c"),
        ))

    register(DataDep(
        "whisper-ggml-base",
        "Whisper base model from HuggingFace (~500 MB)",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        (sha1, "465707469ff3a37a2b9b8d8f89f2f99de7299dac"),
        ))

    register(DataDep(
        "whisper-ggml-small.en",
        "Whisper small.en model from HuggingFace (~1.0 GB)",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin",
        (sha1, "db8a495a91d927739e50b3fc1cc4c6b8f6c2d022"),
        ))

    register(DataDep(
        "whisper-ggml-small",
        "Whisper small model from HuggingFace (~1.0 GB)",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
        (sha1, "55356645c2b361a969dfd0ef2c5a50d530afd8d5"),
        ))

    register(DataDep(
        "whisper-ggml-medium.en",
        "Whisper medium.en model from HuggingFace (~2.6 GB)",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin",
        (sha1, "8c30f0e44ce9560643ebd10bbe50cd20eafd3723"),
        ))

    register(DataDep(
        "whisper-ggml-medium",
        "Whisper medium model from HuggingFace (~2.6 GB)",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
        (sha1, "fd9727b6e1217c2f614f9b698455c4ffd82463b4"),
        ))

    register(DataDep(
        "whisper-ggml-large-v1",
        "Whisper large-v1 model from HuggingFace (~4.7 GB)",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v1.bin",
        (sha1, "b1caaf735c4cc1429223d5a74f0f4d0b9b59a299"),
        ))

    register(DataDep(
        "whisper-ggml-large.en",
        "Whisper large model from HuggingFace (~4.7 GB)",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large.bin",
        (sha1, "0f4c8e34f21cf1a914c59d8b3ce882345ad349d6"),
        ))

end
