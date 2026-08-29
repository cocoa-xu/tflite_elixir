# The LiteRT API is a build option of tflite_beam and is off by default. Rather
# than every LiteRT test failing with "not compiled into this build", they are
# excluded unless the build has it.
litert_exclusions =
  if TFLiteElixir.LiteRT.CompiledModel.available?(), do: [], else: [litert: true]

ExUnit.configure(
  exclude:
    [
      # exclude all tests that require a physical TPU by default
      require_tpu: true
    ] ++ litert_exclusions
)

ExUnit.start()
