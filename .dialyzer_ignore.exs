# deferror/1 expands to the same case for every raising variant:
#
#     {:error, message} when is_list(message) -> raise List.to_string(message)
#     {:error, message} when is_binary(message) -> raise message
#     res -> res
#
# Wherever the wrapped function's own spec says its reason is always a binary,
# the charlist branch cannot be reached and neither can the fallthrough. That is
# true of the macro, not of the function, and it is not something a caller can
# act on. Some tflite_beam functions do hand back a charlist reason -- the
# downloader is one -- so the branch is not dead in general and must stay.
[
  {"lib/tflite_elixir/litert/compiled_model.ex", :guard_fail},
  {"lib/tflite_elixir/litert/compiled_model.ex", :pattern_match_cov},
  {"lib/tflite_elixir/litert/compiled_model_server.ex", :guard_fail},
  {"lib/tflite_elixir/litert/compiled_model_server.ex", :pattern_match_cov},
  {"lib/tflite_elixir/litert/compiled_model_isolated.ex", :guard_fail},
  {"lib/tflite_elixir/litert/compiled_model_isolated.ex", :pattern_match_cov},
  {"lib/tflite_elixir/signature_runner.ex", :guard_fail},
  {"lib/tflite_elixir/signature_runner.ex", :pattern_match_cov},
  {"lib/tflite_elixir/ops/builtin/builtin_resolver.ex", :guard_fail},
  {"lib/tflite_elixir/ops/builtin/builtin_resolver.ex", :pattern_match_cov},
  {"lib/tflite_elixir/interpreter.ex", :guard_fail},
  {"lib/tflite_elixir/interpreter.ex", :pattern_match_cov},
  {"lib/tflite_elixir/interpreter_builder.ex", :guard_fail},
  {"lib/tflite_elixir/interpreter_builder.ex", :pattern_match_cov},
  {"lib/tflite_elixir/flatbuffer_model.ex", :guard_fail},
  {"lib/tflite_elixir/delegate.ex", :guard_fail},
  {"lib/tflite_elixir/delegate.ex", :pattern_match_cov},
  {"lib/tflite_elixir/coral.ex", :guard_fail},
  {"lib/tflite_elixir/coral.ex", :pattern_match_cov},
  {"lib/tflite_elixir/tflite_tensor.ex", :pattern_match_cov},
  # and the {:ok, res} branch is the dead one wherever the wrapped function
  # answers a bare :ok
  {"lib/tflite_elixir/interpreter.ex", :pattern_match},
  {"lib/tflite_elixir/interpreter_builder.ex", :pattern_match},
  {"lib/tflite_elixir/litert/compiled_model.ex", :pattern_match},
  {"lib/tflite_elixir/litert/compiled_model_server.ex", :pattern_match},
  {"lib/tflite_elixir/litert/compiled_model_isolated.ex", :pattern_match}
]
