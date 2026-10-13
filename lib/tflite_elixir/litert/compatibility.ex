defmodule TFLiteElixir.LiteRT.Compatibility do
  @moduledoc false

  @spec supported?(module(), atom(), non_neg_integer()) :: boolean()
  def supported?(module, function, arity) do
    Code.ensure_loaded?(module) and function_exported?(module, function, arity)
  end

  @spec call(module(), atom(), list()) :: term()
  def call(module, function, args) do
    if supported?(module, function, length(args)) do
      apply(module, function, args)
    else
      {:error, "this tflite_beam version does not support #{function}/#{length(args)}"}
    end
  end

  @spec with_backend_options(map(), (map() -> result)) :: result | {:error, String.t()}
        when result: term()
  def with_backend_options(options, fun) do
    if Map.has_key?(options, :backend_options) and
         not supported?(:tflite_beam_litert_compiled_model, :backend_options, 1) do
      {:error, "this tflite_beam version does not support LiteRT backend options"}
    else
      fun.(options)
    end
  end

  @spec metrics(module(), list()) :: term()
  def metrics(module, args) do
    options = Enum.at(args, 2)
    args = if is_list(options), do: List.replace_at(args, 2, Map.new(options)), else: args

    if is_map(Enum.at(args, 2)) and
         not supported?(:tflite_beam_nif, :litert_compiled_model_run_with_metrics, 5) do
      {:error, "this tflite_beam version does not support LiteRT metrics options"}
    else
      apply(module, :run_with_metrics, args)
    end
  end
end
