defmodule TFLiteElixir.Test do
  use ExUnit.Case

  alias TFLiteElixir.Interpreter
  alias TFLiteElixir.TFLiteTensor

  test "print_interpreter_state/1" do
    filename = Path.join([__DIR__, "test_data", "mobilenet_v2_1.0_224_inat_bird_quant.tflite"])
    interpreter = TFLiteElixir.Interpreter.new!(filename)

    assert nil == TFLiteElixir.print_interpreter_state(interpreter)
  end

  test "reset_variable_tensor/1" do
    filename = Path.join([__DIR__, "test_data", "mobilenet_v2_1.0_224_inat_bird_quant.tflite"])
    interpreter = Interpreter.new!(filename)
    t = Interpreter.tensor(interpreter, 0)

    zeros = Nx.broadcast(Nx.tensor(0, type: :u8), {1, 224, 224, 3})
    ones = Nx.broadcast(Nx.tensor(1, type: :u8), {1, 224, 224, 3})

    TFLiteTensor.set_data(t, ones)
    t = Interpreter.tensor(interpreter, 0)
    assert close?(ones, TFLiteTensor.to_nx(t, backend: Nx.BinaryBackend))

    # tensor 0 is this model's input and carries no variable flag, and
    # ResetVariableTensors clears only the tensors that do, so it has to leave
    # this one exactly as it was. The old assertion expected zeros and passed
    # only because Nx.all_close answers with a truthy tensor either way.
    TFLiteElixir.reset_variable_tensor(t)
    t = Interpreter.tensor(interpreter, 0)
    assert close?(ones, TFLiteTensor.to_nx(t, backend: Nx.BinaryBackend))
    refute close?(zeros, TFLiteTensor.to_nx(t, backend: Nx.BinaryBackend))
  end

  with {:module, TFLiteElixir.Coral} <- Code.ensure_compiled(TFLiteElixir.Coral) do
    test "Contains EdgeTpu Custom Op" do
      filename = Path.join([__DIR__, "test_data", "mobilenet_v2_1.0_224_inat_bird_quant.tflite"])
      model = TFLiteElixir.FlatBufferModel.build_from_buffer(File.read!(filename))
      ret = TFLiteElixir.Coral.contains_edge_tpu_custom_op?(model)

      :ok =
        case ret do
          false ->
            :ok

          {:error,
           "Coral support is disabled when compiling this library. Please enable Coral support and recompile this library."} ->
            :ok

          _other ->
            false
        end

      filename =
        Path.join([__DIR__, "test_data", "mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite"])

      model = TFLiteElixir.FlatBufferModel.build_from_buffer(File.read!(filename))
      ret = TFLiteElixir.Coral.contains_edge_tpu_custom_op?(model)

      :ok =
        case ret do
          true ->
            :ok

          {:error,
           "Coral support is disabled when compiling this library. Please enable Coral support and recompile this library."} ->
            :ok

          _other ->
            false
        end
    end
  end

  # Nx.all_close answers with a u8 tensor, and every tensor is truthy, so
  # `assert Nx.all_close(a, b)` held just as firmly when a and b were nothing
  # alike. Compare the number it carries instead.
  defp close?(a, b), do: Nx.to_number(Nx.all_close(a, b)) == 1
end
