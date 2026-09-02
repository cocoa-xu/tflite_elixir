defmodule TFLiteElixir.ResizeTakesTheShapeTest do
  use ExUnit.Case

  alias TFLiteElixir.{Interpreter, InterpreterBuilder, FlatBufferModel, TFLiteTensor}
  alias TFLiteElixir.Ops.Builtin.BuiltinResolver

  @model "test/test_data/mobilenet_v2_1.0_224_inat_bird_quant.tflite"

  # TFLiteTensor.shape/1 returns a tuple. Both this wrapper and the Erlang
  # function under it repeated an is_list guard by hand, so feeding one back in
  # raised FunctionClauseError, and widening only the Erlang side changed
  # nothing here.
  setup do
    model = FlatBufferModel.build_from_file!(@model)
    {:ok, resolver} = BuiltinResolver.new()
    {:ok, builder} = InterpreterBuilder.new(model, resolver)
    {:ok, interpreter} = Interpreter.new()
    :ok = InterpreterBuilder.build(builder, interpreter)
    :ok = Interpreter.allocate_tensors(interpreter)
    {:ok, [index | _]} = Interpreter.inputs(interpreter)
    %{interpreter: interpreter, index: index}
  end

  test "an interpreter resize takes the shape this library hands out", ctx do
    tensor = Interpreter.tensor(ctx.interpreter, ctx.index)
    shape = TFLiteTensor.shape(tensor)
    dims = TFLiteTensor.dims(tensor)

    assert is_tuple(shape)
    assert dims == Tuple.to_list(shape)

    assert :ok == Interpreter.resize_input_tensor(ctx.interpreter, ctx.index, shape)
    assert :ok == Interpreter.resize_input_tensor(ctx.interpreter, ctx.index, dims)
    assert :ok == Interpreter.resize_input_tensor_strict(ctx.interpreter, ctx.index, shape)
    assert :ok == Interpreter.resize_input_tensor_strict(ctx.interpreter, ctx.index, dims)
  end

  test "a signature runner resize takes it too", ctx do
    tensor = Interpreter.tensor(ctx.interpreter, ctx.index)
    shape = TFLiteTensor.shape(tensor)

    case Interpreter.get_signature_defs(ctx.interpreter) do
      {:ok, defs} when map_size(defs) > 0 ->
        [key | _] = Map.keys(defs)
        {:ok, runner} = Interpreter.get_signature_runner(ctx.interpreter, key)
        {:ok, [name | _]} = TFLiteElixir.SignatureRunner.input_names(runner)

        # the point is the guard, not whether TFLite likes these dimensions
        for f <- [
              &TFLiteElixir.SignatureRunner.resize_input_tensor/3,
              &TFLiteElixir.SignatureRunner.resize_input_tensor_strict/3
            ] do
          refute match?({:error, %FunctionClauseError{}}, safe(fn -> f.(runner, name, shape) end))
        end

      _ ->
        :ok
    end
  end

  defp safe(f) do
    {:ok, f.()}
  rescue
    e -> {:error, e}
  end
end
