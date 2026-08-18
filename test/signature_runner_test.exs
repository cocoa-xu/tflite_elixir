defmodule TFLiteElixir.SignatureRunner.Test do
  use ExUnit.Case

  alias TFLiteElixir.{FlatBufferModel, Interpreter, InterpreterBuilder, SignatureRunner}
  alias TFLiteElixir.Ops.Builtin.BuiltinResolver

  @model Path.join([__DIR__, "test_data", "mobilenet_v2_1.0_224_inat_bird_quant.tflite"])
  @input_path Path.join([__DIR__, "test_data", "parrot.bin"])
  @golden_path Path.join([__DIR__, "test_data", "parrot-expected-out.bin"])

  @in_name "map/TensorArrayStack/TensorArrayGatherV3"
  @out_name "prediction"

  defp interpreter do
    model = FlatBufferModel.build_from_file(@model)
    builder = InterpreterBuilder.new!(model, BuiltinResolver.new!())
    interpreter = Interpreter.new!()
    :ok = InterpreterBuilder.build!(builder, interpreter)
    interpreter
  end

  defp runner, do: Interpreter.get_signature_runner!(interpreter(), nil)

  test "nil asks for the primary subgraph" do
    runner = runner()

    assert "<placeholder signature>" == SignatureRunner.signature_key!(runner)
    assert 1 == SignatureRunner.input_size!(runner)
    assert 1 == SignatureRunner.output_size!(runner)
    assert [@in_name] == SignatureRunner.input_names!(runner)
    assert [@out_name] == SignatureRunner.output_names!(runner)
  end

  test "a signature runs the model and agrees with the interpreter" do
    input = File.read!(@input_path)
    golden = File.read!(@golden_path)
    runner = runner()

    :ok = SignatureRunner.allocate_tensors(runner)
    :ok = SignatureRunner.input_tensor(runner, @in_name, input)
    :ok = SignatureRunner.invoke(runner)

    assert golden == SignatureRunner.output_tensor!(runner, @out_name)
  end

  test "predict/2 goes by name in both directions" do
    input = File.read!(@input_path)
    golden = File.read!(@golden_path)

    assert %{@out_name => ^golden} =
             SignatureRunner.predict!(runner(), %{@in_name => input})
  end

  test "an unknown key or tensor name is an error, not a crash" do
    assert {:error, _} = Interpreter.get_signature_runner(interpreter(), "no_such_signature")

    runner = runner()
    :ok = SignatureRunner.allocate_tensors(runner)
    assert {:error, _} = SignatureRunner.input_tensor(runner, "nope", <<0, 1>>)
    assert {:error, _} = SignatureRunner.output_tensor(runner, "nope")
  end

  test "a runner keeps its interpreter alive" do
    input = File.read!(@input_path)
    golden = File.read!(@golden_path)

    # the interpreter term is unreachable from here on purpose
    runner = (fn -> Interpreter.get_signature_runner!(interpreter(), nil) end).()

    Enum.each(Process.list(), &:erlang.garbage_collect/1)
    :erlang.garbage_collect()
    Process.sleep(500)

    assert %{@out_name => ^golden} = SignatureRunner.predict!(runner, %{@in_name => input})
  end
end
