defmodule TFLiteElixir.Interpreter.NewAPIs.Test do
  use ExUnit.Case

  alias TFLiteElixir.{FlatBufferModel, Interpreter, InterpreterBuilder}
  alias TFLiteElixir.Ops.Builtin.BuiltinResolver

  @model Path.join([__DIR__, "test_data", "mobilenet_v2_1.0_224_inat_bird_quant.tflite"])
  @input_path Path.join([__DIR__, "test_data", "parrot.bin"])
  @golden_path Path.join([__DIR__, "test_data", "parrot-expected-out.bin"])

  defp interpreter do
    model = FlatBufferModel.build_from_file(@model)
    builder = InterpreterBuilder.new!(model, BuiltinResolver.new!())
    interpreter = Interpreter.new!()
    :ok = InterpreterBuilder.build!(builder, interpreter)
    interpreter
  end

  defp infer(interpreter) do
    :ok = Interpreter.allocate_tensors(interpreter)
    Interpreter.input_tensor!(interpreter, 0, File.read!(@input_path))
    Interpreter.invoke!(interpreter)
    Interpreter.output_tensor!(interpreter, 0)
  end

  test "resize_input_tensor/3 changes the shape" do
    interpreter = interpreter()
    :ok = Interpreter.allocate_tensors(interpreter)
    assert {1, 224, 224, 3} == Interpreter.tensor(interpreter, 0).shape

    assert :ok == Interpreter.resize_input_tensor(interpreter, 0, [2, 224, 224, 3])
    :ok = Interpreter.allocate_tensors(interpreter)
    assert {2, 224, 224, 3} == Interpreter.tensor(interpreter, 0).shape
  end

  test "resize_input_tensor_strict/3 refuses a dimension the model fixed" do
    interpreter = interpreter()
    :ok = Interpreter.allocate_tensors(interpreter)

    assert {:error, _} = Interpreter.resize_input_tensor_strict(interpreter, 0, [4, 224, 224, 3])
  end

  test "cancelling needs to be enabled first" do
    assert {:error, _} = Interpreter.cancel(interpreter())

    interpreter = interpreter()
    assert :ok == Interpreter.enable_cancellation(interpreter)
    assert :ok == Interpreter.cancel(interpreter)

    # cancelling before an invocation does not spoil the next one
    assert File.read!(@golden_path) == infer(interpreter)
  end

  test "memory can be released and reclaimed between invocations" do
    interpreter = interpreter()
    assert File.read!(@golden_path) == infer(interpreter)

    assert :ok == Interpreter.reset_variable_tensors(interpreter)
    assert :ok == Interpreter.release_non_persistent_memory(interpreter)

    assert File.read!(@golden_path) == infer(interpreter)
  end

  test "fp16 precision can be read back after setting it" do
    interpreter = interpreter()

    assert false == Interpreter.get_allow_fp16_precision_for_fp32!(interpreter)
    assert :ok == Interpreter.set_allow_fp16_precision_for_fp32(interpreter, true)
    assert true == Interpreter.get_allow_fp16_precision_for_fp32!(interpreter)
  end

  test "subgraphs and signature lookups" do
    interpreter = interpreter()

    assert Interpreter.subgraphs_size!(interpreter) >= 1

    # this model declares no signatures, and TFLite answers with an empty map and -1
    assert %{} == Interpreter.signature_inputs!(interpreter, "nope")
    assert %{} == Interpreter.signature_outputs!(interpreter, "nope")
    assert -1 == Interpreter.get_subgraph_index_from_signature!(interpreter, "nope")
  end

  test "verify_and_build_from_buffer/2 checks the buffer first" do
    model = FlatBufferModel.verify_and_build_from_buffer(File.read!(@model))
    assert %FlatBufferModel{initialized: true} = model

    # the field used to be filled with a boolean
    assert is_binary(model.minimum_runtime)

    assert :invalid == FlatBufferModel.verify_and_build_from_buffer(<<0, 1, 2, 3, 4, 5, 6, 7>>)
  end
end
