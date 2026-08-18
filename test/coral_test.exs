defmodule TFLiteElixir.Coral.Test do
  use ExUnit.Case

  alias TFLiteElixir.{Coral, FlatBufferModel, Interpreter}

  @edgetpu_model Path.join([
                   __DIR__,
                   "test_data",
                   "mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite"
                 ])
  @plain_model Path.join([__DIR__, "test_data", "mobilenet_v2_1.0_224_inat_bird_quant.tflite"])

  test "an edgetpu model is recognised by its custom op" do
    assert true ==
             Coral.contains_edge_tpu_custom_op?(FlatBufferModel.build_from_file(@edgetpu_model))

    assert false ==
             Coral.contains_edge_tpu_custom_op?(FlatBufferModel.build_from_file(@plain_model))
  end

  test "listing devices does not need one to be attached" do
    assert is_list(Coral.edge_tpu_devices())
  end

  @tag :require_tpu
  test "a context can be taken, used and taken again" do
    assert [_ | _] = Coral.edge_tpu_devices()

    context = Coral.get_edge_tpu_context!([])

    interpreter =
      Coral.make_edge_tpu_interpreter!(FlatBufferModel.build_from_file(@edgetpu_model), context)

    :ok = Interpreter.allocate_tensors(interpreter)
    Interpreter.input_tensor!(interpreter, 0, :binary.copy(<<128>>, 224 * 224 * 3))
    :ok = Interpreter.invoke(interpreter)
    assert 965 == byte_size(Interpreter.output_tensor!(interpreter, 0))

    # the device goes back once nothing holds it, so it can be taken again
    Enum.each(Process.list(), &:erlang.garbage_collect/1)
    Process.sleep(300)
    assert is_reference(Coral.get_edge_tpu_context!([]))
  end

  @tag :require_tpu
  test "an interpreter outlives the context term it was built from" do
    interpreter =
      (fn ->
         context = Coral.get_edge_tpu_context!([])

         Coral.make_edge_tpu_interpreter!(
           FlatBufferModel.build_from_file(@edgetpu_model),
           context
         )
       end).()

    Enum.each(Process.list(), &:erlang.garbage_collect/1)
    :erlang.garbage_collect()
    Process.sleep(500)

    :ok = Interpreter.allocate_tensors(interpreter)
    Interpreter.input_tensor!(interpreter, 0, :binary.copy(<<128>>, 224 * 224 * 3))
    :ok = Interpreter.invoke(interpreter)
    assert 965 == byte_size(Interpreter.output_tensor!(interpreter, 0))
  end
end
