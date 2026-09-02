# TFLiteElixir

TensorFlow Lite Elixir bindings with optional EdgeTPU support.

For pure Erlang bindings, please see [cocoa-xu/tflite_beam](https://github.com/cocoa-xu/tflite_beam).

[![Hex.pm](https://img.shields.io/hexpm/v/tflite_elixir.svg?style=flat&color=blue)](https://hex.pm/packages/tflite_elixir)

## Getting Started
[![Run in Livebook](https://livebook.dev/badge/v1/gray.svg)](https://livebook.dev/run?url=https%3A%2F%2Fgithub.com%2Fcocoa-xu%2Ftflite_elixir%2Fblob%2Fmain%2Fnotebooks%2Fimage_classification.livemd)

A general workflow looks like this,

```elixir
# will download and install precompiled version
Mix.install([
  {:tflite_elixir, "~> 1.0"}
])

# parrot.jpeg and the tflite file can be found in the test/test_data directory
interpreter = TFLiteElixir.Interpreter.new!("/path/to/mobilenet_v2_1.0_224_inat_bird_quant.tflite")
input =
  StbImage.read_file!("/path/to/parrot.jpeg")
  |> StbImage.resize(224, 224)
  |> StbImage.to_nx()

[output_tensor_0] = TFLiteElixir.Interpreter.predict(interpreter, input)
indices_nx = Nx.flatten(output_tensor_0)

# get top k predictions (numerical id of the class)
# classes can be found in this file,
# https://raw.githubusercontent.com/cocoa-xu/tflite_elixir/main/test/test_data/inat_bird_labels.txt
# each line corresponds to a class
# and the first line = id 0
top_k = 5
sorted_indices = Nx.argsort(indices_nx, direction: :desc)
top_k_indices = Nx.take(sorted_indices, Nx.iota({top_k}))
top_k_preds = Nx.to_flat_list(top_k_indices)
```

And there is an experimental `ImageClassification` module that does everything for you. It supports both CPU and TPU, and it will show more information, including scores (confidence) and the class name of the predicted results. It's also more flexible where you can adjust different parameters like `top_k` and `threshold` (for confidence) and etc.

```elixir
iex> alias TFLiteElixir.ImageClassification
iex> {:ok, pid} = ImageClassification.start("/path/to/mobilenet_v2_1.0_224_inat_bird_quant.tflite")
iex> ImageClassification.predict(pid, "/path/to/parrot.jpeg")
%{class_id: 923, score: 0.70703125}
iex> ImageClassification.set_label_from_associated_file(pid, "inat_bird_labels.txt")
:ok
iex> ImageClassification.predict(pid, "/path/to/parrot.jpeg")
%{class_id: 923, label: "Ara macao (Scarlet Macaw)", score: 0.70703125}
iex> ImageClassification.predict(pid, "/path/to/parrot.jpeg", top_k: 3)
[
  %{class_id: 923, label: "Ara macao (Scarlet Macaw)", score: 0.70703125},
  %{
    class_id: 837,
    label: "Platycercus elegans (Crimson Rosella)",
    score: 0.078125
  },
  %{
    class_id: 245,
    label: "Coracias caudatus (Lilac-breasted Roller)",
    score: 0.01953125
  }
]
```

There is an experimental `ObjectDetection` module in the same shape. It also supports both CPU and TPU, and returns one entry per detection with its class id, score, label and bounding box, `[ymin, xmin, ymax, xmax]`, in the coordinates of the image you gave it.

```elixir
iex> alias TFLiteElixir.ObjectDetection
iex> {:ok, pid} = ObjectDetection.start("/path/to/ssd_mobilenet_v2_coco_quant_postprocess.tflite")
iex> ObjectDetection.predict(pid, "/path/to/cat.jpeg")
[%{class_id: 16, label: nil, score: 0.93359375, bbox: [3, -1, 294, 240]}]
iex> ObjectDetection.set_label(pid, "/path/to/coco_labels.txt")
:ok
iex> ObjectDetection.predict(pid, "/path/to/cat.jpeg")
[%{class_id: 16, label: "cat", score: 0.93359375, bbox: [3, -1, 294, 240]}]
iex> ObjectDetection.predict(pid, "/path/to/cat.jpeg", threshold: 0.99)
[]
```

To run it on a Coral device, start it with `use_tpu: true` and an Edge TPU model.


### Signatures

A model exported with signatures names its inputs and outputs, so neither side has to
depend on the order the tensors happen to be listed in.

```elixir
iex> alias TFLiteElixir.{Interpreter, SignatureRunner}
iex> {:ok, runner} = Interpreter.get_signature_runner(interpreter, "serving_default")
iex> SignatureRunner.input_names!(runner)
["input_1"]
iex> {:ok, outputs} = SignatureRunner.predict(runner, %{"input_1" => input_data})
iex> Map.keys(outputs)
["output_1"]
```

Passing `nil` instead of a key asks for the primary subgraph, which also works for
models that declare no signatures at all.

## Nerves Support

### Prebuilt firmware (Experimental)

[![Nerves](https://github-actions.40ants.com/cocoa-xu/tflite_elixir/matrix.svg?only=nerves-build)](https://github.com/cocoa-xu/tflite_elixir/actions)

Prebuilt firmwares are available [here](https://github.com/cocoa-xu/tflite_elixir/releases). Nightly builds can be found [here](https://github.com/cocoa-xu/tflite_elixir/actions/workflows/nerves-build.yml?query=is%3Asuccess).

Select the most recent run and scroll down to the `Artifacts` section, download the firmware file for your board and run

```bash
fwup /path/to/the/downloaded/firmware.fw
```

In the nerves build, `tflite_elixir` is integrated as one of the dependencies of the [nerves_livebook](https://github.com/livebook-dev/nerves_livebook) project. This means that you can use livebook (as well as other pre-pulled libraries) to explore and evaluate the `tflite_elixir` project.

The default password of the livebook is `nerves` (as the time of writing, if it does not work, please check the nerves_livebook project).

### Build from Source

1. If prefer precompiled binaries
```shell
# for example
export MIX_TARGET=rpi4

# There is no need to explicitly set CPU architecture
#   for the precompiled libedgetpu binaries. The arch
#   is automatically detected by the `TARGET_ARCH`,
#   `TARGET_OS` and `TARGET_ABI` environment vars.
#
# However, if you are using your own nerves target
#   you can manually set the correct arch, e.g.,
#   set `aarch64` for rpi4.
#
# Possible values including
# - aarch64
# - armv7l
# - armv6
# - riscv64
# - x86_64
export TFLITE_BEAM_CORAL_LIBEDGETPU_LIBRARIES=aarch64
```

2. If prefer not to use precompiled binaries
```shell
# for example
export MIX_TARGET=rpi4
# then set env var TFLITE_BEAM_PREFER_PRECOMPILED to false
export TFLITE_BEAM_PREFER_PRECOMPILED=false
```

## Demo
### Mix Task Demo
0. List all available Edge TPU
```shell
mix list_edgetpu
```

1. Image classification
```shell
mix help classify_image

# Note: The first inference on Edge TPU is slow because it includes,
# loading the model into Edge TPU memory
mix classify_image \
  --model test/test_data/mobilenet_v2_1.0_224_inat_bird_quant.tflite \
  --input test/test_data/parrot.jpeg \
  --labels test/test_data/inat_bird_labels.txt
```

Output from the mix task
```
----INFERENCE TIME----
Note: The first inference on Edge TPU is slow because it includes, loading the model into Edge TPU memory.
6.7ms
-------RESULTS--------
Ara macao (Scarlet Macaw): 0.70703
```

2. Object detection
```shell
mix help detect_image

# Note: The first inference on Edge TPU is slow because it includes,
# loading the model into Edge TPU memory
mix detect_image \
  --model test/test_data/ssd_mobilenet_v2_coco_quant_postprocess.tflite \
  --input test/test_data/cat.jpeg \
  --labels test/test_data/coco_labels.txt
```

Output from the mix task
```
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
----INFERENCE TIME----
13.2ms
cat
  id   : 16
  score: 0.953
  bbox : [3, -1, 294, 240]
```

test files used here are downloaded from [google-coral/test_data](https://github.com/google-coral/test_data) and [wikipedia](https://commons.wikimedia.org/wiki/File:Cat03.jpg).

### Demo code
Model: [mobilenet_v2_1.0_224_inat_bird_quant.tflite](https://github.com/google-coral/edgetpu/blob/master/test_data/mobilenet_v2_1.0_224_inat_bird_quant.tflite)

Input image:
- [parrot.jpg](https://github.com/google-coral/edgetpu/blob/master/test_data/parrot.jpg)
- Or use pre-converted input [parrot.bin](https://github.com/cocoa-xu/tflite_beam/blob/main/test/test_data/parrot.bin)

Labels: [inat_bird_labels.txt](https://github.com/google-coral/edgetpu/blob/master/test_data/inat_bird_labels.txt)

```elixir
alias Evision, as: Cv
alias TFLiteElixir, as: TFLite

# load labels
labels = File.read!("inat_bird_labels.txt") |> String.split("\n")

# load tflite model
filename = "mobilenet_v2_1.0_224_inat_bird_quant.tflite"
model = TFLite.FlatBufferModel.build_from_file(filename)
resolver = TFLite.Ops.Builtin.BuiltinResolver.new!()
builder = TFLite.InterpreterBuilder.new!(model, resolver)
interpreter = TFLite.Interpreter.new!()
:ok = TFLite.InterpreterBuilder.build!(builder, interpreter)
:ok = TFLite.Interpreter.allocate_tensors(interpreter)

# verify loaded model, feel free to skip
# [0] = TFLite.Interpreter.inputs!(interpreter)
# [171] = TFLite.Interpreter.outputs!(interpreter)
# "map/TensorArrayStack/TensorArrayGatherV3" = TFLite.Interpreter.get_input_name!(interpreter, 0)
# "prediction" = TFLite.Interpreter.get_output_name!(interpreter, 0)
# input_tensor = TFLite.Interpreter.tensor(interpreter, 0)
# [1, 224, 224, 3] = TFLite.TFLiteTensor.dims(input_tensor)
# {:u, 8} = TFLite.TFLiteTensor.type(input_tensor)
# output_tensor = TFLite.Interpreter.tensor(interpreter, 171)
# [1, 965] = TFLite.TFLiteTensor.dims(output_tensor)
# {:u, 8} = TFLite.TFLiteTensor.type(output_tensor)

# parrot.bin - if you don't have :evision
binary = File.read!("parrot.bin")
# parrot.jpg - if you have :evision
# load image, resize it, covert to RGB and to binary
binary =
  Cv.imread("parrot.jpg")
  |> Cv.resize({224, 224})
  |> Cv.cvtColor(Cv.cv_COLOR_BGR2RGB)
  |> Cv.Mat.to_binary(mat)

# set input, run forwarding, get output
TFLite.Interpreter.input_tensor(interpreter, 0, binary)
TFLite.Interpreter.invoke(interpreter)
output_data = TFLite.Interpreter.output_tensor!(interpreter, 0)

# if you have :nx
# get predicted label
output_data
|> Nx.from_binary(:u8)
|> Nx.argmax()
|> Nx.to_scalar()
|> then(&Enum.at(labels, &1))
```

## Delegates

`TFLiteElixir.InterpreterBuilder.build/2` attaches an XNNPACK delegate for you,
unless you have attached one yourself. TfLite would otherwise apply XNNPACK on its
own, invisibly, inside `allocate_tensors/1` -- with a thread count nothing could
reach and no way to decline it. The acceleration is the same; where it happens is
now visible.

```elixir
alias TFLiteElixir.{Delegate, InterpreterBuilder}

{:ok, delegate} = Delegate.xnnpack(num_threads: 4)
:ok = InterpreterBuilder.add_delegate!(builder, delegate)
```

`Delegate.available/0` lists what this build can create: `:xnnpack` on every target
except armv6 and armv7l, and `:external` everywhere.

To hand delegation back to TfLite, ask the resolver for it:

```elixir
resolver = TFLiteElixir.Ops.Builtin.BuiltinResolver.new!(apply_default_delegates: true)
```

Anything implementing TfLite's delegate plugin interface can be loaded at runtime,
which covers a GPU delegate built elsewhere and any vendor delegate:

```elixir
{:ok, delegate} =
  Delegate.external("/opt/lib/libvendor_delegate.so", device: 0, precision: :fp16)
```

A delegate must outlive every interpreter built from the builder it was added to,
so there is no way to detach or free one: the builder and each interpreter hold it
for as long as they need it. An interpreter, and any delegate attached to it,
belongs to one process at a time.

## Coral Support

An Edge TPU is reachable as a delegate too, which puts it on the ordinary builder
path -- composable with `set_num_threads/2` and with anything else attached:

```elixir
{:ok, delegate} = TFLiteElixir.Coral.edge_tpu_delegate()
:ok = TFLiteElixir.InterpreterBuilder.add_delegate!(builder, delegate)
```

`TFLiteElixir.Coral.make_edge_tpu_interpreter/2` still works and is unchanged. It
builds its own interpreter internally, though, so nothing set on a builder reaches
it. Both routes produce identical output; asking for a device that is not attached
is an ordinary `{:error, reason}`.

### Dependencies
For macOS
```shell
# only required if not using precompiled binaries
# for compiling libusb
brew install autoconf automake
```

For some Linux OSes you need to manually execute the following command to update udev rules, otherwise, libedgetpu will fail to initialize Coral devices.

```shell
mix deps.get
bash "3rd_party/cache/${TFLITE_BEAM_CORAL_LIBEDGETPU_RUNTIME}/edgetpu_runtime/install.sh"
```

### Compile-Time Environment Variable
- `TFLITE_BEAM_PREFER_PRECOMPILED`

  Use precompiled binaries when `TFLITE_BEAM_PREFER_PRECOMPILED` is `true`. Otherwise, this library will compile from source.

  Defaults to `true`.

- `TFLITE_BEAM_CORAL_SUPPORT`

  Enable Coral Support.

  Defaults to `true`.

- `TFLITE_BEAM_CORAL_USB_THROTTLE`

  Throttling USB Coral Devices. Please see the official warning here, [google-coral/libedgetpu](https://github.com/google-coral/libedgetpu#warning).

  Defaults to `true`.

  Note that only when `TFLITE_BEAM_CORAL_USB_THROTTLE` is set to `false`, `:tflite_beam` will use the non-throttled libedgetpu libraries.

- `TFLITE_BEAM_CORAL_LIBEDGETPU_LIBRARIES`

  Choose which ones of the libedgetpu libraries to copy to the `priv` directory of the `:tflite_beam` app.

  Default value is `native` - only native libraries will be downloaded and copied. `native` corresponds to the host OS and CPU architecture when compiling this library.

  When set to a specific value, e.g, `darwin_arm64` or `darwin_x86_64`, then the corresponding one will be downloaded and copied. This option is expected to be used for cross-compiling, like with nerves.

  Available values for this option are:

  | Value            | OS/CPU              |
  |------------------|---------------------|
  | `aarch64`        | Linux arm64         |
  | `armv7l`         | Linux armv7         |
  | `armv6`          | Linux armv6         |
  | `k8`             | Linux x86_64        |
  | `x86_64`         | Linux x86_64        |
  | `riscv64`        | Linux riscv64       |
  | `darwin_arm64`   | macOS Apple Silicon |
  | `darwin_x86_64`  | macOS x86_64        |


## Installation

Add `:tflite_elixir` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:tflite_elixir, "1.0.0"}
  ]
end
```

1.0.0 builds the runtime from [LiteRT](https://github.com/google-ai-edge/LiteRT)
rather than from TensorFlow. The 0.3 line is the last one built from TensorFlow
itself:

```elixir
def deps do
  [
    {:tflite_elixir, "~> 0.3"}
  ]
end
```

`~> 0.3` will not reach it, which is deliberate: a two part 0.x requirement means
everything below 1.0.0, so releasing this as 0.4.0 would have moved every
existing user onto a different upstream without their asking. Nothing in the
Elixir API was removed or renamed. The one answer that changed is
`TFLiteElixir.tflite_version/0`, which reports LiteRT's version rather than
TensorFlow's; the two version lines are not comparable, so LiteRT's 2.2.0 is
newer than TensorFlow's 2.21.0 rather than older. That number is what a delegate
plugin has to match.

This release requires `nx ~> 0.11`, up from `~> 0.5`. LiteRT reports two eight
bit float formats, and Nx gained `{:f, 8}` for E5M2 in 0.9.0 and
`{:f8_e4m3fn, 8}` for E4M3FN in 0.11.0. On an older Nx, `TFLiteTensor.to_nx/2`
raises `ArgumentError` on either of those tensors, so the floor states what the
binding can actually hand over rather than leaving it to be discovered.

## What the numbers actually say

Measured on an Apple M4 Max, macOS 15.7.3, with
`mobilenet_v2_1.0_224_inat_bird_quant`, 64 images per run. Your machine will give
different figures; what is worth carrying over is the shape of them.

**Batching buys almost nothing.** `resize_input_tensor/3` to a batch and feeding
several images at once is a real thing you can do, and it barely pays:

| batch | CPU, per image | Edge TPU, per image |
| ----- | -------------- | ------------------- |
| 1     | 7.50 ms        | 10.01 ms            |
| 4     | 7.19 ms        | 9.89 ms             |
| 8     | 7.23 ms        | 9.82 ms             |
| 16    | 7.25 ms        | 9.78 ms             |

Sixteen times the batch is 3% off the per image cost on CPU and 2% on the TPU.
TFLite's CPU kernels already thread inside a single image, so the batch axis is
an outer loop rather than the thing that fills the machine; on the USB TPU the
bytes crossing the wire scale with the batch, so batching them together saves no
transfer. Batch if it suits how your inputs arrive, not to go faster.

**Two of the three detection models here cannot batch at all.**
`ssd_mobilenet_v2_coco_quant_postprocess` fails `allocate_tensors/1` after the
resize. `lite-model_efficientdet_lite4` returns `:ok` and then leaves its output
at `{1, 25, 4}`: the batch never reaches the answer, so four images in gets one
result out and the other three are quietly lost. Detection models with fixed
output postprocessing generally behave this way. If you batch, check the output
shape followed your resize before trusting it.

**The Edge TPU is not automatically the fast path.** On this machine the USB
Coral is 33% slower than the CPU for the same model, because a strong CPU plus
XNNPACK beats a 150KB round trip over USB. On a Raspberry Pi that ordering
reverses and it is not close. Measure on the machine you will deploy to.

Documentation can be found at <https://hexdocs.pm/tflite_elixir>.
