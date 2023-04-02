# defmodule Example do
#   def vocab do
#     buffer = File.read!("/Users/cocoa/Downloads/lite-model_mobilebert_1_metadata_1.tflite")
#     vocab = TFLiteElixir.FlatBufferModel.get_associated_file(buffer, "vocab.txt")
#     vocabs = String.split(vocab, "\n")
#     Map.new(Enum.with_index(vocabs))
#   end
#   def run(text \\ "unaffable") do
#     vocab_map = vocab()
#     tokens = TFLiteBeam.Tokenizer.FullTokenizer.tokenize(text, true, vocab_map)
#     TFLiteBeam.Tokenizer.FullTokenizer.convert_to_id(tokens, vocab_map)
#   end
# end
