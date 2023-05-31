defmodule TFLiteElixir.MobileBert do
  defstruct [:tensors, :interpreter, :vocab_map]
  alias __MODULE__, as: T
  alias TFLiteElixir.FlatBufferModel
  alias TFLiteElixir.Interpreter
  alias TFLiteElixir.TFLiteTensor
  alias TFLiteElixir.Tokenizer.FullTokenizer

  @max_seq_len 384
  @max_ans_len 32
  @output_offset 1
  @predict_answer_count 5

  @spec init(String.t()) :: %T{}
  def init(model_file) do
    model_buffer = File.read!(model_file)
    vocab = FlatBufferModel.get_associated_file(model_buffer, "vocab.txt")
    vocabs = String.split(vocab, "\n")
    vocab_map = Map.new(Enum.with_index(vocabs))
    {:ok, interpreter} = Interpreter.new_from_buffer(model_buffer)

    with {:inputs, {:ok, [input_ids_idx, input_mask_idx, segment_ids_idx]}} <-
           {:inputs, Interpreter.inputs(interpreter)},
         {:outputs, {:ok, [end_logits_idx, start_logits_idx]}} <-
           {:outputs, Interpreter.outputs(interpreter)},
         {:input_tensor, input_ids_tensor = %TFLiteTensor{shape: {1, 384}}} <-
           {:input_tensor, Interpreter.tensor(interpreter, input_ids_idx)},
         {:input_tensor, input_mask_tensor = %TFLiteTensor{shape: {1, 384}}} <-
           {:input_tensor, Interpreter.tensor(interpreter, input_mask_idx)},
         {:input_tensor, segment_ids_tensor = %TFLiteTensor{shape: {1, 384}}} <-
           {:input_tensor, Interpreter.tensor(interpreter, segment_ids_idx)},
         {:output_tensor, end_logits_tensor = %TFLiteTensor{shape: {1, 384}}} <-
           {:output_tensor, Interpreter.tensor(interpreter, end_logits_idx)},
         {:output_tensor, start_logits_tensor = %TFLiteTensor{shape: {1, 384}}} <-
           {:output_tensor, Interpreter.tensor(interpreter, start_logits_idx)} do
      {:ok,
       %T{
         tensors: %{
           :input_ids => input_ids_tensor,
           :input_mask => input_mask_tensor,
           :segment_ids => segment_ids_tensor,
           :end_logits => end_logits_tensor,
           :start_logits => start_logits_tensor
         },
         interpreter: interpreter,
         vocab_map: vocab_map
       }}
    else
      {:inputs, _} ->
        raise RuntimeError, "Unexpected model: Number of input tensors"

      {:input_tensor, _} ->
        raise RuntimeError, "Unexpected model: Expect input tensor shape to be {1, 384}"

      {:outputs, _} ->
        raise RuntimeError, "Unexpected model: Number of Output Tensors"

      {:output_tensor, tensor} ->
        raise RuntimeError,
              "Unexpected model: Expect output tensor (#{tensor.name}) shape to be {1, 384}, got #{inspect(tensor.shape)}"
    end
  end

  def run(self = %T{}, query, content) when is_binary(query) and is_binary(content) do
    {features, content_data} = preprocessing(self.vocab_map, query, content)

    :ok = TFLiteTensor.set_data(self.tensors.input_ids, Nx.tensor(features.input_ids, type: :s32))

    :ok =
      TFLiteTensor.set_data(self.tensors.input_mask, Nx.tensor(features.input_mask, type: :s32))

    :ok =
      TFLiteTensor.set_data(self.tensors.segment_ids, Nx.tensor(features.segment_ids, type: :s32))

    :ok = Interpreter.invoke(self.interpreter)

    end_logits = Nx.squeeze(TFLiteTensor.to_nx(self.tensors.end_logits))
    start_logits = Nx.squeeze(TFLiteTensor.to_nx(self.tensors.start_logits))

    postprocessing(self, start_logits, end_logits, content_data)
  end

  defp preprocessing(vocab_map, query, content) do
    query_tokens = FullTokenizer.tokenize(query, true, vocab_map)
    content_words = String.split(content)

    content_tokens = Enum.map(content_words, &FullTokenizer.tokenize(&1, true, vocab_map))
    content_token_idx_to_word_idx_mapping =
      for {token, i} <- Enum.with_index(content_tokens), reduce: [] do
        acc ->
          [List.duplicate(i, Enum.count(token)) | acc]
      end
      |> Enum.reverse()
      |> List.flatten()

    content_tokens = List.flatten(content_tokens)

    # -3 accounts for [CLS], [SEP] and [SEP].
    max_content_len = @max_seq_len - Enum.count(query_tokens) - 3
    content_tokens = Enum.take(content_tokens, max_content_len)

    # Start of generating the `InputFeatures`.
    tokens = ["[CLS]" | query_tokens]
    segment_ids = List.duplicate(0, Enum.count(query_tokens) + 1)

    tokens = tokens ++ ["[SEP]"] ++ content_tokens
    segment_ids = segment_ids ++ [0] ++ List.duplicate(1, Enum.count(content_tokens))

    tokens_count = Enum.count(query_tokens) + 2

    token_idx_to_word_idx_mapping =
      for {_doc_token, i} <- Enum.with_index(content_tokens), reduce: [] do
        acc ->
          [{i + tokens_count, Enum.at(content_token_idx_to_word_idx_mapping, i)} | acc]
      end
      |> Map.new()

    tokens = tokens ++ ["[SEP]"]
    segment_ids = segment_ids ++ [1]

    {:ok, input_ids} = FullTokenizer.convert_to_id(tokens, vocab_map)
    input_mask = List.duplicate(1, Enum.count(input_ids))

    n_padding = Enum.count(input_ids) - @max_seq_len

    {input_ids, input_mask, segment_ids} =
      if n_padding > 0 do
        padding = List.duplicate(0, n_padding)

        {
          input_ids ++ padding,
          input_mask ++ padding,
          segment_ids ++ padding
        }
      else
        {input_ids, input_mask, segment_ids}
      end

    {
      %{
        :input_ids => input_ids,
        :input_mask => input_mask,
        :segment_ids => segment_ids
      },
      %{
        :content_words => content_words,
        :token_idx_to_word_idx_mapping => token_idx_to_word_idx_mapping,
        :original_content => content
      }
    }
  end

  defp postprocessing(self = %T{}, start_logits, end_logits, content_data) do
    start_indexes = candidate_answer_indexes(start_logits)
    end_indexes = candidate_answer_indexes(end_logits)

    word_range =
      for start <- start_indexes, end_idx <- end_indexes, reduce: [] do
        acc ->
          if start <= end_idx do
            if end_idx - start + 1 < @max_ans_len do
              start_index = content_data.token_idx_to_word_idx_mapping[start + @output_offset]
              end_index = content_data.token_idx_to_word_idx_mapping[end_idx + @output_offset]

              if start_index < end_index do
                [
                  {start_index, end_index,
                   Nx.to_number(Nx.add(start_logits[start], end_logits[end_idx]))}
                  | acc
                ]
              else
                acc
              end
            else
              acc
            end
          else
            acc
          end
      end
      |> Enum.reject(&is_nil/1)
      |> Enum.sort(fn {_, _, a_logit}, {_, _, b_logit} ->
        a_logit > b_logit
      end)
      |> Enum.take(@predict_answer_count)

    answers =
      softmaxed(word_range)
      |> Enum.map(fn {score, {start_idx, end_idx, _}} ->
        {score, excerpt_words(content_data, start_idx, end_idx)}
      end)
      |> Enum.reject(&is_nil/1)

    answers
  end

  defp excerpt_words(content_data, start_idx, end_idx) do
    pattern =
      Enum.slice(content_data.content_words, start_idx..end_idx-1)
      |> Enum.map(&Regex.escape/1)
      |> Enum.join("\\s+")

    with {:ok, reg} <- Regex.compile(pattern),
         exceprt = Regex.run(reg, content_data.original_content),
         [first_match | _] <- exceprt do
      first_match
    else
      _ ->
        nil
    end
  end

  defp softmaxed([]), do: []

  defp softmaxed(word_range) do
    max_logit = elem(Enum.at(word_range, 0), 2)

    numerators =
      Enum.map(word_range, fn {_, _, l} ->
        :math.exp(l - max_logit)
      end)

    sum = Enum.sum(numerators)
    Enum.map(Enum.zip(numerators, word_range), fn {s, word_r} -> {s / sum, word_r} end)
  end

  defp candidate_answer_indexes(logits) do
    Nx.to_flat_list(logits[[0..(@max_seq_len - 1)]])
    |> Enum.with_index()
    |> Enum.sort(fn {a, _}, {b, _} -> a > b end)
    |> Enum.take(@predict_answer_count)
    |> Enum.map(fn {_, offset} -> offset end)
  end
end
