from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# quick BPE tokenizer

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

# 4k smol

trainer = trainers.BpeTrainer(
    vocab_size=1024,
    special_tokens=["<|endoftext|>", "<|padding|>"],
    min_frequency=2
)

# json

files = ["data/synth_2/synth_part_000.txt"]
tokenizer.train(files, trainer)
tokenizer.save("nano_1k.json")

# full

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="nano_1k.json",
    model_max_length=1024  # Match this to your LLM's context length
)

tokenizer.pad_token = "<|padding|>"
tokenizer.eos_token = "<|endoftext|>"
tokenizer.bos_token = "<|endoftext|>"
tokenizer.unk_token = "<|endoftext|>" # Fallback

tokenizer.save_pretrained("nano_1k")

tok = AutoTokenizer.from_pretrained("nano_1k")

# quick check
text = "The tiny alpaca jumped over the moon."
encoded = tok(text, padding="max_length", truncation=True, max_length=20)
print("IDs:", encoded["input_ids"])
print("Tokens:", tok.convert_ids_to_tokens(encoded["input_ids"]))
print("Decoded:", tok.decode(encoded["input_ids"], skip_special_tokens=True))

