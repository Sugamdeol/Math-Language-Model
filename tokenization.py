from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# Initialize a tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Train the tokenizer on the math articles
trainer = trainers.BpeTrainer(vocab_size=30_000, special_tokens=["<pad>", "<s>", "</s>", "<unk>"])
tokenizer.train(['math_articles.txt'], trainer)

# Save the tokenizer
tokenizer.save('math_tokenizer.json')
