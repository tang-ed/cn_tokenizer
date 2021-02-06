from cn_tokenizer import SelfTokenizer

# tokenizer = SelfTokenizer()
# tokenizer.train(["data/test.txt"], special_tokens=["__null__", "__start__", "__end__", "__unk__", "__newln__"])
# tokenizer.save("vocab")
tokenizer = SelfTokenizer("vocab.json")
inputs = tokenizer.encoder(["本项目又作者个人开发，如有侵权，请给开发者留言，核实后做处理。", "欢迎你的使用。"],padding=True, truncation=True, add_special_tokens=True, max_len=20, beginning_to_end=False)
input_ids = inputs["input_ids"]
print(tokenizer.decoder(input_ids))