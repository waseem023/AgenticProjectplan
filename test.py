if platform == "Ollama":
    llm = Ollama(model=model.split(":")[0], base_url="http://localhost:11435")
    embeddings = HuggingFaceEmbeddings(model_name=experiment["embedding"])

    