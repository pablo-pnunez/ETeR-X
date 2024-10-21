import spacy

model = "output/model-best" 

trained_nlp = spacy.load(model)

text = "Hola esto es un texto de prueba"
doc = trained_nlp(text)

for token in doc:
    print(" | ".join([str(d) for d in [token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_]]))

print("-"*50)

model = "es_core_news_sm"

trained_nlp = spacy.load(model)

text = "Hola esto es un texto de prueba"
doc = trained_nlp(text)

for token in doc:
    print(" | ".join([str(d) for d in [token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_]]))
    
