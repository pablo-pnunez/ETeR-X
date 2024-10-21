# Data download and conversion

### Download

* [UD Spanish AnCora v2.8](https://github.com/UniversalDependencies/UD_Spanish-AnCora)
* [WikiNER](https://github.com/dice-group/FOX/tree/master/input/Wikiner)

### Conversion
``` bash
python -m spacy convert -n 10 /path/to/file.<iob|conll> /output/directory
```

# Train model

### Install libraries
``` bash
pip install spacy
pip install spacy[lookups]  # Necesario para el lematizador
```

### Create cfg file

```bash
python -m spacy init config config.cfg --lang es --pipeline tok2vec,morphologizer,parser,senter,ner,attribute_ruler,lemmatizer --optimize efficiency
```

### Train

```python
import spacy
from spacy.cli.train import train
from codecarbon import EmissionsTracker

# Inicia el rastreador de emisiones
tracker = EmissionsTracker()
tracker.start()  # Inicia el seguimiento de emisiones

# Define las rutas a tus archivos de configuración y datos
config_path = "config.cfg"  # Ruta a tu archivo de configuración
output_dir = "output"  # Directorio donde guardar el modelo entrenado
train_data_path = "pp_data/train"  # Ruta a tus datos de entrenamiento
dev_data_path = "pp_data/es_ancora-ud-dev.spacy"  # Ruta a tus datos de validación

# Llama a la función de entrenamiento
train(config_path, output_dir, 
      overrides={"paths.train": train_data_path, "paths.dev": dev_data_path})

# Finaliza el seguimiento y guarda el reporte
emissions = tracker.stop()  # Detiene el seguimiento de emisiones y guarda el reporte
print(f"Emisiones de CO2: {emissions} kg")
```
### Evaluate

```bash
python -m spacy evaluate es_core_news_sm ./pp_data/es_ancora-ud-test.spacy 
```

```bash
python -m spacy evaluate output/model-best ./pp_data/es_ancora-ud-test.spacy 
```

### Test

```python
import spacy
model = "output/model-best" 

trained_nlp = spacy.load(model)
text = "Hola esto es un texto de prueba"
doc = trained_nlp(text)
for token in doc:
    print(" | ".join([str(d) for d in [token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_]]))
    
```