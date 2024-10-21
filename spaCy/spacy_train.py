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
train(config_path, output_dir, overrides={"paths.train": train_data_path, "paths.dev": dev_data_path})

# Finaliza el seguimiento y guarda el reporte
emissions = tracker.stop()  # Detiene el seguimiento de emisiones y guarda el reporte
print(f"Emisiones de CO2: {emissions} kg")
