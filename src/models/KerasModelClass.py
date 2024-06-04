# -*- coding: utf-8 -*-
from tabnanny import verbose
from src.models.ModelClass import *
from src.Common import print_e, print_g
from src.Callbacks import linear_decay, CustomStopper, EpochTime, Emissions

import os
import numpy as np
import pandas as pd
import tensorflow as tf; 
tf.get_logger().setLevel('INFO')
import matplotlib.pylab as plt
import seaborn as sns


class KerasModelClass(ModelClass):

    def __init__(self, config, dataset):
        ModelClass.__init__(self, config=config, dataset=dataset)

    def __config_session__(self, mixed_precision=False):
        # Selecciona una de las gpu dispobiles
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.CONFIG["session"]["gpu"])
        
        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)

    def __fix_seeds__(self):
        # Fijar las semillas de numpy y TF
        np.random.seed(self.CONFIG["model"]["seed"])
        random.seed(self.CONFIG["model"]["seed"])
        tf.random.set_seed(self.CONFIG["model"]["seed"])

    def get_train_dev_sequences(self, dev):
        """ Si dev es cierto retorna 2 secuencias, una para train y otra dev. Ej: return train, dev
            Si es falso retorna una con train+dev. Ej: return train_dev
        """
        raise NotImplementedError

    def train(self, dev=False, save_model=False, train_cfg={}, callbacks=[]):

        train_cfg = {"verbose": 2, "workers": 6, "class_weight": None, "max_queue_size": 20, "multiprocessing": True}

        if dev:
            train_seq, dev_seq = self.get_train_dev_sequences(dev=dev)
            self.__train_model__(train_sequence=train_seq, dev_sequence=dev_seq, save_model=save_model, train_cfg=train_cfg, callbacks=callbacks)
        else:
            train_dev_seq = self.get_train_dev_sequences(dev=dev)
            self.__train_model__(train_sequence=train_dev_seq, dev_sequence=None, save_model=save_model, train_cfg=train_cfg, callbacks=callbacks)

    def __train_model__(self, train_sequence=None, dev_sequence=None, save_model=False, train_cfg={}, callbacks=[]):

        is_dev = dev_sequence is not None

        # callbacks = []

        # Learning rate decay (lineal o cosine)
        lrs = tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: linear_decay(current_lr=lr,
                                                                                      initial_rl=self.CONFIG["model"]["learning_rate"],
                                                                                      final_lr=self.CONFIG["model"]["final_learning_rate"],
                                                                                      epochs=self.CONFIG["model"]['epochs']))
        callbacks.append(lrs)

        # La carpeta cambia si es train_dev o train final
        final_folder = "dev/" if is_dev else ""

        # El número de epochs es el que se hizo en dev o el que se obtenga utilizando un early stopping
        if not is_dev:
            dev_log_path = self.MODEL_PATH+"dev/log.csv"
            if os.path.exists(dev_log_path):
                dev_log_data = pd.read_csv(dev_log_path)

                if self.CONFIG["model"]["early_st_monitor_mode"] == "min":
                    final_epoch_number = dev_log_data[self.CONFIG["model"]["early_st_monitor"]].argmin()+1
                else:
                    final_epoch_number = dev_log_data[self.CONFIG["model"]["early_st_monitor"]].argmax()+1
                                
                print_g("Best epoch number: %d" % final_epoch_number)


            else:
                print_e("Unknown DEV epoch number...")
                exit()

        # Si es DEV, se hace añade el EarlyStopping (Esta versión custom permite no mirar las x primeras epochs)
        else:
            est = CustomStopper(monitor=self.CONFIG["model"]["early_st_monitor"], start_epoch=self.CONFIG["model"]["early_st_first_epoch"],
                                patience=self.CONFIG["model"]["early_st_patience"], verbose=1, mode=self.CONFIG["model"]["early_st_monitor_mode"])
            callbacks.append(est)

        # CO2 logger
        log = Emissions(self.MODEL_PATH + final_folder)
        callbacks.append(log)

        # Si se quiere almacenar la salida del modelo (pesos/csv)
        if save_model:
            if os.path.exists(self.MODEL_PATH + final_folder + "checkpoint"):
                print_g("Model already trained. Loading weights...")
                self.MODEL.load_weights(self.MODEL_PATH + final_folder + "weights")
                return

            # Crear la carpeta
            os.makedirs(self.MODEL_PATH + final_folder, exist_ok=True)

            # Time logger
            log = EpochTime()
            callbacks.append(log)

            # CSV logger
            log = tf.keras.callbacks.CSVLogger(self.MODEL_PATH + final_folder + "log.csv", separator=',', append=False)
            callbacks.append(log)

            # Solo guardar mirando "stop_monitor" en val cuando hay DEV
            if is_dev:
                mc = tf.keras.callbacks.ModelCheckpoint(self.MODEL_PATH + final_folder + "weights", save_weights_only=True, save_best_only=True,
                                                        # verbose=1, save_freq=(len(train_sequence)//self.CONFIG["model"]['batch_size'])*5,
                                                        monitor=self.CONFIG["model"]["early_st_monitor"], mode=self.CONFIG["model"]["early_st_monitor_mode"])
                callbacks.append(mc)

        else:
            print_g("Not saving the model...")

        # Si es el entrenamiento final, no hay dev
        if not is_dev:
            # Si es un dataset de tensorflow
            if isinstance(train_sequence, tf.data.Dataset):
                hist = self.MODEL.fit(train_sequence.batch(self.CONFIG["model"]['batch_size']).cache().prefetch(tf.data.AUTOTUNE),
                                      epochs=final_epoch_number,
                                      verbose=train_cfg["verbose"],
                                      callbacks=callbacks,
                                      class_weight=train_cfg["class_weight"],
                                      max_queue_size=train_cfg["max_queue_size"])
            else:
                hist = self.MODEL.fit(train_sequence,
                                      steps_per_epoch=train_sequence.__len__(),
                                      epochs=final_epoch_number,
                                      verbose=train_cfg["verbose"],
                                      workers=train_cfg["workers"],
                                      use_multiprocessing=train_cfg["multiprocessing"],
                                      callbacks=callbacks,
                                      class_weight=train_cfg["class_weight"],
                                      max_queue_size=train_cfg["max_queue_size"])

            self.MODEL.save_weights(self.MODEL_PATH + "weights")

        # Si es para gridsearch se añade el dev
        else:
            # Si es un dataset de tensorflow
            if isinstance(train_sequence, tf.data.Dataset):
                dseq = train_sequence.batch(self.CONFIG["model"]['batch_size']).cache().prefetch(tf.data.AUTOTUNE)  # .apply(tf.data.experimental.copy_to_device(f'/gpu:{self.CONFIG["session"]["gpu"]}'))
                hist = self.MODEL.fit(dseq,
                                      epochs=self.CONFIG["model"]['epochs'],
                                      verbose=train_cfg["verbose"],
                                      validation_data=dev_sequence.cache().batch(self.CONFIG["model"]['batch_size']).prefetch(tf.data.AUTOTUNE),
                                      callbacks=callbacks,
                                      class_weight=train_cfg["class_weight"],
                                      max_queue_size=train_cfg["max_queue_size"])
            else:
                hist = self.MODEL.fit(train_sequence,
                                      steps_per_epoch=train_sequence.__len__(),
                                      epochs=self.CONFIG["model"]['epochs'],
                                      verbose=train_cfg["verbose"],
                                      validation_data=dev_sequence,
                                      validation_steps=dev_sequence.__len__(),
                                      workers=train_cfg["workers"],
                                      use_multiprocessing=train_cfg["multiprocessing"],
                                      callbacks=callbacks,
                                      class_weight=train_cfg["class_weight"],
                                      max_queue_size=train_cfg["max_queue_size"])
            
        # Almacenar gráfico con el entrenamiento
        if save_model:
            done_epochs = len(hist.history["loss"])
            fg_sz = (max(8, int((done_epochs*8)/500)), 8)
            plt.figure(figsize=fg_sz)  # HAY QUE MEJORAR ESTO
            hplt = sns.lineplot(x=range(done_epochs), y=hist.history[self.CONFIG["model"]["early_st_monitor"].replace("val_", "")], label=self.CONFIG["model"]["early_st_monitor"].replace("val_", ""))
            if is_dev:
                hplt = sns.lineplot(x=range(done_epochs), y=hist.history[self.CONFIG["model"]["early_st_monitor"]], label=self.CONFIG["model"]["early_st_monitor"])
            # hplt.set_yticks(np.asarray(range(0, 110, 10)) / 100)
            # hplt.set_xticks(range(0, done_epochs, 20))
            # hplt.set_xticklabels(range(0, done_epochs, 20), rotation=45)
            hplt.set_title("Train history")
            hplt.set_xlabel("Epochs")
            hplt.set_ylabel(self.CONFIG["model"]["early_st_monitor"])
            hplt.grid(True)
            if not is_dev:
                plt.savefig(self.MODEL_PATH + final_folder + "history.jpg")
            else:
                plt.savefig(self.MODEL_PATH + final_folder + "history.jpg")
            plt.clf()

        # Cargar el mejor modelo (por defecto está el de la última epoch)
        if save_model:
            model_weights_path = self.MODEL_PATH + final_folder
           
            if os.path.exists(model_weights_path+"checkpoint"):
                print_g("Loading best model...")
                self.MODEL.load_weights(model_weights_path+"weights")
            else:
                print_e("Weights not available, Ifinite loss?")
