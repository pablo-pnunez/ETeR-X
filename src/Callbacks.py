# -*- coding: utf-8 -*-

from codecarbon import EmissionsTracker
import tensorflow as tf
import numpy as np
import time


class Emissions(tf.keras.callbacks.Callback):

    def __init__(self, output_path):
        super().__init__()
        self.emissions_tracker = EmissionsTracker(log_level="error", output_dir=output_path, output_file="emissions_train.csv", tracking_mode="process")

    def on_epoch_end(self, epoch, logs):
        self.emissions_tracker._project_name = f"Epoch {epoch}"
        logs['CO2(gr)'] = self.emissions_tracker.flush() * 1000 # CO2 emissions in grams

    def on_train_begin(self, logs):
        self.emissions_tracker.start()
        
    def on_train_end(self, logs):
        self.emissions_tracker.stop()
        print( f"{int(self.emissions_tracker.final_emissions_data.duration//60)}'{int(self.emissions_tracker.final_emissions_data.duration%60)}\" of training time" )
        print( f"{self.emissions_tracker.final_emissions_data.emissions*1000:.3f}g of CO2" )
        print( f"{self.emissions_tracker.final_emissions_data.energy_consumed*1000:.3f}Wh of electricity" )


class EpochTime(tf.keras.callbacks.Callback):

    def __init__(self):
        super().__init__()
        self.time = 0

    def on_epoch_end(self, epoch, logs):
        logs['e_time'] = time.time()-self.time

    def on_epoch_begin(self, epoch, logs):
        self.time = time.time()



class CustomStopper(tf.keras.callbacks.EarlyStopping):

    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, start_epoch=100,
                 restore_best_weights=False):  # add argument for starting epoch
        super(CustomStopper, self).__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode, baseline=baseline,
                                            restore_best_weights=restore_best_weights)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            super().on_epoch_end(epoch, logs)


def cosine_decay(current_epoch, max_epochs, learning_rate):
    num_periods = 0.5
    alpha = 0.0
    beta = 0.001
    decay_steps = max_epochs
    global_step = current_epoch
    learning_rate = learning_rate

    ln_decay = (decay_steps - global_step) / decay_steps
    cs_decay = 0.5 * (1 + np.cos(np.pi * 2 * num_periods * global_step / decay_steps))
    decayed = (alpha + ln_decay) * cs_decay + beta
    decayed_learning_rate = learning_rate * decayed

    tf.summary.scalar('learning_rate', decayed_learning_rate)

    if current_epoch % 10 == 0:
        print(decayed_learning_rate)

    return decayed_learning_rate


def linear_decay(current_lr, initial_rl, final_lr, epochs):
    step = (initial_rl - final_lr) / epochs
    decayed_learning_rate = current_lr - step

    tf.summary.scalar('learning_rate', decayed_learning_rate)

    return decayed_learning_rate
