import os
import json
from tkinter.messagebox import NO
import pandas as pd
import numpy as np
from src.Common import print_e

def format_range(wks, fr, fc, lr, lc, fmt):

    for row in range(fr, lr+1):
        for col in range(fc, lc+1):

            try:
                value = wks.table[row][col].number
            except Exception:
                value = None

            wks.write(row, col, value, fmt)      


# model = "ATT2ITM"
models = ["MOSTPOP2ITM", "ATT2ITM", "ATT2ITM_2", "BOW2ITM", "USEM2ITM", "BERT2ITM"]

sets = {"restaurants": ["gijon", "barcelona", "madrid", "paris", "newyorkcity", "london"],
        "pois": ["barcelona", "madrid", "paris", "newyorkcity", "london"],
        "amazon": ["fashion", "digital_music"]}

# models = ["BERT2ITM"]

        
dev = True
best_models = []

# Para cada modelo/conjunto/ciudad, mirar si se acabó la ejecución
for model in models:
    for dataset in sets.keys():

        writer = pd.ExcelWriter(f"models/{model}_{dataset}_GS.xlsx", engine="xlsxwriter")

        for subset in sets[dataset]:

            if "RSTVAL" in model:
                columns = {"val_loss": "min", "val_valor_model_mean_absolute_error": "min", "val_output_rst_top_5": "max", "val_output_rst_top_10": "max"}  # adaptar al nuevo modo
                raise NotImplemented
            elif "RST" in model:
                columns = {"val_accuracy": "max", "val_top_5": "max", "val_top_10": "max"}  # adaptar al nuevo modo
                raise NotImplemented
            elif "VAL" in model:
                columns = {"val_mean_absolute_error": "min"}  # adaptar al nuevo modo
                raise NotImplemented
            elif "ITM" in model:
                # columns = {"val_r1": {"best": "max", "others": ["val_loss", "epoch", "model_md5"]}}  # ["val_5", "val_r10", "epoch", "model_md5"]}}
                if "BERT" in model:
                    columns = {"val_NDCG@10": {"best": "max", "format": "pct"}, "val_loss": {"format": "pct"}, "epoch": {"format": "pct"}, "model_md5": {"format": "pct"}}  # ["val_5", "val_r10", "epoch", "model_md5"]}}
                elif "MOSTPOP" in model:
                    columns = {"val_NDCG@10": {"best": "max", "format": "pct"}, "val_loss": {"format": "pct"}, "epoch": {"format": "pct"}, "model_md5": {"format": "pct"}}
                else:
                    columns = {"val_r1": {"best": "max", "format": "pct"}, "val_loss": {"format": "pct"}, "epoch": {"format": "pct"}, "model_md5": {"format": "pct"}}  # ["val_5", "val_r10", "epoch", "model_md5"]}}

            path = f"models/{model}/{dataset}/{subset}/"

            if not os.path.exists(path): continue

            # Obtener nombres de columnas y como obtener el mejor valors
            column_name = list(columns.keys())[0]
            column_best = [columns[c]["best"] for c in columns.keys() if "best" in columns[c].keys()][0]
            column_others = [c for c in columns.keys() if "best" not in columns[c].keys()]
            column_best_name = column_best + "_" + column_name

            ret = []
            # Para cada fichero de resultados, obtener el mejor valor utilizando los parámetros establecidos en "columns"
            for f in os.listdir(path):
                config_file = path+f+"/cfg.json"
                log_file = path+f+("/dev/" if dev else "")+"log.csv"

                try: 
                    log_data = pd.read_csv(log_file)
                    log_data["epoch"] = log_data["epoch"]+1  # Sumar siempre 1 a las epochs
                except Exception:
                    print_e(f"No se ha encontrado fichero de log en '{log_file}'")
                    continue

                with open(config_file) as json_file:
                    config_data = json.load(json_file)

                res = {**config_data["model"], **config_data["dataset_config"]}

                method = (np.min, np.argmin) if column_best == "min" else (np.max, np.argmax)
                best_epoch_data = log_data.iloc[method[1](log_data[column_name])]
                res[column_best_name] = best_epoch_data[column_name]

                for othc in column_others: 
                    if othc == "model_md5": res["model_md5"] = f
                    else: res[othc] = best_epoch_data[othc]

                ret.append(list(res.values()))

            ret = pd.DataFrame(ret, columns=list(res.keys()))
            if len(ret)>1: ret = ret.loc[:, ret.apply(pd.Series.nunique) != 1]  # Eliminar columnas que no varían.

            # Una vez tenemos todos los mejores resultados guardadoes en "ret", podemos generar informes
            # Si hay varios modelos, separar los resultados
            index_pivot=["learning_rate"]

            if "model_version" in ret.columns:
                index_pivot = ["model_version"]+index_pivot
                ret["model_version"] = ret["model_version"].astype(int)

            # res_table = ret.pivot_table(index=index_pivot, columns=["batch_size"], values=column_best_name)
            res_table = ret.pivot_table(index=index_pivot, columns=["batch_size"], values=[column_best_name]+column_others, aggfunc=lambda x: x)[[column_best_name]+column_others]

            # epoch_table = ret.pivot_table(index=index_pivot, columns=["batch_size"], values=column_best_name_epoch)
            # md5_table = ret.pivot_table(index=index_pivot, columns=["batch_size"], values="model_md5", aggfunc=lambda x: x)

            n_cols = len(res_table.reset_index().columns)
            n_cols_data = len(res_table.columns)
            n_cols_dif = n_cols - n_cols_data
            n_rows = len(res_table)

            mrg = 1  # margin

            # Cuál es el mejor de todos?
            if column_best == "min": best_result = ret.sort_values(column_best_name).iloc[0]
            else: best_result = ret.sort_values(column_best_name, ascending=False).iloc[0]

            if "model_version" not in best_result.keys(): best_result["model_version"] = 0
            line_best_result = [dataset, subset, model, best_result["model_version"], best_result["learning_rate"], best_result["batch_size"], best_result[column_best_name]]
            for othc in column_others: line_best_result.append(best_result[othc])
            best_models.append(line_best_result)
            print("\t".join(map(str, line_best_result)))

            res_table.to_excel(writer, sheet_name=subset, startrow=1+mrg, startcol=mrg)
            # epoch_table.to_excel(writer, sheet_name=subset, startrow=2*mrg+3+n_rows, startcol=mrg)
            # md5_table.to_excel(writer, sheet_name=subset, startrow=1+mrg, startcol=2*mrg + n_cols)

            # FORMATEAR EL FICHERO EXCEL ###################################################################################################

            perc_format = writer.book.add_format({'num_format': '0.00%', 'align': 'center', 'valign': 'vcenter'})
            center_format = writer.book.add_format({'align': 'center', 'valign': 'vcenter'})
            border_format = writer.book.add_format({'border': 1})

            worksheet = writer.sheets[subset]

            # Títulos
            title_format = writer.book.add_format({'bold': 1, 'border': 1, 'align': 'center', 'valign': 'vcenter', 'fg_color': 'gray'})
            worksheet.merge_range(mrg, mrg, mrg, mrg+n_cols-1, column_best_name.title(), title_format)
            worksheet.set_column(mrg, mrg+n_cols-1, 15, center_format)

            # Aplicar bordes a todas las filas de la tabla
            worksheet.conditional_format(mrg, mrg, mrg+n_rows+3, mrg+n_cols-1, {'type': 'formula', 'criteria': 'TRUE', 'format': border_format})

        best_models_columns = ["dataset", "subset", "model", "model_version", "learning_rate", "batch_size", column_best_name]
        best_models_columns.extend(column_others)
        pd.DataFrame(best_models, columns=best_models_columns).to_csv("models/best_models.csv")

        writer.save()
