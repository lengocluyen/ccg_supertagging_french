import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib2tikz import save as tikz_save
import pandas as pd
import pickle
import os
import numpy as np
from keras.utils import plot_model

class Visualisation():
	
    def save_model_to_graphic(self,model, file_name="implemented_mode.png",folder=None):
        save_path = os.path.join(folder, file_name)
        plot_model(model,to_file=save_path)

    def accuracy_history(self,history,folder=None, load_with='crf'):
        if load_with=="non_crf":
            pp_accuracy = PdfPages(os.path.join(folder, 'accuracy.pdf'))
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train set', 'Test set'], loc='lower right')
            plt.savefig(os.path.join(folder, 'accuracy.png'))
            tikz_save(os.path.join(folder, 'accuracy.tikz'), figureheight='\\figureheight', figurewidth='\\figurewidth')
            pp_accuracy.savefig()
            # plt.show()
            plt.close()

            # summarize history for loss
            pp_loss = PdfPages(os.path.join(folder, 'loss.pdf'))
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train set', 'Test set'], loc='upper right')
            plt.savefig(os.path.join(folder, 'loss.png'))
            tikz_save(os.path.join(folder, 'accuracy.tikz'), figureheight='\\figureheight', figurewidth='\\figurewidth')
            pp_loss.savefig()
            # plt.show()
            plt.close()
        else:
            pp_accuracy = PdfPages(os.path.join(folder,'accuracy.pdf'))
            plt.plot(history.history['crf_viterbi_accuracy'])
            plt.plot(history.history['val_crf_viterbi_accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train set', 'Test set'], loc='lower right')
            plt.savefig(os.path.join(folder,'accuracy.png'))
            tikz_save(os.path.join(folder,'accuracy.tikz'),figureheight = '\\figureheight',figurewidth = '\\figurewidth')
            pp_accuracy.savefig()
            #plt.show()
            plt.close()

                # summarize history for loss
            pp_loss = PdfPages(os.path.join(folder,'loss.pdf'))
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train set', 'Test set'], loc='upper right')
            plt.savefig(os.path.join(folder,'loss.png'))
            tikz_save(os.path.join(folder,'accuracy.tikz'),figureheight = '\\figureheight',figurewidth = '\\figurewidth')
            pp_loss.savefig()
                #plt.show()
            plt.close()

    def read_history_file(self, history_file):
        history = []
        with open(history_file,"r"):
            while True:
                try:
                    history.append(pickle.load(history_file))
                except EOFError:
                    break
        return history
