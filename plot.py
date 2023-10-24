import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



class MakePlot():

    def __init__(self, path):
        self.df = self.load_data(path)
        self.n_games = len(self.df['Game'].unique())

    @staticmethod
    def smooth_data(data, window_size):
        kernel = np.ones(window_size) / window_size
        smoothed_data = np.convolve(data, kernel, mode='same')
        return smoothed_data

    def load_data(self, path):
        # NOTE: this should return a DataFrame object with columns 'Game', 'Score', 'Time'
        raise NotImplementedError

    def plot(self, n_cols=4, smoothing_size=5, original=True, figsize=(10, 15), suptitle=None, file='./plot.pdf'):
        fig, axs = plt.subplots(self.n_games // n_cols, n_cols, figsize=figsize, sharex=True)
        if suptitle is not None:
            fig.suptitle(suptitle)

        for i, (game, group) in enumerate(self.df.groupby('Game')):
            ax = axs[i]
            ax.set_title(game)
            smoothed_scores = self.smooth_data(group['Score'].values, window_size=smoothing_size)
            
            ax.plot(group['Time'], smoothed_scores, label='Scores')
            ax.set_xlabel('Time')
            ax.set_ylabel('Scores')
            if original:
                ax.plot(group['Time'], group['Score'], alpha=0.4, label='Original Scores', linestyle='--')

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        fig.savefig(file)
        plt.show()
