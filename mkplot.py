import os
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import pickle
import click

seaborn.set_style("whitegrid")

G_colors = ['orange', 'blue', 'black', 'red', 'grey', 'green', 'cyan', 'brown']
G_line_widths = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
G_line_styles = ['-']*10
# G_markers = ['o', 's', 'D', '+', 'x', '>', '<']
G_markers = ['']*10
G_markers_size = [12]*10


def draw(ax, mode, options, nb_expt, figsize=(10,10), adjustment=None, legend=True, legend_loc=None, font_size=22,
         colors=[], line_widths=[], line_styles=[], markers=[], markers_size=[], window_size=10):
    if mode == 'scores':
        ff = 'scores.csv'
        txt = 'Average Scores'
        if legend_loc is None:
            legend_loc = 'lower right'
    else:
        ff = 'steps.csv'
        txt = 'Average Steps'
        if legend_loc is None:
            legend_loc = 'upper right'
    for i, (loc, name) in enumerate(options):
        rewards_list = []
        if nb_expt == -1:  # use the last experiment
            idx = 0
            while os.path.exists(loc[:-1] + str(idx)):
                idx += 1
            idx -= 1
            nb_expt = 1
        else:
            idx = int(loc[-1])
        loc = loc[:-1]
        for j in range(nb_expt):
            data = np.genfromtxt(loc + str(idx + j) + '/' + ff, delimiter=',')
            rewards = data[1:, 1]
            # moving average:
            r_avg = np.zeros_like(rewards).astype(np.float32)
            for k in range(len(rewards)):
                if k < window_size:
                    r_avg[k] = np.mean(rewards[:k])
                else:
                    r_avg[k] = np.mean(rewards[k-window_size: k])
            rewards_list.append(r_avg)
        seaborn.tsplot(np.array(rewards_list), condition=name, legend=legend, color=colors[i], ci=80,
                       linewidth=line_widths[i], ls=line_styles[i], marker=markers[i], markersize=markers_size[i], ax=ax)

    # ax.set_ylabel(txt, fontsize=font_size)
    ax.set_xlabel('Epochs', fontsize=font_size)

    for x_label in ax.xaxis.get_majorticklabels():
        x_label.set_fontsize(font_size)
    for y_label in ax.yaxis.get_majorticklabels():
        y_label.set_fontsize(font_size)
    if legend:
        ax.legend(loc=legend_loc, prop={'size': 20})


def single_suite_plot(root_dir, nb_expt=1, game=None):
    root_dir = os.path.abspath(root_dir)
    f, ax = plt.subplots(1, 1, sharex=True)
    options = [(root_dir, game)]
    adjustment = {}
    size = (10, 7)
    legend = False
    draw(ax, 'scores', options, nb_expt=nb_expt, figsize=size, adjustment=adjustment, legend=legend, colors=['orange'], 
            line_widths=[3], line_styles=['-'], markers=[''], markers_size=[5])
    ax.set_title(game, fontsize=24, y=1.08)
    plt.show()


@click.command()
@click.option('--root_dir', '-d', help="directory path containing all the game results")
@click.option('--game', '-g', default='catch', help="game name")
@click.option('--expt', '-e', type=int, default=1, help="num experiments. -1 for the last one.")
def run(root_dir, expt, game):
    single_suite_plot(root_dir, expt, game)

if __name__ == '__main__':
    run()
