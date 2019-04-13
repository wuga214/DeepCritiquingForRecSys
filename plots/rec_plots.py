import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from utils.io import load_yaml
from ast import literal_eval
sns.axes_style("white")


def show_training_progress(df, hue='model', metric='NDCG', name="epoch_vs_ndcg", save=True):
    fig, ax = plt.subplots(figsize=(6, 3))
    #plt.axhline(y=0.165, color='r', linestyle='-')
    ax = sns.lineplot(x='epoch', y=metric, hue=hue, style=hue, data=df, ci=68)
    ax.set_xlabel("Epoch")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    plt.tight_layout()
    if save:
        fig_path = load_yaml('config/global.yml', key='path')['figs']
        fig.savefig('{0}/{1}.pdf'.format(fig_path, name),
                    bbox_inches="tight", pad_inches=0, format='pdf')
    else:
        plt.show()


def show_critiquing(df, name="falling_rank", x='model', y='Falling Rank', hue='type', save=True):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax = sns.boxplot(x=x, y=y, hue=hue, data=df, palette="Set3", width=0.7)
    ax.set_xlabel("Model")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:])
    plt.tight_layout()
    if save:
        fig_path = load_yaml('config/global.yml', key='path')['figs']
        fig.savefig('{0}/{1}.pdf'.format(fig_path, name),
                    bbox_inches="tight", pad_inches=0, format='pdf')
    else:
        plt.show()