from utils.io import load_yaml

import matplotlib.pyplot as plt
import seaborn as sns
sns.axes_style("white")


def show_training_progress(df, hue='model', metric='NDCG', name="epoch_vs_ndcg", save=True):
    fig, ax = plt.subplots(figsize=(6, 3))
    df = df.sort_values(by=['model'])
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
