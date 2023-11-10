
import matplotlib.pyplot as plt
import spectrum_utils.plot as sup
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import textwrap


class Plotting:

    @staticmethod
    def plot_spectrum(spectrum):
        # Plot the spectrum.
        fig, ax = plt.subplots(figsize=(12, 6))
        sup.spectrum(spectrum, grid=False, ax=ax)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        #plt.savefig("quickstart.png", bbox_inches="tight", dpi=300, transparent=True)

    @staticmethod
    def plot_molecule_pair_spectrum(molecule_pair, verbose=True):
        plt.plot(molecule_pair.vector_0, label = 'spectrum 0', alpha=0.5)
        plt.plot(molecule_pair.vector_1, label= 'spectrum 1', alpha=0.5)
        plt.xlabel('m/z')
        plt.ylabel('intensity')
        plt.legend()
        #plt.title(f'molecule_0: {high_molecule_pairs[index].smiles_0}, molecule_1 = {high_molecule_pairs[index].smiles_1}, similarity: {high_molecule_pairs[index].similarity}')
        plt.grid()

        if verbose:
            print(f'molecule_0: {molecule_pair.smiles_0}, molecule_1 = {molecule_pair.smiles_1}, similarity: {molecule_pair.similarity}')

    
    @staticmethod
    def plot_similarity_graphs(similarities, similarities_tanimoto):
        mosaic = """
        11111.
        222223
        222223
        222223
        222223
        222223
        """

        bins = 100
        tick_locators = mticker.FixedLocator(np.arange(0, bins + 1, bins / 4))
        tick_labels = np.asarray([f"{a:.2f}" for a in np.arange(0, 1.01, 0.25)])

    

        with sns.plotting_context("paper", font_scale=1.0):
            fig = plt.figure(constrained_layout=True, figsize=(7.2 * 2, 7.2 / 1.618 * 3))
            gs = GridSpec(3, 4, figure=fig)
            
            # Top panel: Compare different similarities.
            axes_left = fig.add_subfigure(gs[0, 0]).subplot_mosaic(mosaic)
            axes_middle = fig.add_subfigure(gs[0, 1]).subplot_mosaic(mosaic)
            axes_right = fig.add_subfigure(gs[0, 2]).subplot_mosaic(mosaic)
            axes_extra = fig.add_subfigure(gs[0, 3]).subplot_mosaic(mosaic)
            cbar_ax = fig.add_axes([-0.04, 0.75, 0.02, 0.15])
            
            labels = np.asarray([
                ["cosine", "modified_cosine"],
                ["neutral_loss", "cosine"],
                ["neutral_loss", "modified_cosine"],
                ["model_score", "modified_cosine"]
            ])

            for i, (axes, (xlabel, ylabel)) in enumerate(
                zip([axes_left, axes_middle, axes_right, axes_extra], labels)
            ):
                # Plot heatmaps.
                hist, _, _ = np.histogram2d(
                    similarities[xlabel],
                    similarities[ylabel],
                    bins=bins,
                    range=[[0, 1], [0, 1]],
                )
                hist /= len(similarities)
                

                heatmap = sns.heatmap(
                    np.rot90(hist),
                    vmin=np.min(hist),
                    vmax=np.max(hist),
                    cmap="viridis",
                    cbar=i == 2,
                    cbar_kws={"format": mticker.StrMethodFormatter("{x:.3%}")},
                    cbar_ax=cbar_ax if i == 2 else None,
                    square=True,
                    xticklabels=False,
                    yticklabels=False,
                    ax=axes["2"],
                    norm=LogNorm(vmax=np.max(hist)),
                )
                axes["2"].yaxis.set_major_locator(tick_locators)
                axes["2"].set_yticklabels(tick_labels[::-1])
                axes["2"].xaxis.set_major_locator(tick_locators)
                axes["2"].set_xticklabels(tick_labels)
                for _, spine in heatmap.spines.items():
                    spine.set_visible(True)
                axes["2"].set_xlabel(xlabel.replace("_", " ").capitalize())
                axes["2"].set_ylabel(ylabel.replace("_", " ").capitalize())

                axes["2"].plot(
                    [0, bins], [bins, 0], color="black", linestyle="dashed"
                )

                sns.despine(ax=axes["2"])

                # Plot density plots.
                sns.kdeplot(
                    data=similarities,
                    x=xlabel,
                    clip=(0, 1),
                    legend=True,
                    color="black",
                    fill=True,
                    ax=axes["1"],
                )
                axes["1"].set_xlim(0, 1)
                axes["1"].xaxis.set_ticklabels([])
                axes["1"].yaxis.set_major_locator(tick_locators)
                axes["1"].set_yticks([])
                sns.despine(ax=axes["1"], left=True)
                sns.kdeplot(
                    data=similarities,
                    y=ylabel,
                    clip=(0, 1),
                    legend=True,
                    color="black",
                    fill=True,
                    ax=axes["3"],
                )
                axes["3"].set_ylim(0, 1)
                axes["3"].yaxis.set_ticklabels([])
                axes["3"].xaxis.set_major_locator(tick_locators)
                axes["3"].set_xticks([])
                sns.despine(ax=axes["3"], bottom=True)
                for ax in [axes[c] for c in "13"]:
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    
            cbar_ax.set_ylabel("Proportion of pairs")
            cbar_ax.yaxis.set_label_position("left")
            cbar_ax.spines["outline"].set(visible=True, lw=.8, edgecolor="black")
            
            # Middle panel: Compare similarities vs explained intensity.
            axes_left = fig.add_subfigure(gs[1, 0]).subplot_mosaic(mosaic)
            axes_middle = fig.add_subfigure(gs[1, 1]).subplot_mosaic(mosaic)
            axes_right = fig.add_subfigure(gs[1, 2]).subplot_mosaic(mosaic)
            axes_extra = fig.add_subfigure(gs[1, 3]).subplot_mosaic(mosaic)
            cbar_ax = fig.add_axes([-0.04, 0.45, 0.02, 0.15])
            
            labels = np.asarray([
                ["cosine_explained", "cosine"],
                ["neutral_loss_explained", "neutral_loss"],
                ["modified_cosine_explained", "modified_cosine"],
                ["model_score_explained", "model_score"]
            ])

            for i, (axes, (xlabel, ylabel)) in enumerate(
                zip([axes_left, axes_middle, axes_right, axes_extra], labels)
            ):
                # Plot heatmaps.
                hist, _, _ = np.histogram2d(
                    similarities[xlabel],
                    similarities[ylabel],
                    bins=bins,
                    range=[[0, 1], [0, 1]],
                )
                hist /= len(similarities)
                heatmap = sns.heatmap(
                    np.rot90(hist),
                vmin=np.min(hist),
                    vmax=np.max(hist),
                    cmap="viridis",
                    cbar=i == 2,
                    cbar_kws={"format": mticker.StrMethodFormatter("{x:.3%}")},
                    cbar_ax=cbar_ax if i == 2 else None,
                    square=True,
                    xticklabels=False,
                    yticklabels=False,
                    ax=axes["2"],
                    norm=LogNorm(vmax=np.max(hist)),
                )
                axes["2"].yaxis.set_major_locator(tick_locators)
                axes["2"].set_yticklabels(tick_labels[::-1])
                axes["2"].xaxis.set_major_locator(tick_locators)
                axes["2"].set_xticklabels(tick_labels)
                axes["2"].xaxis.set_major_formatter(mticker.PercentFormatter())
                for _, spine in heatmap.spines.items():
                    spine.set_visible(True)
                axes["2"].set_xlabel("Explained intensity")
                axes["2"].set_ylabel(ylabel.replace("_", " ").capitalize())

                sns.despine(ax=axes["2"])

                # Plot density plots.
                sns.kdeplot(
                    data=similarities,
                    x=xlabel,
                    clip=(0, 1),
                    legend=True,
                    color="black",
                    fill=True,
                    ax=axes["1"],
                )
                axes["1"].set_xlim(0, 1)
                axes["1"].xaxis.set_ticklabels([])
                axes["1"].yaxis.set_major_locator(tick_locators)
                axes["1"].set_yticks([])
                sns.despine(ax=axes["1"], left=True)
                sns.kdeplot(
                    data=similarities,
                    y=ylabel,
                    clip=(0, 1),
                    legend=True,
                    color="black",
                    fill=True,
                    ax=axes["3"],
                )
                axes["3"].set_ylim(0, 1)
                axes["3"].yaxis.set_ticklabels([])
                axes["3"].xaxis.set_major_locator(tick_locators)
                axes["3"].set_xticks([])
                sns.despine(ax=axes["3"], bottom=True)
                for ax in [axes[c] for c in "13"]:
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    
            cbar_ax.set_ylabel("Proportion of pairs")
            cbar_ax.yaxis.set_label_position("left")
            cbar_ax.spines["outline"].set(visible=True, lw=.8, edgecolor="black")
            
            # Bottom panel: Evaluate similarities in terms of the Tanimoto index.
            ax = fig.add_subplot(gs[2, :])
            
            sns.violinplot(
                data=similarities_tanimoto,
                x="tanimoto_interval",
                y="value",
                hue="variable",
                hue_order=["cosine", "neutral_loss", "modified_cosine", "model_score"],
                cut=0,
                scale="width",
                scale_hue=False,
                ax=ax,
            )
            ax.set_xlabel("Tanimoto index")
            ax.set_ylabel("Spectrum similarity")
            for label in ax.legend().get_texts():
                label.set_text(label.get_text().replace("_", " ").capitalize())
            sns.move_legend(
                ax,
                "lower center",
                bbox_to_anchor=(.5, 1),
                ncol=3,
                title=None,
                frameon=False,
            )

            sns.despine(ax=ax)
            
            # Subplot labels.
            for y, label in zip([1, 2/3, 0.35], "abc"):
                fig.text(
                    -0.05, y, label, fontdict=dict(fontsize="xx-large", weight="bold")
                )

            # Save figure.
            #plt.savefig("gnps_libraries.png", dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()

    @staticmethod
    def plot_similarity_graphs_deprecated(similarities, similarities_tanimoto):
        mosaic = """
        11111.
        222223
        222223
        222223
        222223
        222223
        """

        bins = 100
        tick_locators = mticker.FixedLocator(np.arange(0, bins + 1, bins / 4))
        tick_labels = np.asarray([f"{a:.2f}" for a in np.arange(0, 1.01, 0.25)])

    

        with sns.plotting_context("paper", font_scale=1.6):
            fig = plt.figure(constrained_layout=True, figsize=(7.2 * 2, 7.2 / 1.618 * 3))
            gs = GridSpec(3, 3, figure=fig)
            
            # Top panel: Compare different similarities.
            axes_left = fig.add_subfigure(gs[0, 0]).subplot_mosaic(mosaic)
            axes_middle = fig.add_subfigure(gs[0, 1]).subplot_mosaic(mosaic)
            axes_right = fig.add_subfigure(gs[0, 2]).subplot_mosaic(mosaic)
            cbar_ax = fig.add_axes([-0.04, 0.75, 0.02, 0.15])
            
            labels = np.asarray([
                ["cosine", "modified_cosine"],
                ["neutral_loss", "cosine"],
                ["neutral_loss", "modified_cosine"]
            ])

            for i, (axes, (xlabel, ylabel)) in enumerate(
                zip([axes_left, axes_middle, axes_right], labels)
            ):
                # Plot heatmaps.
                hist, _, _ = np.histogram2d(
                    similarities[xlabel],
                    similarities[ylabel],
                    bins=bins,
                    range=[[0, 1], [0, 1]],
                )
                hist /= len(similarities)
                

                heatmap = sns.heatmap(
                    np.rot90(hist),
                    vmin=np.min(hist),
                    vmax=np.max(hist),
                    cmap="viridis",
                    cbar=i == 2,
                    cbar_kws={"format": mticker.StrMethodFormatter("{x:.3%}")},
                    cbar_ax=cbar_ax if i == 2 else None,
                    square=True,
                    xticklabels=False,
                    yticklabels=False,
                    ax=axes["2"],
                    norm=LogNorm(vmax=np.max(hist)),
                )
                axes["2"].yaxis.set_major_locator(tick_locators)
                axes["2"].set_yticklabels(tick_labels[::-1])
                axes["2"].xaxis.set_major_locator(tick_locators)
                axes["2"].set_xticklabels(tick_labels)
                for _, spine in heatmap.spines.items():
                    spine.set_visible(True)
                axes["2"].set_xlabel(xlabel.replace("_", " ").capitalize())
                axes["2"].set_ylabel(ylabel.replace("_", " ").capitalize())

                axes["2"].plot(
                    [0, bins], [bins, 0], color="black", linestyle="dashed"
                )

                sns.despine(ax=axes["2"])

                # Plot density plots.
                sns.kdeplot(
                    data=similarities,
                    x=xlabel,
                    clip=(0, 1),
                    legend=True,
                    color="black",
                    fill=True,
                    ax=axes["1"],
                )
                axes["1"].set_xlim(0, 1)
                axes["1"].xaxis.set_ticklabels([])
                axes["1"].yaxis.set_major_locator(tick_locators)
                axes["1"].set_yticks([])
                sns.despine(ax=axes["1"], left=True)
                sns.kdeplot(
                    data=similarities,
                    y=ylabel,
                    clip=(0, 1),
                    legend=True,
                    color="black",
                    fill=True,
                    ax=axes["3"],
                )
                axes["3"].set_ylim(0, 1)
                axes["3"].yaxis.set_ticklabels([])
                axes["3"].xaxis.set_major_locator(tick_locators)
                axes["3"].set_xticks([])
                sns.despine(ax=axes["3"], bottom=True)
                for ax in [axes[c] for c in "13"]:
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    
            cbar_ax.set_ylabel("Proportion of pairs")
            cbar_ax.yaxis.set_label_position("left")
            cbar_ax.spines["outline"].set(visible=True, lw=.8, edgecolor="black")
            
            # Middle panel: Compare similarities vs explained intensity.
            axes_left = fig.add_subfigure(gs[1, 0]).subplot_mosaic(mosaic)
            axes_middle = fig.add_subfigure(gs[1, 1]).subplot_mosaic(mosaic)
            axes_right = fig.add_subfigure(gs[1, 2]).subplot_mosaic(mosaic)
            cbar_ax = fig.add_axes([-0.04, 0.45, 0.02, 0.15])
            
            labels = np.asarray([
                ["cosine_explained", "cosine"],
                ["neutral_loss_explained", "neutral_loss"],
                ["modified_cosine_explained", "modified_cosine"],
            ])

            for i, (axes, (xlabel, ylabel)) in enumerate(
                zip([axes_left, axes_middle, axes_right], labels)
            ):
                # Plot heatmaps.
                hist, _, _ = np.histogram2d(
                    similarities[xlabel],
                    similarities[ylabel],
                    bins=bins,
                    range=[[0, 1], [0, 1]],
                )
                hist /= len(similarities)
                heatmap = sns.heatmap(
                    np.rot90(hist),
                vmin=np.min(hist),
                    vmax=np.max(hist),
                    cmap="viridis",
                    cbar=i == 2,
                    cbar_kws={"format": mticker.StrMethodFormatter("{x:.3%}")},
                    cbar_ax=cbar_ax if i == 2 else None,
                    square=True,
                    xticklabels=False,
                    yticklabels=False,
                    ax=axes["2"],
                    norm=LogNorm(vmax=np.max(hist)),
                )
                axes["2"].yaxis.set_major_locator(tick_locators)
                axes["2"].set_yticklabels(tick_labels[::-1])
                axes["2"].xaxis.set_major_locator(tick_locators)
                axes["2"].set_xticklabels(tick_labels)
                axes["2"].xaxis.set_major_formatter(mticker.PercentFormatter())
                for _, spine in heatmap.spines.items():
                    spine.set_visible(True)
                axes["2"].set_xlabel("Explained intensity")
                axes["2"].set_ylabel(ylabel.replace("_", " ").capitalize())

                sns.despine(ax=axes["2"])

                # Plot density plots.
                sns.kdeplot(
                    data=similarities,
                    x=xlabel,
                    clip=(0, 1),
                    legend=True,
                    color="black",
                    fill=True,
                    ax=axes["1"],
                )
                axes["1"].set_xlim(0, 1)
                axes["1"].xaxis.set_ticklabels([])
                axes["1"].yaxis.set_major_locator(tick_locators)
                axes["1"].set_yticks([])
                sns.despine(ax=axes["1"], left=True)
                sns.kdeplot(
                    data=similarities,
                    y=ylabel,
                    clip=(0, 1),
                    legend=True,
                    color="black",
                    fill=True,
                    ax=axes["3"],
                )
                axes["3"].set_ylim(0, 1)
                axes["3"].yaxis.set_ticklabels([])
                axes["3"].xaxis.set_major_locator(tick_locators)
                axes["3"].set_xticks([])
                sns.despine(ax=axes["3"], bottom=True)
                for ax in [axes[c] for c in "13"]:
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    
            cbar_ax.set_ylabel("Proportion of pairs")
            cbar_ax.yaxis.set_label_position("left")
            cbar_ax.spines["outline"].set(visible=True, lw=.8, edgecolor="black")
            
            # Bottom panel: Evaluate similarities in terms of the Tanimoto index.
            ax = fig.add_subplot(gs[2, :])
            
            sns.violinplot(
                data=similarities_tanimoto,
                x="tanimoto_interval",
                y="value",
                hue="variable",
                hue_order=["cosine", "neutral_loss", "modified_cosine"],
                cut=0,
                scale="width",
                scale_hue=False,
                ax=ax,
            )
            
            ax.set_xlabel("Tanimoto index")
            ax.set_ylabel("Spectrum similarity")
            for label in ax.legend().get_texts():
                label.set_text(label.get_text().replace("_", " ").capitalize())
            sns.move_legend(
                ax,
                "lower center",
                bbox_to_anchor=(.5, 1),
                ncol=3,
                title=None,
                frameon=False,
            )

            sns.despine(ax=ax)
            
            # Subplot labels.
            for y, label in zip([1, 2/3, 0.35], "abc"):
                fig.text(
                    -0.05, y, label, fontdict=dict(fontsize="xx-large", weight="bold")
                )

            # Save figure.
            #plt.savefig("gnps_libraries.png", dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()

    @staticmethod
    def plot_gnps_libraries_tanimoto(similarities):
        
        similarities_filtered = similarities[similarities["tanimoto"] > 0.9]
        print(
            f"Number of spectrum pairs with Tanimoto > 0.9: "
            f"{len(similarities_filtered):,}"
        )

        pair_classes = (
            similarities_filtered
            #.join(metadata["class"], on="id1")
            #.join(metadata["class"], on="id2", rsuffix="2")
            #.rename(columns={"class": "class1"})
        )

        
        pair_classes["class"] = pair_classes[["class1", "class2"]].apply(
            lambda x: x.class1 if x.class1 == x.class2 else f"{x.class1} ↔ {x.class2}",
            axis=1,
        )
        pair_top = pair_classes["class"].value_counts().head(10).index
        pair_classes.loc[
            ~pair_classes["class"].isin(pair_top), "class"
        ] = "Other compound classes"
        similarities_pairs = pd.melt(
            pair_classes,
            id_vars="class",
            value_vars=["cosine", "neutral_loss", "modified_cosine"],
        )
        order = (pair_classes.groupby("class")["modified_cosine"]
                .median().sort_values().index)


        mosaic = """
        11111.
        222223
        222223
        222223
        222223
        222223
        """

        bins = 100
        tick_locators = mticker.FixedLocator(np.arange(0, bins + 1, bins / 4))
        tick_labels = np.asarray([f"{a:.2f}" for a in np.arange(0, 1.01, 0.25)])


        with sns.plotting_context("paper", font_scale=1.6):
            fig = plt.figure(constrained_layout=True, figsize=(7.2 * 2, 7.2 / 1.618 * 2))
            gs = GridSpec(2, 3, figure=fig)
            
            # Top panel: Compare different similarities.
            axes_left = fig.add_subfigure(gs[0, 0]).subplot_mosaic(mosaic)
            axes_middle = fig.add_subfigure(gs[0, 1]).subplot_mosaic(mosaic)
            axes_right = fig.add_subfigure(gs[0, 2]).subplot_mosaic(mosaic)
            cbar_ax = fig.add_axes([-0.04, 0.75, 0.02, 0.15])
            
            labels = np.asarray([
                ["cosine_explained", "cosine"],
                ["neutral_loss_explained", "neutral_loss"],
                ["modified_cosine_explained", "modified_cosine"],
            ])

            for i, (axes, (xlabel, ylabel)) in enumerate(
                zip([axes_left, axes_middle, axes_right], labels)
            ):
                # Plot heatmaps.
                hist, _, _ = np.histogram2d(
                    similarities_filtered[xlabel],
                    similarities_filtered[ylabel],
                    bins=bins,
                    range=[[0, 1], [0, 1]],
                )
                hist /= len(similarities_filtered)
                hist[hist == 0.0] = np.nan
                heatmap = sns.heatmap(
                    np.rot90(hist),
                    vmin=0.0,
                    vmax=0.001,
                    cmap="viridis",
                    cbar=i == 2,
                    cbar_kws={"format": mticker.StrMethodFormatter("{x:.2%}")},
                    cbar_ax=cbar_ax if i == 2 else None,
                    square=True,
                    xticklabels=False,
                    yticklabels=False,
                    ax=axes["2"],
                )
                axes["2"].yaxis.set_major_locator(tick_locators)
                axes["2"].set_yticklabels(tick_labels[::-1])
                axes["2"].xaxis.set_major_locator(tick_locators)
                axes["2"].set_xticklabels(tick_labels)
                axes["2"].xaxis.set_major_formatter(mticker.PercentFormatter())
                for _, spine in heatmap.spines.items():
                    spine.set_visible(True)
                axes["2"].set_xlabel("Explained intensity")
                axes["2"].set_ylabel(ylabel.replace("_", " ").capitalize())

                sns.despine(ax=axes["2"])

                # Plot density plots.
                sns.kdeplot(
                    data=similarities_filtered,
                    x=xlabel,
                    clip=(0, 1),
                    legend=True,
                    color="black",
                    fill=True,
                    ax=axes["1"],
                )
                axes["1"].set_xlim(0, 1)
                axes["1"].xaxis.set_ticklabels([])
                axes["1"].yaxis.set_major_locator(tick_locators)
                axes["1"].set_yticks([])
                sns.despine(ax=axes["1"], left=True)
                sns.kdeplot(
                    data=similarities_filtered,
                    y=ylabel,
                    clip=(0, 1),
                    legend=True,
                    color="black",
                    fill=True,
                    ax=axes["3"],
                )
                axes["3"].set_ylim(0, 1)
                axes["3"].yaxis.set_ticklabels([])
                axes["3"].xaxis.set_major_locator(tick_locators)
                axes["3"].set_xticks([])
                sns.despine(ax=axes["3"], bottom=True)
                for ax in [axes[c] for c in "13"]:
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    
            cbar_ax.set_ylabel("Proportion of pairs")
            cbar_ax.yaxis.set_label_position("left")
            cbar_ax.spines["outline"].set(visible=True, lw=.8, edgecolor="black")
            
            # Bottom panel: Similarities per compound class.
            ax = fig.add_subplot(gs[1, :])
            
            sns.violinplot(
                data=similarities_pairs,
                x="class",
                y="value",
                hue="variable",
                order=order,
                hue_order=["cosine", "neutral_loss", "modified_cosine"],
                cut=0,
                scale="width",
                scale_hue=False,
                ax=ax,
            )
            ax.set_xlabel("")
            ax.set_ylabel("Spectrum similarity")
            ax.set_xticklabels(
                [
                    textwrap.fill(label.get_text(), width=10, break_long_words=False)
                    for label in ax.get_xticklabels()
                ],
                fontdict={"fontsize": "x-small"},
            )
            for label in ax.legend().get_texts():
                label.set_text(label.get_text().replace("_", " ").capitalize())
            sns.move_legend(
                ax,
                "lower center",
                bbox_to_anchor=(0.5, 0.95),
                ncol=3,
                title=None,
                frameon=False,
            )

            sns.despine(ax=ax)
            
            # Subplot labels.
            for y, label in zip([1, 0.52], "ab"):
                fig.text(
                    -0.05, y, label, fontdict=dict(fontsize="xx-large", weight="bold")
                )

            # Save figure.
            #plt.savefig("gnps_libraries_tanimoto.png", dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()

    @staticmethod
    def plot_roc_curve(recall_scores, precision_scores):
        # Plot the precision-recall curve
        plt.plot(recall_scores, precision_scores, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid()
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.show()