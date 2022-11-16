import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from moabb.datasets import BNCI2014009
from moabb.paradigms import P300
from pyriemann.utils.viz import plot_waveforms

###############################################################################
# Local functions


def plot_individual_proba(subj, scorers, df_char, trials=None):
    from matplotlib.lines import Line2D

    plt.set_cmap("tab10")  # "cividis"
    colors = ["C0", "C3"]
    trials = df_char["trial"].unique() if trials is None else trials
    for trial in trials:
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
        for ns, ax in zip(scorers.keys(), axes):
            ntdf = df_char[
                (df_char["type"] == "Non-target")
                & (df_char["scorer"] == ns)
                & (df_char["trial"] == trial)
            ]
            data = ntdf[["proba"]].to_numpy().reshape((120, 35))
            ax.plot(data, alpha=0.5, lw=0.7, ls="-", c=colors[0])
            tdf = df_char[
                (df_char["type"] == "Target")
                & (df_char["scorer"] == ns)
                & (df_char["trial"] == trial)
            ]
            data = tdf[["proba"]].to_numpy()
            ax.plot(data, alpha=1, lw=1.2, ls="-", c=colors[1])
            if ns == "ASAP":
                ax.set(
                    title=f"{ns}", xlabel="Flash index", ylabel="Character probability",
                )
            else:
                ax.set(
                    title=f"{ns}", xlabel="Flash index",
                )
        tgt_lgd = [
            Line2D([0], [0], alpha=0.5, lw=0.7, ls="-", c=colors[0]),
            Line2D([0], [0], alpha=1, lw=1.2, ls="-", c=colors[1]),
        ]
        plt.legend(tgt_lgd, ["Non-target", "Target"])
        plt.tight_layout()
        plt.savefig(f"figarticle_proba_individual_trial_{trial}_subj{subj}.png")
        plt.close()


def plot_proba_avg_repetition(subj, df_char, scorers, ci=95):
    plt.set_cmap("tab10")  # "cividis"
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.lineplot(
        data=df_char[0].rename(columns={"scorer": "Method", "type": "Character"}),
        x="repetition",
        y="proba",
        style="Character",
        hue="Method",
        ci=ci,
        palette=["C0", "C3"],
        ax=axes[0],
        legend=True,
    )
    axes[0].set(
        title="BNCI 2014-008",
        ylabel="Averaged character probabilities",
        xlabel="Repetition index",
    )
    sns.lineplot(
        data=df_char[1].rename(columns={"scorer": "Method", "type": "Character"}),
        x="repetition",
        y="proba",
        style="Character",
        hue="Method",
        ci=ci,
        palette=["C0", "C3"],
        ax=axes[1],
        legend=True,
    )
    axes[1].set(
        title="BNCI 2014-009",
        ylabel="Averaged character probabilities",
        xlabel="Repetition index",
    )
    plt.tight_layout()
    plt.savefig("figarticle_proba_repetition_all_trials_alldataset.png")
    plt.close()


def plot_acc_allsubj(df_allsubj, df_allsubj2, ci=95):
    plt.set_cmap("tab10")  # "cividis"
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.pointplot(
        data=df_allsubj[df_allsubj["type"] == "Target"].rename(
            columns={"scorer": "Method", "type": "Character"}
        ),
        x="repetition",
        y="acc",
        hue="Method",
        palette=["C0", "C3", "C4", "C6"],
        dodge=True,
        ci=ci,
        errwidth=0.4,
        capsize=0.1,
        scale=0.5,
        ax=axes[0],
    )
    axes[0].set(
        title="BNCI 2014-008",
        ylabel="Accuracy of character classification",
        xlabel="Repetition index",
    )
    sns.pointplot(
        data=df_allsubj2[df_allsubj2["type"] == "Target"].rename(
            columns={"scorer": "Method", "type": "Character"}
        ),
        x="repetition",
        y="acc",
        hue="Method",
        palette=["C0", "C3", "C4", "C6"],
        dodge=True,
        ci=ci,
        errwidth=0.4,
        capsize=0.1,
        scale=0.5,
        ax=axes[1],
    )
    axes[1].set(
        title="BNCI 2014-009",
        ylabel="Accuracy of character classification",
        xlabel="Repetition index",
    )
    plt.tight_layout()
    plt.savefig("figarticle_acc_repetition_all_subj.png")
    plt.close()


def plot_itr_allsubj(df_allsubj, df_allsubj2, ci=95):
    plt.set_cmap("tab10")  # "cividis"
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.pointplot(
        data=df_allsubj[df_allsubj["type"] == "Target"].rename(
            columns={"scorer": "Method", "type": "Character"}
        ),
        x="repetition",
        y="ITR",
        hue="Method",
        palette=["C0", "C3", "C4", "C6"],
        dodge=True,
        ci=ci,
        errwidth=0.4,
        capsize=0.1,
        scale=0.5,
        ax=axes[0],
    )
    axes[0].set(
        title="BNCI 2014-008", ylabel="ITR (bits/min)", xlabel="Repetition index"
    )
    sns.pointplot(
        data=df_allsubj2[df_allsubj2["type"] == "Target"].rename(
            columns={"scorer": "Method", "type": "Character"}
        ),
        x="repetition",
        y="ITR",
        hue="Method",
        palette=["C0", "C3", "C4", "C6"],
        dodge=True,
        ci=ci,
        errwidth=0.4,
        capsize=0.1,
        scale=0.5,
        ax=axes[1],
    )
    axes[1].set(
        title="BNCI 2014-009", ylabel="BCI ITR (bits/min)", xlabel="Repetition index"
    )
    plt.tight_layout()
    plt.savefig("figarticle_itr_repetition_allsubj.png")
    plt.close()


###############################################################################
# Read CSV data and reformat

df_allsubj = pd.read_csv("bnci2014008.csv")
scorers = {"ASAP": None, "MDM+OM": None, "XDAWN+OM": None, "RegLDA+OM": None}
rename_scorers = {
    "pMDM+BA": "ASAP",
    "maxmin": "MDM+OM",
    "Xdawn+lda+OM": "xDAWN+OM",
    "reglda+OM": "RegLDA+OM",
}

for ns, nn in rename_scorers.items():
    df_allsubj.loc[df_allsubj["scorer"] == ns, "scorer"] = nn
rename_tgt = {"target": "Target", "nontarget": "Non-target"}
for ts, tn in rename_tgt.items():
    df_allsubj.loc[df_allsubj["type"] == ts, "type"] = tn
df_allsubj2 = pd.read_csv("bnci2014009_addclassif2.csv")
for ns, nn in rename_scorers.items():
    df_allsubj2.loc[df_allsubj2["scorer"] == ns, "scorer"] = nn
for ts, tn in rename_tgt.items():
    df_allsubj2.loc[df_allsubj2["type"] == ts, "type"] = tn

###############################################################################
# Plot example of individual probabilities

subj_prob, trial_prob = 2, 13
plot_individual_proba(
    subj_prob, scorers, df_allsubj[df_allsubj["subject"] == subj_prob], [trial_prob]
)

###############################################################################
# Plot average repetition for a subjet

subj_prob = 3
plot_proba_avg_repetition(subj_prob, [df_allsubj, df_allsubj2], scorers, ci=95)

###############################################################################
# Plot average accuracies for all subject and all pipelines

plot_acc_allsubj(df_allsubj, df_allsubj2, ci=95)

###############################################################################
# Compute and plot ITR for all subjects and pipelines

N = 36  # Number of possible choices in the speller keyboard
df_allsubj["ITR"] = 0.0
for subj in df_allsubj["subject"].unique():
    for sc in df_allsubj["scorer"].unique():
        for rep in df_allsubj["repetition"].unique():
            P = df_allsubj[
                (df_allsubj["subject"] == subj)
                & (df_allsubj["repetition"] == rep)
                & (df_allsubj["scorer"] == sc)
            ]["acc"].mean()
            T = rep * 3 / 60
            B = np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1))
            ITR = B / T
            df_allsubj.loc[
                (df_allsubj["subject"] == subj)
                & (df_allsubj["repetition"] == rep)
                & (df_allsubj["scorer"] == sc),
                "ITR",
            ] = ITR

df_allsubj2["ITR"] = 0.0
for subj in df_allsubj2["subject"].unique():
    for sc in df_allsubj["scorer"].unique():
        for rep in df_allsubj2["repetition"].unique():
            P = df_allsubj2[
                (df_allsubj2["subject"] == subj)
                & (df_allsubj2["repetition"] == rep)
                & (df_allsubj2["scorer"] == sc)
            ]["acc"].mean()
            T = rep * 3 / 60
            B = np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1))
            ITR = B / T
            df_allsubj2.loc[
                (df_allsubj2["subject"] == subj)
                & (df_allsubj2["repetition"] == rep)
                & (df_allsubj2["scorer"] == sc),
                "ITR",
            ] = ITR
plot_itr_allsubj(df_allsubj, df_allsubj2, ci=95)

###############################################################################
# generate a copy-paste output for a latex table

ds1, ds2 = "BNCI 2014-008", "BNCI 2014-009"
tab = r"""
\begin{table*}[t]
  \centering
  \begin{tabular}{|l|c|c|c|c|c|c|c|c|c|c|}
    \hline
       Repetition index & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10
       \rule[-7pt]{0pt}{20pt} \\ \hline
       \hline
"""

for sc in df_allsubj["scorer"].unique():
    tab += f"       {ds1}, {sc} "
    for rep in df_allsubj["repetition"].unique():
        acc = df_allsubj[
            (df_allsubj["repetition"] == rep) & (df_allsubj["scorer"] == sc)
        ]["acc"].mean()
        tab += f"& {acc:.2} "
    tab += "\n" + r"       \rule[-7pt]{0pt}{20pt} \\ \hline" + "\n"
# tab += r"       \cline{1-9}\n"
for sc in df_allsubj2["scorer"].unique():
    tab += f"       {ds2}, {sc} "
    for rep in df_allsubj2["repetition"].unique():
        acc = df_allsubj2[
            (df_allsubj2["repetition"] == rep) & (df_allsubj2["scorer"] == sc)
        ]["acc"].mean()
        tab += f"& {acc:.2} "
    tab += (
        r"& \multicolumn{2}{c}{} \n"
        + r"       \rule[-7pt]{0pt}{20pt} \\ \cline{1-9}"
        + "\n"
    )
tab += r"""  \end{tabular}
  \caption{Numerical values of Figure~\ref{fig:accuracy}: averaged character classification accuracy as a function of repetition.}
  \label{tab:accuracy}
\end{table*}"""
print(tab)


tab = r"""
\begin{table*}[t]
  \centering
  \begin{tabular}{|l|c|c|c|c|c|c|c|c|c|c|}
    \hline
    Repetition index & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10
    \rule[-7pt]{0pt}{20pt} \\ \hline
    \hline
"""
for sc in df_allsubj["scorer"].unique():
    tab += f"       {ds1}, {sc} "
    for rep in df_allsubj["repetition"].unique():
        itr = df_allsubj[
            (df_allsubj["repetition"] == rep) & (df_allsubj["scorer"] == sc)
        ]["ITR"].mean()
        tab += f"& {itr:.1f} "
    tab += "\n" + r"       \rule[-7pt]{0pt}{20pt} \\ \hline" + "\n"
# tab += r"       \cline{1-9}\n"
for sc in df_allsubj2["scorer"].unique():
    tab += f"       {ds2}, {sc} "
    for rep in df_allsubj2["repetition"].unique():
        itr = df_allsubj2[
            (df_allsubj2["repetition"] == rep) & (df_allsubj2["scorer"] == sc)
        ]["ITR"].mean()
        tab += f"& {itr:.1f} "
    tab += (
        r"& \multicolumn{2}{c}{} \n"
        + r"       \rule[-7pt]{0pt}{20pt} \\ \cline{1-9}"
        + "\n"
    )
tab += r"""  \end{tabular}
  \caption{Numerical values of Figure~\ref{fig:itr}: averaged BCI ITR as a function of repetition.}
  \label{tab:itr}
\end{table*}"""
print(tab)
