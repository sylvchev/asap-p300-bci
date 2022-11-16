import matplotlib.pyplot as plt
import mne
import pandas as pd
import seaborn as sns
from dataset_helpers import c2l_BNCI2014009, l2c_BNCI2014009, relabel_events_BNCI2014009
from mne.decoding import Vectorizer
from moabb.datasets import BNCI2014009
from moabb.paradigms import P300
from p300bayesianaccumulation import Scorer_BA, Scorer_OM
from pyriemann.classification import MDM
from pyriemann.estimation import XdawnCovariances
from pyriemann.spatialfilters import Xdawn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline

###############################################################################
# Local functions


def get_epochs(subj, sess, tmin, tmax, baseline, fmin, fmax):
    """get epoch from BNCI 2014-009 subject"""
    data = dataset.get_data(subjects=[subj])
    raw = data[subj][f"session_{sess}"]["run_0"]
    raw_ev = mne.find_events(raw)
    ev, event_id = relabel_events_BNCI2014009(raw_ev)

    raw = raw.filter(fmin, fmax, method="iir")
    picks = mne.pick_types(raw.info, eeg=True, stim=False)
    ep = mne.Epochs(
        raw,
        ev,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        proj=False,
        baseline=baseline,
        preload=True,
        verbose=False,
        picks=picks,
        on_missing="ignore",
    )
    return ep


def calibration(ep, n_calibration, nfilter, est, xest):
    """Train MDM pipeline on n_calibration trials, return pipeline"""
    ep_calib, ev_tgt, ev_nontgt = [], [], []
    for i in range(n_calibration):
        ep_calib.append(ep[f"trial{i+1}"])
        ev_nontgt += [
            eid for eid in ep[f"trial{i+1}"].event_id.values() if eid % 2 == 0
        ]
        ev_tgt += [eid for eid in ep[f"trial{i+1}"].event_id.values() if eid % 2 != 0]
    ep_calib = mne.concatenate_epochs(ep_calib)
    merged_ev = mne.merge_events(ep_calib.events, ev_nontgt, 0, replace_events=True)
    merged_ev = mne.merge_events(merged_ev, ev_tgt, 1, replace_events=True)
    ep_calib.events = merged_ev
    ep_calib.event_id = {"Target": 1, "NonTarget": 0}
    X_calib = ep_calib.get_data()
    y_calib = ep_calib.events[:, -1]

    cov_est = XdawnCovariances(
        nfilter=nfilter,
        applyfilters=True,
        classes=[ep_calib.event_id["Target"]],
        estimator=est,
        xdawn_estimator=xest,
        baseline_cov=None,
    )
    mdm = MDM(metric="riemann")
    ppl = Pipeline([("cov_est", cov_est), ("mdm", mdm)])
    ppl.fit(X_calib, y_calib)
    return ppl


def calibration_ppl(ep, n_calibration, sfreq, ppl):
    """Train MDM pipeline on n_calibration trials, return pipeline"""
    ep_calib, ev_tgt, ev_nontgt = [], [], []
    for i in range(n_calibration):
        ep_calib.append(ep[f"trial{i+1}"])
        ev_nontgt += [
            eid for eid in ep[f"trial{i+1}"].event_id.values() if eid % 2 == 0
        ]
        ev_tgt += [eid for eid in ep[f"trial{i+1}"].event_id.values() if eid % 2 != 0]
    ep_calib = mne.concatenate_epochs(ep_calib).resample(sfreq, npad="auto")
    merged_ev = mne.merge_events(ep_calib.events, ev_nontgt, 0, replace_events=True)
    merged_ev = mne.merge_events(merged_ev, ev_tgt, 1, replace_events=True)
    ep_calib.events = merged_ev
    ep_calib.event_id = {"Target": 1, "NonTarget": 0}
    X_calib = ep_calib.get_data()
    y_calib = ep_calib.events[:, -1]

    ppl.fit(X_calib, y_calib)
    return ppl


def update_mdm_cm(mdm_eval, erp_likelihood, flash, tgt, group):
    if erp_likelihood[flash, 0] > erp_likelihood[flash, 1] and tgt not in group:
        mdm_eval["TN"] += 1
    elif erp_likelihood[flash, 0] < erp_likelihood[flash, 1] and tgt in group:
        mdm_eval["TP"] += 1
    if erp_likelihood[flash, 0] > erp_likelihood[flash, 1] and tgt in group:
        mdm_eval["FN"] += 1
    elif erp_likelihood[flash, 0] < erp_likelihood[flash, 1] and tgt not in group:
        mdm_eval["FP"] += 1
    return mdm_eval


def score_subject(ep, ppl, scorers, sess, n_trials, sfreq=None):
    """Use ppl pipelines and scorers on ep epochs for n_trials"""
    results_char = []
    mdm_eval = {"TN": 0, "TP": 0, "FN": 0, "FP": 0}
    for trial in range(n_trials):
        for sc in scorers.values():
            sc.reset()

        ep_test = ep[f"trial{trial+1}"]
        X_trial = ep_test.resample(sfreq).get_data() if sfreq else ep_test.get_data()

        # find target letter
        tgt_letters = [
            set(chars[pos.split("/")[1]])
            for pos in list(ep_test["target"].event_id.keys())
        ]
        tgt = (tgt_letters[0] & tgt_letters[1]).pop()
        print(f"Target is: {tgt}")

        erp_likelihood = ppl.predict_proba(X_trial)
        for flash in range(len(X_trial)):
            group = chars[list(ep_test[flash].event_id.keys())[0].split("/")[1]]

            mdm_eval = update_mdm_cm(mdm_eval, erp_likelihood, flash, tgt, group)
            for sc in scorers.values():
                sc.update(proba=erp_likelihood[flash, :], group=group)

            res_header = {
                "subject": subj,
                "trial": trial + 1,
                "session": sess + 1,
                "repetition": (flash // 12) + 1,
            }
            for char in symbols:
                for ns, sc in scorers.items():
                    results_char.append(
                        {
                            **res_header,
                            "flashes": flash + 1,
                            "letter": char,
                            "scorer": ns,
                            "proba": sc.get_proba()[l2c_BNCI2014009(char)],
                            "acc": 1 if sc.predict() == tgt else 0,
                            "type": "target" if tgt == char else "nontarget",
                        }
                    )
        print("prediction is ", end="")
        for ns, sc in scorers.items():
            print(f"{ns}={sc.predict()} {sc.get_proba().max():.3f} - ", end="")
        print()

    print(f"Overall MDM perf: {mdm_eval}")
    df_char = pd.DataFrame(results_char)

    return df_char


paradigm = P300()
dataset = BNCI2014009()

raw_event_id = {
    1: "NonTarget",
    2: "Target",
    3: "col_1",
    4: "col_2",
    5: "col_3",
    6: "col_4",
    7: "col_5",
    8: "col_6",
    9: "row_1",
    10: "row_2",
    11: "row_3",
    12: "row_4",
    13: "row_5",
    14: "row_6",
}
chars = {
    "col1": "AGMSY5",
    "col2": "BHNTZ6",
    "col3": "CIOU17",
    "col4": "DJPV28",
    "col5": "EKQW39",
    "col6": "FLRX4_",
    "row1": "ABCDEF",
    "row2": "GHIJKL",
    "row3": "MNOPQR",
    "row4": "STUVWX",
    "row5": "YZ1234",
    "row6": "56789_",
}
symbols = set("".join([c for c in chars.values()]))

df_allsubj = []
for subj in dataset.subject_list:
    ###############################################################################
    # Get epochs

    tmin, tmax = -0.3, 1.0
    baseline = -0.3, 0.0
    fmin, fmax = 0.1, 24
    ep_sess0 = get_epochs(subj, 0, tmin, tmax, baseline, fmin, fmax)
    ep_sess1 = get_epochs(subj, 1, tmin, tmax, baseline, fmin, fmax)
    ep_sess2 = get_epochs(subj, 2, tmin, tmax, baseline, fmin, fmax)

    ###############################################################################
    # Calibration on first session

    nfilter = 8
    est, xest = "scm", "scm"
    n_calibration = 6
    resample_freq = 128
    ppl = calibration(ep_sess0, n_calibration, nfilter, est, xest)
    # "MDM+OM" A. Barachant and M. Congedo, "A Plug&Play P300 BCI using
    # information geometry," arXiv preprint arXiv:1409.0107, 2014
    pxd = Pipeline(
        [
            ("xdawn", Xdawn(nfilter=2)),
            ("vect", Vectorizer()),
            ("lda", LDA(solver="lsqr", shrinkage="auto")),
        ]
    )
    ppl_xd = calibration_ppl(ep_sess0, n_calibration, resample_freq, pxd)
    # "xDAWN+OM" B. Rivet, A. Souloumiac, V. Attina, and G. Gibert,
    # "xDAWN algorithm to enhance evoked potentials: Application to
    # brain-computer interface," IEEE Trans Biomed Eng, vol. 56,
    # pp. 2035-2043, 2009.
    pld = Pipeline(
        [("vect", Vectorizer()), ("lda", LDA(solver="lsqr", shrinkage="auto"))]
    )
    ppl_ld = calibration_ppl(ep_sess0, n_calibration, resample_freq, pld)
    # "RegLDA+OM" B. Blankertz, S. Lemm, M. Treder, S. Haufe, and K. R. Müller,
    # "Single-trial analysis and classification of ERP components - A tutorial,"
    # NeuroImage, vol. 56, pp. 814-825, 2011.

    ###############################################################################
    # Online evaluation per trial

    for ep, sess in zip([ep_sess1, ep_sess2], [1, 2]):
        n_trials = len(set([t.split("/")[0] for t in list(ep.event_id.keys())]))

        scorers = {
            "pMDM+BA": Scorer_BA(symbols, c2l_BNCI2014009, l2c_BNCI2014009),
            "MDM+OM": Scorer_OM(symbols, c2l_BNCI2014009, l2c_BNCI2014009),
        }
        df_char = score_subject(ep, ppl, scorers, sess, n_trials)
        df_allsubj.append(df_char)
        sc_xd = {
            "Xdawn+lda+OM": Scorer_OM(symbols, c2l_BNCI2014009, l2c_BNCI2014009),
        }
        df_xd = score_subject(ep, ppl_xd, sc_xd, sess, n_trials, resample_freq)
        df_allsubj.append(df_xd)
        sc_lda = {
            "reglda+OM": Scorer_OM(symbols, c2l_BNCI2014009, l2c_BNCI2014009),
        }
        df_xd = score_subject(ep, ppl_ld, sc_lda, sess, n_trials, resample_freq)
        df_allsubj.append(df_xd)

df_allsubj = pd.concat(df_allsubj)
plot_acc_allsubj(df_allsubj)
df_allsubj.to_csv("bnci2014009_addclassif2.csv")
