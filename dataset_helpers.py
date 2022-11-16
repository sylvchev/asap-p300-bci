import numpy as np


def relabel_events_BNCI(ev, n_stim, n_repetition):
    """Reformat labels from BNCI dataset for cumulative evaluation

    In BNCI, events are grouped by pair: target/nontarget labels and
    position labels (column 1 to 6 or row 1 to 6). Those event pair share the
    same time sample and are not ordered.

    Original event label are 1 for non target, 2 for target, 3 to 8 for column
    1 to 6 and 9 to 14 for line 1 to 6

    Output events are encoded with 4+ digits: thousands encode the trial number,
    tens/hundreds indicate the position and unit indicate target status:
    1010 is trial 1/col 1/non target,
    2041 is trial 2/col 4/target,
    4080 is trial 4/line 3/non target,
    35111 is trial 35/line 6/target
    """
    i, t_len, n_ev = 0, n_stim * n_repetition * 2, len(ev)

    new_ev = []
    while i < n_ev:
        tgt = pos = None

        if ev[i, 2] == 1:
            # non target event is first
            tgt, t = 0, ev[i, 0]
        elif ev[i, 2] == 2:
            # target event is first
            tgt, t = 1, ev[i, 0]
        else:
            # position event is first
            pos, t = ev[i, 2] - 2, ev[i, 0]

        i += 1
        trial = (i // t_len) + 1
        if t != ev[i, 0]:
            raise ValueError("event time differs within pair")
        if ev[i, 2] == 1:
            tgt = 0
        elif ev[i, 2] == 2:
            tgt = 1
        else:
            pos = ev[i, 2] - 2
        new_ev.append([t, 0, trial * 1000 + pos * 10 + tgt])
        i += 1
    new_ev = np.array(new_ev)

    event_id = {}
    for trial_idx in range(trial):
        tc = (trial_idx + 1) * 1000
        for pos_idx in range(12):
            pc = (pos_idx + 1) * 10
            if pos_idx < 6:
                event_id[f"trial{trial_idx + 1}/col{pos_idx + 1}/nontarget"] = tc + pc
                event_id[f"trial{trial_idx + 1}/col{pos_idx + 1}/target"] = tc + pc + 1
            else:
                event_id[f"trial{trial_idx + 1}/row{pos_idx % 6 + 1}/nontarget"] = (
                    tc + pc
                )
                event_id[f"trial{trial_idx + 1}/row{pos_idx % 6 + 1}/target"] = (
                    tc + pc + 1
                )

    return new_ev, event_id


def relabel_events_BNCI2014008(ev):
    n_stim, n_repetition = 12, 10
    return relabel_events_BNCI(ev, n_stim, n_repetition)


def relabel_events_BNCI2014009(ev):
    n_stim, n_repetition = 12, 8
    return relabel_events_BNCI(ev, n_stim, n_repetition)


def l2c_BNCI2014008(char):
    """Encode BNCI 2014-008 char into specific code

    Symbols used in BNCI 2014-008 are A-Z, 1-9 and _. The A-Z letters are encoded
    as 0-25, 1-9 numbers as 26-34 and _ as 35.
    """
    if char.isalpha() and char.isupper():
        return ord(char) - 65
    elif char.isdigit():
        return ord(char) - 23
    elif char == "_":
        return ord(char) - 60
    else:
        raise ValueError("Undefined letter")


def c2l_BNCI2014008(code):
    """Decode BNCI 2014-008 int code into char

    Symbols used in BNCI 2014-008 are A-Z, 1-9 and _. The A-Z letters are encoded
    as 0-25, 1-9 numbers as 26-34 and _ as 35.
    """
    if code < 26:
        return chr(code + 65)
    elif 26 <= code < 35:
        return chr(code + 23)
    elif code == 35:
        return "_"
    else:
        raise ValueError("Undefined letter")


l2c_BNCI2014009 = l2c_BNCI2014008
c2l_BNCI2014009 = c2l_BNCI2014008

tf_symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"


def l2c_tf(char):
    code = tf_symbols.find(char)
    if code == -1:
        raise ValueError("Undefined letter")
    return code


def c2l_tf(code):
    return tf_symbols[code]
