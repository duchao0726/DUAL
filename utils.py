from __future__ import division
import numpy as np
import time
import os
import errno
import logging
import coloredlogs


def get_floats(floats_str):
    if not floats_str:
        return []
    return [float(s) for s in floats_str.split(",")]


def get_ints(ints_str):
    if not ints_str:
        return []
    return [int(s) for s in ints_str.split(",")]


# parameters tagging helper function
def param_tag(value):
    if value == 0.0:
        return "00"
    exp = np.floor(np.log10(value))
    leading = ("%e" % value)[0]
    return "%s%d" % (leading, exp)


def setup_logging(args):
    dirname_base = args.output if hasattr(args, "output") else "output"
    logger = logging.getLogger("COLOREDLOGS")
    FORMAT = "[%(asctime)s] %(message)s"
    DATEFMT = "%H:%M:%S"
    LEVEL_STYLES = dict(
        debug=dict(color="blue"),
        info=dict(color="green"),
        verbose=dict(),
        warning=dict(color="yellow"),
        error=dict(color="red"),
        critical=dict(color="magenta"),
    )
    coloredlogs.install(logger=logger, level=args.loglv, fmt=FORMAT, datefmt=DATEFMT, level_styles=LEVEL_STYLES)

    # Determine suffix
    suffix = ""
    suffix += args.dataset if hasattr(args, "dataset") else ""
    suffix += "-" if suffix else ""
    suffix += args.method if hasattr(args, "method") else ""
    suffix += "-" if suffix else ""
    suffix += "{{" + (str(args.message if args.message else "debug") if hasattr(args, "message") else "") + "}}"
    suffix += "-nn" + str(args.dims) if hasattr(args, "dims") else ""
    suffix += "-lr" + str(args.lr) if hasattr(args, "lr") else ""
    suffix += "-lrd" + str(args.lr_decay) if hasattr(args, "lr_decay") else ""
    suffix += "-wd" + str(args.weight_decay) if hasattr(args, "weight_decay") else ""
    suffix += "-bs" + str(args.batch_size) if hasattr(args, "batch_size") else ""
    suffix += "-ep" + str(args.epochs) if hasattr(args, "epochs") else ""
    suffix += "-ml" + str(args.maxlen) if hasattr(args, "maxlen") else ""
    suffix += "-seed" + str(args.seed) if hasattr(args, "seed") else ""
    suffix += "-ind" + str(args.n_ind) if hasattr(args, "n_ind") else ""
    suffix += "-ls" + str(args.lengthscale) if hasattr(args, "lengthscale") else ""
    suffix += "-ap" + str(args.amplitude) if hasattr(args, "amplitude") else ""
    suffix += "-T" + str(args.temp) if hasattr(args, "temp") else ""
    suffix += "-km" + str(args.km_coeff) if hasattr(args, "km_coeff") else ""
    suffix += "-diag" + str(args.diag_cov) if hasattr(args, "diag_cov") else ""
    suffix += "-pm" + str(args.prior_mean) if hasattr(args, "prior_mean") else ""

    # Determine prefix
    prefix = time.strftime("%Y-%m-%d--%H-%M")

    prefix_counter = 0
    dirname = dirname_base + "/%s.%s" % (prefix, suffix)
    while True:
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise e
            prefix_counter += 1
            dirname = dirname_base + "/%s+%d.%s" % (prefix, prefix_counter, suffix)
        else:
            break

    formatter = logging.Formatter(FORMAT, DATEFMT)
    logger_fname = os.path.join(dirname, "logfile.txt")
    fh = logging.FileHandler(logger_fname)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger, dirname


def calc_auc(raw_arr):
    arr = sorted(raw_arr, key=lambda d: d[0], reverse=True)
    pos, neg = 0.0, 0.0
    for record in arr:
        if record[1] == 1.0:
            pos += 1
        else:
            neg += 1

    fp, tp = 0.0, 0.0
    xy_arr = []
    for record in arr:
        if record[1] == 1.0:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp / (neg + 1e-8), tp / (pos + 1e-8)])

    auc = 0.0
    prev_x = 0.0
    prev_y = 0.0
    for x, y in xy_arr:
        if x != prev_x:
            auc += (x - prev_x) * (y + prev_y) / 2.0
            prev_x = x
            prev_y = y

    return auc
