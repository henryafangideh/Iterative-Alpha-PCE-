import json, sys, time, os
sys.path.insert(0, "/home/claude/bench")
from multiprocessing import Pool
import engine as E

TASKS = [  # (task, n_train, n_test)
    ("wdbc", 100, 100),
    ("wine", 100, 78),
    ("digits_0v8", 100, 100),
    ("digits_3v5", 100, 100),
    ("digits_4v9", 100, 100),
    ("digits_all", 300, 300),
]
SEEDS = range(10)
OUT = "/home/claude/bench/results.jsonl"


def job(args):
    task, ntr, nte, seed = args
    t0 = time.time()
    r = E.run_one(task, seed, ntr, nte)
    r["seconds"] = round(time.time() - t0, 1)
    return r


if __name__ == "__main__":
    jobs = [(t, a, b, s) for (t, a, b) in TASKS for s in SEEDS]
    open(OUT, "w").close()
    with Pool(2) as pool:
        for r in pool.imap_unordered(job, jobs):
            with open(OUT, "a") as f:
                f.write(json.dumps(r) + "\n")
    open("/home/claude/bench/DONE", "w").write("ok")
