# 교재 방식 MLP(회귀, 항등 출력 + MSE) 
# leave-one-person-out
# 규칙: 학습용 14명으로만 계산, 모든 fold에서 하이퍼파라미터 동일, 테스트 결과는 학습 중 선택에 미사용, fold마다 모델 초기화

import os, json, time, argparse
import numpy as np
import reader
from preprocess.mpii_norm import gaze_to_3d

ROOT = "data/EyeBased/MPIIGaze"
PERSONS = [f"p{k:02d}" for k in range(15)]
CACHE = "data/cache_mlp.npz"          # data/ 는 .gitignore 대상

def load_person(p):
    ds = reader.loader(f"{ROOT}/Label/{p}.label", f"{ROOT}/Image")
    X, Y = [], []
    for i in range(len(ds)):
        info, y = ds[i]
        X.append(np.concatenate([info["eye"].ravel(), info["head_pose"]])); Y.append(y)
    return np.array(X, np.float32), np.array(Y, np.float32)

def load_all():
    if os.path.exists(CACHE):
        z = np.load(CACHE); return {p: (z[p + "_X"], z[p + "_Y"]) for p in PERSONS}
    data = {p: load_person(p) for p in PERSONS}
    np.savez(CACHE, **{f"{p}_{k}": v for p, (x, y) in data.items() for k, v in (("X", x), ("Y", y))})
    return data

def angular_error(pred, gt):
    a, b = gaze_to_3d(pred), gaze_to_3d(gt)
    cos = np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

class MLP:                                     # Affine-ReLU-Affine-ReLU-Affine
    def __init__(self, sizes, wd):
        self.wd = wd
        self.W = [(np.sqrt(2.0 / a) * np.random.randn(a, b)).astype(np.float32) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
    def forward(self, x):
        self.xs = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1: x = np.maximum(0, x)
            self.xs.append(x)
        return x
    def backward(self, pred, y):
        d = (pred - y) / len(y)                # MSE (항등 출력)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = self.xs[i].T @ d + self.wd * self.W[i]
            gb[i] = d.sum(0)
            if i > 0: d = (d @ self.W[i].T) * (self.xs[i] > 0)
        return gW, gb

class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999):
        self.lr, self.b1, self.b2, self.t = lr, b1, b2, 0
        self.m = [np.zeros_like(p) for p in params]; self.v = [np.zeros_like(p) for p in params]
    def step(self, params, grads):
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.b2**self.t) / (1 - self.b1**self.t)
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m += (1 - self.b1) * (g - m); v += (1 - self.b2) * (g * g - v)
            p -= lr_t * m / (np.sqrt(v) + 1e-7)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    ap.add_argument("--folds", nargs="+", default=PERSONS)
    ap.add_argument("--out", default="results/mlp_15fold.json")
    a = ap.parse_args()

    data = load_all(); errs = {}
    for test in a.folds:
        np.random.seed(0)                                  # 재현성, fold마다 새 초기화
        tr = [p for p in PERSONS if p != test]
        Xtr = np.vstack([data[p][0] for p in tr]); Ytr = np.vstack([data[p][1] for p in tr])
        Xte, Yte = data[test]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6            # 학습용 14명으로만 계산
        ymu, ysd = Ytr.mean(0), Ytr.std(0)
        Xtr, Xte, Ytr_n = (Xtr - mu) / sd, (Xte - mu) / sd, (Ytr - ymu) / ysd

        net = MLP([Xtr.shape[1]] + a.hidden + [2], a.wd)
        params = net.W + net.b; opt = Adam(params, a.lr); t0 = time.time()
        for ep in range(a.epochs):
            idx = np.random.permutation(len(Xtr)); tot = 0.0
            for s in range(0, len(idx), a.batch):
                b = idx[s:s + a.batch]; pred = net.forward(Xtr[b])
                tot += 0.5 * np.mean((pred - Ytr_n[b]) ** 2) * len(b)
                gW, gb = net.backward(pred, Ytr_n[b]); opt.step(params, gW + gb)
            print(f"  [{test}] epoch {ep+1}/{a.epochs}  train loss {tot/len(Xtr):.4f}")
        pred = net.forward(Xte) * ysd + ymu                # 정규화 해제
        errs[test] = float(angular_error(pred, Yte).mean())
        print(f"{test}: {errs[test]:.2f}°  ({time.time()-t0:.0f}s)")

    mean = float(np.mean(list(errs.values())))
    print("평균", round(mean, 2))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"per_person": errs, "mean": mean, "args": vars(a)}, open(a.out, "w"), indent=2)

if __name__ == "__main__":
    main()