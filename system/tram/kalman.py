"""칼만 필터 추적기 (라즈베리 파이에서 돌리는 부분, 예제 07 정리판)

- 선형 칼만 필터 + 등속도 모델.  상태 = 바닥 평면 미터 좌표 [X, Z, Vx, Vz]
  (X: 선로 중심에서 옆 거리, Z: 앞 거리, V: 트램 기준 상대 속도)
- 측정은 이미 미터로 바꾼 (X, Z)라서 측정식이 선형(H = [I 0]) -> EKF/UKF 필요 없음
- 멀리 있을수록 궤간 픽셀이 작아 Z 오차가 Z²에 비례 -> 측정 잡음 R을 거리에 따라 바꿈
- 측정 시각(t)을 상태와 함께 들고 있어서, 노트북 답이 늦게 와도 '찍은 시각' 기준으로 갱신하고
  판단은 '지금 시각'까지 예측한 값으로 함 (지연 보정)
numpy만 씀. 물체 하나당 1 ms 미만.
"""
import copy

import numpy as np

from .config import cfg


def meas_sigma(z, name="person"):
    """궤간 기반 거리의 1σ 오차.  Z = f·G/w  ->  dZ = Z²/(f·G)·dw
    차량은 박스 밑변(바퀴)이 사람 발보다 덜 또렷해서 픽셀 오차를 더 크게 봄."""
    z = max(z, cfg.GAUGE_M / 1.435)                 # 1 m (모형이면 0.1 m) 아래는 같은 값
    f = cfg.focal_px
    px = cfg.KF_PX_ERR * (1.0 if name == "person" else cfg.KF_VEHICLE_PX_MULT)
    sig_z = z * z / (f * cfg.GAUGE_M) * px
    sig_x = 0.035 * cfg.GAUGE_M + z / f * px * 2     # 옆 거리: 박스 중심 흔들림
    return sig_x, sig_z


class GroundKF:
    """물체 하나.  x = [X, Z, Vx, Vz]"""

    def __init__(self, X, Z, t, name="person"):
        self.x = np.array([X, Z, 0.0, 0.0])
        sx, sz = meas_sigma(Z, name)
        v0 = cfg.GAUGE_M / 1.435                       # 모형이면 속도 단위도 줄임
        self.P = np.diag([sx ** 2, sz ** 2, (2.0 * v0) ** 2, (8.0 * v0) ** 2])  # 속도는 처음엔 모름
        self.t = t
        self.H = np.hstack([np.eye(2), np.zeros((2, 2))])
        self.name = name
        self.hits = 1
        self.t_last_meas = t
        self.meas_k = None                             # 마지막으로 이 트랙에 붙은 측정 번호 (화면에 박스 짝 표시용)

    def predict(self, t):
        dt = t - self.t
        if dt <= 0:
            return self.x
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        g = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + g @ g.T * cfg.KF_ACC_SIGMA ** 2
        self.t = t
        return self.x

    def _S_y(self, X, Z):
        sx, sz = meas_sigma(Z, self.name)
        S = self.H @ self.P @ self.H.T + np.diag([sx ** 2, sz ** 2])
        y = np.array([X, Z]) - self.H @ self.x
        return S, y

    def gate(self, X, Z):
        """마할라노비스 거리². 9.21 이하면 99% 안 = 이 트랙의 측정으로 봄."""
        S, y = self._S_y(X, Z)
        return float(y @ np.linalg.solve(S, y))

    def update(self, X, Z, t, name=None):
        S, y = self._S_y(X, Z)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.hits += 1
        self.t_last_meas = t
        if name:
            self.name = name

    def reanchor(self, X, Z, t, name=None):
        """검출기가 바뀐 직후(노트북 <-> 파이) 첫 측정: 박스 모양이 검출기마다 조금씩 달라서
        그대로 갱신하면 '가짜 속도'가 생김. 위치만 새 값으로 옮기고 속도는 그대로 둠."""
        sx, sz = meas_sigma(Z, self.name)
        self.x[0], self.x[1] = X, Z
        self.P[0:2, :] = 0.0
        self.P[:, 0:2] = 0.0
        self.P[0, 0], self.P[1, 1] = sx ** 2, sz ** 2
        self.hits += 1
        self.t_last_meas = t
        if name:
            self.name = name

    def state(self):
        X, Z, vx, vz = self.x
        return {"X": float(X), "Z": float(Z), "vx": float(vx), "v_close": float(-vz),
                "sigma_X": float(np.sqrt(self.P[0, 0])), "sigma_Z": float(np.sqrt(self.P[1, 1])),
                "sigma_vx": float(np.sqrt(self.P[2, 2])),
                "name": self.name, "hits": self.hits}


# 검출기가 종류를 헷갈리기 쉬운 것끼리 묶음. 같은 묶음끼리만 같은 물체로 이어 붙임
GROUP = {"person": 0, "bicycle": 1, "motorcycle": 1, "car": 2, "bus": 2, "truck": 2}


class Tracker:
    """여러 물체.  게이트 안 짝 중 가장 가까운 것부터 묶고(greedy), 남은 측정은 새 트랙."""

    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.t = None                                  # 마지막으로 측정을 넣은 시각
        self.source = None                             # 마지막 측정을 만든 검출기 ("laptop" / "pi")

    def update(self, t, meas, source=None):
        """meas: [(X, Z, 이름), ...] 를 시각 t에 찍힌 측정으로 반영."""
        if self.t is not None and t < self.t:
            return False                               # 이미 더 최신 측정이 들어감 (늦게 온 답)
        self.t = t
        switched = source is not None and self.source is not None and source != self.source
        self.source = source
        for tr in self.tracks.values():
            tr.predict(t)
        pairs = []
        for tid, tr in self.tracks.items():
            for k, (X, Z, name) in enumerate(meas):
                if GROUP.get(name, -1) != GROUP.get(tr.name, -1):
                    continue
                d2 = tr.gate(X, Z)
                if d2 < cfg.KF_GATE:
                    # 확정된(오래 본) 트랙이 먼저 고름 -> 새로 생긴 트랙이 측정을 가로채지 않게
                    pairs.append((tr.hits < cfg.CONFIRM_HITS, d2, tid, k))
        used_t, used_m = set(), set()
        for _, d2, tid, k in sorted(pairs):
            if tid in used_t or k in used_m:
                continue
            X, Z, name = meas[k]
            if switched:
                self.tracks[tid].reanchor(X, Z, t, name)
            else:
                self.tracks[tid].update(X, Z, t, name)
            self.tracks[tid].meas_k = k
            used_t.add(tid)
            used_m.add(k)
        for k, (X, Z, name) in enumerate(meas):
            if k not in used_m:
                self.tracks[self.next_id] = GroundKF(X, Z, t, name)
                self.tracks[self.next_id].meas_k = k
                self.next_id += 1
        self.prune(t)
        return True

    def prune(self, t):
        for tid in list(self.tracks):
            tr = self.tracks[tid]
            limit = cfg.TRACK_MAX_COAST_S if tr.hits >= cfg.CONFIRM_HITS else cfg.TRACK_TENTATIVE_COAST_S
            if t - tr.t_last_meas > limit:
                del self.tracks[tid]

    def states_at(self, t):
        """지금 시각 t까지 예측한 상태 (원본 트랙은 그대로 둠)."""
        out = {}
        for tid, tr in self.tracks.items():
            if t - tr.t_last_meas > cfg.TRACK_MAX_COAST_S:
                continue
            c = copy.deepcopy(tr)
            c.predict(t)
            s = c.state()
            s["age"] = t - tr.t_last_meas              # 마지막 측정 이후 시간
            s["meas_k"] = tr.meas_k if tr.t_last_meas == self.t else None   # 이번 측정 중 몇 번째와 짝인지
            out[tid] = s
        return out
