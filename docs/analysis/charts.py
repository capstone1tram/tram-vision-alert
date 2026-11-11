import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for p in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc']:
    fm.fontManager.addfont(p)
plt.rcParams.update({'font.family': 'Noto Sans CJK KR', 'font.size': 9, 'axes.edgecolor': '#c9c8c3',
                     'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 200})
S1, S2 = '#2a78d6', '#eb6834'   # 1 = 개선안(고정), 2 = 지금(프레임마다)
def chart(c, col, scol, ylabel, title, path, invert=True, ylim=None):
    g = pd.read_csv(f'smooth_c{c}.csv')
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.plot(g.t, g[col], color=S2, lw=1.2, label='지금 (프레임마다 새로 계산)')
    ax.plot(g.t, g[scol], color=S1, lw=2.0, label='개선안 (최근 값 중앙값으로 고정)')
    ax.grid(axis='y', color='#e6e5e0', lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel('영상 시각 (초)'); ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    if invert and not ylim: ax.invert_yaxis()
    ax.legend(frameon=False, loc='lower left', fontsize=8, ncol=2, bbox_to_anchor=(0, 1.0))
    fig.tight_layout(); fig.savefig(path, facecolor='white'); plt.close(fig)
O = '/home/claude/work/pdf/img/'
chart(1, 'y50', 'y50_s', '화면 줄 (px, 위 = 멀리)', '50 m 눈금이 그려지는 위치 (12분 19초 구간)', O + 'chart_y50.png', ylim=(400, 200))
chart(3, 'y_top', 'ytop_s', '화면 줄 (px, 위 = 멀리)', '위험 구역의 먼 쪽 끝 (29분 55초 구간)', O + 'chart_ytop.png', ylim=(520, 280))
chart(1, 'cam_h', 'h_s', '추정 카메라 높이 (m)', '궤간 폭 기울기로 구한 카메라 높이 (원래는 일정해야 함)', O + 'chart_camh.png', invert=False, ylim=(0.8, 3.0))
print('ok')
