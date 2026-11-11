"""거리 추정 (카메라 한 대, 깊이 센서 없음)

핵심: 실제 크기를 아는 것(궤간 1.435 m)이 화면에서 몇 픽셀로 보이는지 재면 거리를 안다.
   거리 = 초점거리(px) x 실제 크기 / 픽셀 크기

- 앞 거리 Z : 물체 발 위치 줄의 궤간 픽셀 폭 g 로  Z = f * 궤간 / g
- 옆 거리 X : (물체 x - 중심선 x) * 궤간 / g      (+면 오른쪽)
- 백업      : 발 위치가 선로 검출 범위 밖이면 사람 키로  Z = f * 1.7 / 박스 높이
              (옆 거리는 이때 중심선을 연장해서 계산)
"""
from .config import cfg


def z_from_gauge(gauge_px):
    return cfg.focal_px * cfg.GAUGE_M / gauge_px


def z_from_height(box_h_px, real_h=None):
    return cfg.focal_px * (real_h or cfg.PERSON_HEIGHT_M) / box_h_px


def obj_distance(box, rails, name="person"):
    """박스(x1,y1,x2,y2) + RailModel -> (Z, X, 방법). 계산 못 하면 (None, None, 'none')."""
    if rails is None:
        return None, None, "none"
    x1, y1, x2, y2 = box
    foot_x, foot_y = (x1 + x2) / 2, y2              # 발 위치 = 박스 밑변 가운데
    # 선로를 실제로 찾은 범위 안(조금 위 여유 포함)이면 궤간 방식
    if foot_y >= rails.y_top - 0.15 * (rails.y_bottom - rails.y_top):
        g = rails.gauge_px(foot_y)
        if g > 2:
            return z_from_gauge(g), (foot_x - rails.center_x(foot_y)) * cfg.GAUGE_M / g, "gauge"
    # 더 멀리 있으면 사람 키 백업 (사람만; 차량은 크기가 제각각이라 안 씀)
    if name == "person" and y2 - y1 > 8:
        z = z_from_height(y2 - y1)
        # 옆 거리: 그 거리에서의 1 px 크기 = Z / f
        y_ref = min(max(foot_y, rails.y_top), rails.y_bottom)
        return z, (foot_x - rails.center_x(y_ref)) * z / cfg.focal_px, "height"
    return None, None, "none"
