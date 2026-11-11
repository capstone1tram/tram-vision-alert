"""04. 거리 추정 (카메라 한 대 기준)

핵심 아이디어: 실제 크기를 아는 물건(레일 간격 1.435 m, 사람 키 1.7 m)이 화면에서
몇 픽셀로 보이는지 재면 거리를 알 수 있다.   거리 = 초점거리(px) x 실제크기 / 픽셀크기

- 앞 거리 Z      : 물체 발 위치 줄의 궤간 픽셀 폭 g 로  Z = f * 1.435 / g
- 옆 거리 d_lat  : (물체 x - 기준선 x) * 1.435 / g
- 백업          : 레일이 가려졌으면 사람 박스 높이 h 로  Z = f * 1.7 / h
- 가까운 곳      : OAK-D 깊이값이 있으면 그것을 우선 사용 (약 8 m 이내)

실행:  python 04_distance.py      -> 거리별 픽셀 크기와 1 px 오차 표 출력
"""
import math

import config as C


def focal_px(width=C.IMG_W, hfov_deg=C.HFOV_DEG):
    """초점거리를 픽셀 단위로. 1920px·60도면 약 1663px."""
    return (width / 2) / math.tan(math.radians(hfov_deg) / 2)


def z_from_gauge(gauge_px, f=None):
    f = f or focal_px()
    return f * C.GAUGE_M / gauge_px


def lateral_from_gauge(x_obj, x_center, gauge_px):
    """+면 기준선 오른쪽, -면 왼쪽 (m)."""
    return (x_obj - x_center) * C.GAUGE_M / gauge_px


def z_from_height(box_h_px, real_h=C.PERSON_HEIGHT_M, f=None):
    f = f or focal_px()
    return f * real_h / box_h_px


def fuse_distance(z_depth=None, z_gauge=None, z_height=None, depth_max=8.0):
    """쓸 수 있는 값 중 믿을 만한 순서로 고름: 깊이(가까울 때) > 궤간 > 사람 키."""
    if z_depth is not None and 0.3 < z_depth <= depth_max:
        return z_depth, "depth"
    if z_gauge is not None:
        return z_gauge, "gauge"
    if z_height is not None:
        return z_height, "height"
    return None, "none"


def obj_distance(box, rails, z_depth=None):
    """박스(x1,y1,x2,y2)와 RailModel(03번)로 (Z, d_lat, 방법) 계산."""
    x1, y1, x2, y2 = box
    foot_x, foot_y = (x1 + x2) / 2, y2            # 발 위치 = 박스 밑변 가운데
    z_g = d_lat = None
    if rails is not None and foot_y >= rails.y_top:
        g = rails.gauge_px(foot_y)
        if g > 2:
            z_g = z_from_gauge(g)
            d_lat = lateral_from_gauge(foot_x, rails.center_x(foot_y), g)
    z_h = z_from_height(y2 - y1) if y2 - y1 > 4 else None
    z, method = fuse_distance(z_depth, z_g, z_h)
    return z, d_lat, method


if __name__ == "__main__":
    for w in (1280, 1920):
        f = focal_px(w)
        print(f"\n화면 너비 {w}px, 화각 {C.HFOV_DEG:.0f}도 -> f = {f:.0f}px")
        print(" 거리(m) | 궤간(px) | 1px 오차(m) | 사람 키(px)")
        for Z in (5, 10, 20, 50, 100):
            g = f * C.GAUGE_M / Z
            err = z_from_gauge(g - 1, f) - Z
            print(f" {Z:7.0f} | {g:8.1f} | {err:10.2f} | {f*C.PERSON_HEIGHT_M/Z:9.1f}")
