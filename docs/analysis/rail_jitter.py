import sys, glob, csv, cv2, numpy as np
sys.path.insert(0, '/home/claude/work/tram_system')
from tram.config import cfg
cfg.load_json('/home/claude/work/tram_system/settings_video.json'); cfg.IMG_W, cfg.IMG_H = 960, 540
from tram.vision import RailTracker
from tram import ground
out = csv.writer(open('/home/claude/work/analysis/rail_jitter.csv', 'w', newline=''))
out.writerow(['clip','t','status','y_top','y_bot','cx90','g90','a','b','y_h','cam_h','y10','y20','y50','z_top'])
for c in range(4):
    f = glob.glob(f'/mnt/user-data/uploads/tram_/logs/probe/clips/clip{c}_*.mp4')[0]
    cap = cv2.VideoCapture(f); fps = cap.get(5); rt = RailTracker(); i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        m, _ = rt.update(fr); t = i / fps; i += 1
        if m is None:
            out.writerow([c, f'{t:.3f}', rt.status] + [''] * 12); continue
        gm = ground.ground_of(m); y = 0.9 * 540
        row = [c, f'{t:.3f}', rt.status, m.y_top, m.y_bottom, f'{m.center_x(y):.1f}', f'{m.gauge_px(y):.1f}']
        if gm:
            zt = cfg.focal_px * cfg.GAUGE_M / max(m.gauge_px(m.y_top), 1)
            row += [f'{gm.a:.4f}', f'{gm.b:.2f}', f'{gm.y_h:.1f}', f'{gm.camera_height():.3f}'] + [f'{gm.y_of_z(z):.1f}' for z in (10, 20, 50)] + [f'{zt:.1f}']
        else:
            row += [''] * 8
        out.writerow(row)
