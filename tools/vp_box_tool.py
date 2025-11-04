#!/usr/bin/env python3
import sys, json, os, cv2

USAGE = "Usage: python3 vp_box_tool.py /path/to/image.jpg [class_id (default 0)]"

def main():
    if len(sys.argv) < 2:
        print(USAGE); sys.exit(1)
    img_path = sys.argv[1]
    default_cls = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}"); sys.exit(1)

    img = cv2.imread(img_path)
    if img is None:
        print("OpenCV failed to read the image."); sys.exit(1)
    h, w = img.shape[:2]
    print(f"Loaded {img_path} ({w}x{h}). Draw one or more boxes.")
    print("Instructions: click-drag to draw a box; ENTER to accept; ESC/ENTER again to finish.")

    rois = cv2.selectROIs("Draw boxes", img, showCrosshair=True, fromCenter=False)  # (N,4) as x,y,w,h
    cv2.destroyAllWindows()
    if rois is None or len(rois) == 0:
        print("No boxes selected."); sys.exit(2)

    bboxes = []
    cls = []
    last = default_cls
    for i,(x,y,w_,h_) in enumerate(rois):
        x1,y1,x2,y2 = int(x), int(y), int(x+w_), int(y+h_)
        # clamp just in case
        x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
        if x2 <= x1 or y2 <= y1:
            print(f"Skipping degenerate box #{i}: {(x1,y1,x2,y2)}")
            continue
        bboxes.append([x1,y1,x2,y2])
        try:
            c = input(f"Class id for box #{i} (default {last}): ").strip()
            c = int(c) if c != "" else last
        except:
            c = last
        cls.append(c); last = c

    # Ready-to-paste YAML strings for your launch command
    bboxes_yaml = json.dumps(bboxes)
    cls_yaml = json.dumps(cls)

    print("\nCopy these into your launch command:")
    print(f"vp_bboxes:='{bboxes_yaml}'")
    print(f"vp_cls:='{cls_yaml}'")

    # (Optional) write to a small file next to the image
    out_txt = os.path.splitext(img_path)[0] + "_vp.txt"
    with open(out_txt, "w") as f:
        f.write(f"vp_bboxes:='{bboxes_yaml}'\nvp_cls:='{cls_yaml}'\n")
    print(f"Wrote {out_txt}")

if __name__ == "__main__":
    main()
