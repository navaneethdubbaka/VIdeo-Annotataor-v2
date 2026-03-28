"""
compositor.py — Multi-panel annotated frame builder
====================================================
Layout  (out_w × out_h):

  ┌─────────────────────────────┬──────┬─────────────────────────┐
  │  Camera  (undistorted)      │ info │  3-D hand skeleton      │
  │  + 2D hand overlay          │strip │  (top-right)            │
  │  + annotation card          │      ├─────────────────────────┤
  │  + NL caption               │      │  Full-body pose         │
  │  + metadata bar             │      │  (bottom-right)         │
  └─────────────────────────────┴──────┴─────────────────────────┘

New in this version vs old pipeline:
  • IMU horizon indicator on annotation card
  • World-frame RPY in info strip (not camera-relative)
  • IMU accel sparkline
  • Correct label for step boundary method

All drawing is pure OpenCV — no external dependencies.
"""

from __future__ import annotations
import math, textwrap

import cv2
import numpy as np

from kinematics import LM_FINGER, HAND_CONN, TIPS


# ── Colour palette ────────────────────────────────────────────────────────────

FCOL = {
    "palm":   (180,  80,  20),
    "thumb":  (  0, 140, 255),
    "index":  ( 30, 180,  30),
    "middle": (200,  60,  20),
    "ring":   ( 20,  20, 200),
    "pinky":  (170,  30, 170),
}
L_BGR  = ( 46, 204, 113)
R_BGR  = ( 52,  73, 235)
AX_X   = ( 40,  40, 220)
AX_Y   = ( 40, 180,  40)
AX_Z   = (220, 130,  30)
BONE3D = (165, 165, 165)
BODY_B = (180, 180, 180)
BODY_J = (120, 120, 120)
GRID_C = (210, 210, 210)
BORDER = (210, 210, 210)
BG_META= (248, 248, 248)
TXT_D  = ( 40,  40,  40)
TXT_G  = (130, 130, 130)
TXT_L  = (170, 170, 170)
FONT   = cv2.FONT_HERSHEY_SIMPLEX
FONT_B = cv2.FONT_HERSHEY_DUPLEX

# Body landmark indices
P_LS=11; P_RS=12; P_LE=13; P_RE=14; P_LW=15; P_RW=16

_SW=0.22; _TH=0.18; _UA=0.26; _FA=0.22; _HW=0.10; CUBE_H=0.16

TPOSE = {
    "nose": np.array([0,.52,0]),
    "ls":   np.array([-_SW,.30,0]),  "rs":  np.array([_SW,.30,0]),
    "le":   np.array([-_SW-_UA,.30,0]), "re": np.array([_SW+_UA,.30,0]),
    "lw":   np.array([-_SW-_UA-_FA,.30,0]), "rw": np.array([_SW+_UA+_FA,.30,0]),
    "lh":   np.array([-_HW,-_TH,0]), "rh": np.array([_HW,-_TH,0]),
}
SKEL_BONES = [
    ("nose","ls"),("nose","rs"),("ls","rs"),("ls","lh"),("rs","rh"),
    ("lh","rh"),("ls","le"),("le","lw"),("rs","re"),("re","rw"),
]
TRAIL_LEN = 30


# ── 3-D projection helpers ────────────────────────────────────────────────────

def _make_rot(elev=10, azim=-15):
    el, az = math.radians(elev), math.radians(azim)
    Ry = np.array([[math.cos(az),0,math.sin(az)],[0,1,0],
                   [-math.sin(az),0,math.cos(az)]])
    Rx = np.array([[1,0,0],[0,math.cos(el),-math.sin(el)],
                   [0,math.sin(el), math.cos(el)]])
    return (Rx @ Ry).astype(np.float64)

R3D = _make_rot()

def _p3(pt, R, cx, cy, sc):
    r = R @ np.asarray(pt, np.float64)
    return (int(r[0]*sc+cx), int(-r[1]*sc+cy))

def _zd(pt, R):
    return float((R @ np.asarray(pt, np.float64))[2])

def _cs(lms, half=CUBE_H, fill=0.60):
    c = lms - lms[0]
    e = np.abs(c).max()
    return c * (half*fill/e) if e > 1e-9 else c

def _sv(start, end, length):
    v = end - start
    n = np.linalg.norm(v)
    return (v/n*length) if n > 1e-6 else np.zeros(3)

def _local_frame(a, b):
    fwd = b - a
    fn  = np.linalg.norm(fwd)
    if fn < 1e-9: return np.eye(3)
    fwd /= fn
    up = np.array([0.,1.,0.])
    if abs(np.dot(fwd,up)) > 0.95: up = np.array([0.,0.,1.])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    up2   = np.cross(right, fwd)
    return np.array([right, up2, fwd])


# ── Grid / axes helpers ───────────────────────────────────────────────────────

def _dash_line(img, a, b, col=GRID_C, th=1, gap=6):
    a, b = np.array(a, float), np.array(b, float)
    d = b-a; L = np.linalg.norm(d)
    if L < 1: return
    n = max(2, int(L/gap))
    for s in range(n):
        if s%2==0:
            t0,t1 = s/n, min(1.,(s+1)/n)
            cv2.line(img, tuple((a+d*t0).astype(int)),
                         tuple((a+d*t1).astype(int)), col, th, cv2.LINE_AA)

def _draw_grid(img, R, cx, cy, sc, half, ndiv=4):
    h = half
    for i in range(ndiv+1):
        v = -h + i*2*h/ndiv
        _dash_line(img, _p3([v,-h,-h],R,cx,cy,sc), _p3([v,-h,h],R,cx,cy,sc))
        _dash_line(img, _p3([-h,-h,v],R,cx,cy,sc), _p3([h,-h,v],R,cx,cy,sc))
        _dash_line(img, _p3([v,-h,-h],R,cx,cy,sc), _p3([v,h,-h],R,cx,cy,sc))

def _draw_axes(img, R, cx, cy, sc, half):
    ext = half*1.15
    o = _p3([0,0,0],R,cx,cy,sc)
    for end3, col, lbl in [([ext,0,0],AX_X,"X"),([0,ext,0],AX_Y,"Y"),([0,0,ext],AX_Z,"Z")]:
        e = _p3(end3,R,cx,cy,sc)
        cv2.arrowedLine(img,o,e,col,1,cv2.LINE_AA,tipLength=0.15)
        d = np.array(e)-np.array(o); dn = np.linalg.norm(d)
        if dn>0:
            lp = np.array(e)+(d/dn*12).astype(int)
            cv2.putText(img,lbl,tuple(lp.astype(int)),FONT,0.35,col,1,cv2.LINE_AA)

def _draw_orient(img, origin3, frame3x3, R, cx, cy, sc, length=0.012):
    o = _p3(origin3,R,cx,cy,sc)
    for i,col in enumerate([AX_X,AX_Y,AX_Z]):
        e = _p3(origin3+frame3x3[i]*length,R,cx,cy,sc)
        cv2.line(img,o,e,col,1,cv2.LINE_AA)


# ── Utility ───────────────────────────────────────────────────────────────────

def _fmt_ts(s):
    m = int(s)//60
    return f"{m:02d}:{s-m*60:06.3f}"


# ── Panel: left (camera + annotation) ────────────────────────────────────────

def draw_left(frame, nlms, hness,
              W, H,
              macro_task, micro_step, step_idx, total_steps,
              t0, tc,
              nl_caption, env, scene, oph,
              imu_roll=0.0, imu_pitch=0.0, imu_yaw=0.0,
              step_method="equal_split"):

    CAP_H = 40; META_H = 52; VH = H - CAP_H - META_H
    canvas = np.full((H,W,3), 250, np.uint8)

    # Video fill-crop
    fh,fw = frame.shape[:2]
    sc = max(W/fw, VH/fh)
    nw,nh = int(fw*sc), int(fh*sc)
    res  = cv2.resize(frame,(nw,nh))
    x0c  = max(0,(nw-W)//2); y0c = max(0,(nh-VH)//2)
    crop = res[y0c:y0c+VH, x0c:x0c+W]
    canvas[:crop.shape[0],:crop.shape[1]] = crop

    # 2D hand overlay
    for lms, lab in zip(nlms, hness):
        pts = [(int(l[0]*W),int(l[1]*VH)) for l in lms]
        hc  = L_BGR if lab=="Left" else R_BGR
        for a,b in HAND_CONN:
            cv2.line(canvas,pts[a],pts[b],hc,2,cv2.LINE_AA)
        for i,pt in enumerate(pts):
            c = FCOL[LM_FINGER[i]]; r = 6 if i in TIPS else 4
            cv2.circle(canvas,pt,r+1,(255,255,255),-1,cv2.LINE_AA)
            cv2.circle(canvas,pt,r,c,-1,cv2.LINE_AA)

    # Annotation card
    if macro_task or micro_step:
        cw,ch = min(W-20,360), 100
        cx0,cy0 = 10,10
        ov = canvas.copy()
        cv2.rectangle(ov,(cx0,cy0),(cx0+cw,cy0+ch),(255,255,255),-1)
        cv2.addWeighted(ov,0.87,canvas,0.13,0,canvas)
        cv2.rectangle(canvas,(cx0,cy0),(cx0+cw,cy0+ch),(200,200,200),1)

        step_label = f"Step {step_idx}/{total_steps}: " if total_steps>1 else ""
        cv2.putText(canvas,"Action:",(cx0+8,cy0+17),FONT,0.33,TXT_G,1,cv2.LINE_AA)
        cv2.putText(canvas,(step_label+(micro_step or ""))[:55],
                    (cx0+58,cy0+17),FONT_B,0.42,(20,20,180),1,cv2.LINE_AA)

        if total_steps>1:
            bx,by = cx0+8,cy0+29
            bw = cw-16
            cv2.rectangle(canvas,(bx,by),(bx+bw,by+5),(220,220,220),-1)
            filled = int(bw*min(step_idx,total_steps)/total_steps)
            col_bar = (20,20,180) if step_method=="imu_detected" else (100,100,200)
            cv2.rectangle(canvas,(bx,by),(bx+filled,by+5),col_bar,-1)

        cv2.putText(canvas,"Goal:",(cx0+8,cy0+50),FONT,0.30,TXT_G,1,cv2.LINE_AA)
        cv2.putText(canvas,(macro_task or "")[:55],
                    (cx0+50,cy0+50),FONT,0.37,TXT_D,1,cv2.LINE_AA)
        cv2.putText(canvas,f"{_fmt_ts(t0)}  →  {_fmt_ts(tc)}",
                    (cx0+8,cy0+68),FONT,0.30,TXT_G,1,cv2.LINE_AA)

        # IMU horizon indicator
        _draw_horizon(canvas, cx0+cw-70, cy0+50, 28, imu_roll, imu_pitch)

    # NL caption
    cy = VH
    cv2.rectangle(canvas,(0,cy),(W,cy+CAP_H),(238,238,238),-1)
    cv2.line(canvas,(0,cy),(W,cy),BORDER,1)
    for i,ln in enumerate((textwrap.wrap(nl_caption or "",width=W//8) or [""])[:2]):
        cv2.putText(canvas,ln,(12,cy+16+i*16),FONT,0.42,TXT_D,1,cv2.LINE_AA)

    # Metadata bar
    my = VH+CAP_H
    cv2.rectangle(canvas,(0,my),(W,H),BG_META,-1)
    cv2.line(canvas,(0,my),(W,my),BORDER,1)
    cw3 = W//3
    for ci,(lb,vl) in enumerate([
            ("ENVIRONMENT", env or "—"),
            ("SCENE",       scene or "—"),
            ("OP HEIGHT",   f"{oph:.0f}cm" if isinstance(oph,float) else str(oph))]):
        lx = ci*cw3+14
        cv2.circle(canvas,(ci*cw3+8,my+18),3,TXT_G,-1,cv2.LINE_AA)
        cv2.putText(canvas,lb,(lx,my+20),FONT,0.28,TXT_G,1,cv2.LINE_AA)
        cv2.putText(canvas,vl,(lx,my+38),FONT_B,0.40,TXT_D,1,cv2.LINE_AA)
        if ci>0:
            cv2.line(canvas,(ci*cw3,my+5),(ci*cw3,H-5),BORDER,1)

    return canvas


def _draw_horizon(img, cx, cy, r, roll_deg, pitch_deg):
    """Small artificial horizon indicator showing camera tilt."""
    cv2.circle(img,(cx,cy),r,(220,220,220),1,cv2.LINE_AA)
    angle = math.radians(roll_deg)
    px = int(cx + r * 0.8 * math.cos(angle + math.pi))
    py = int(cy - r * 0.8 * math.sin(angle + math.pi))
    qx = int(cx + r * 0.8 * math.cos(angle))
    qy = int(cy - r * 0.8 * math.sin(angle))
    cv2.line(img,(px,py),(qx,qy),(46,180,113),2,cv2.LINE_AA)
    # Pitch dot
    pdx = int(cx + (pitch_deg/90.0)*r*0.5 * math.cos(angle+math.pi/2))
    pdy = int(cy - (pitch_deg/90.0)*r*0.5 * math.sin(angle+math.pi/2))
    cv2.circle(img,(pdx,pdy),3,(46,180,113),-1,cv2.LINE_AA)


# ── Panel: info strip ─────────────────────────────────────────────────────────

def draw_info_panel(W, H, rpy_data, grasp_data, joint_data,
                    imu_accel_history: list[float] | None = None):
    """
    rpy_data   : {label: (roll, pitch, yaw)}  — world-frame
    grasp_data : {label: (type, aperture, contact)}
    joint_data : {label: {joint: angle_deg}}
    imu_accel_history : recent accel magnitudes for sparkline
    """
    canvas = np.full((H,W,3),(248,248,248),np.uint8)
    PAD=6; y=10

    cv2.putText(canvas,"WORLD POSE / GRASP",(PAD,y),FONT,0.28,TXT_G,1,cv2.LINE_AA)
    y+=3; cv2.line(canvas,(PAD,y),(W-PAD,y),BORDER,1); y+=8

    for side, hcol in [("Left",L_BGR),("Right",R_BGR)]:
        cv2.rectangle(canvas,(PAD,y-1),(PAD+4,y+9),hcol,-1)
        cv2.putText(canvas,side,(PAD+8,y+8),FONT_B,0.35,hcol,1,cv2.LINE_AA)
        y+=14

        rpy = rpy_data.get(side)
        if rpy:
            r_,p_,y_ = rpy
            for lbl,val in [("R",r_),("P",p_),("Y",y_)]:
                bar_w = W-PAD*2
                filled = int(bar_w*min(abs(val)/180.,1.))
                bar_col = AX_X if lbl=="R" else (AX_Y if lbl=="P" else AX_Z)
                cv2.rectangle(canvas,(PAD,y+1),(PAD+bar_w,y+7),(225,225,225),-1)
                cv2.rectangle(canvas,(PAD,y+1),(PAD+filled,y+7),bar_col,-1)
                cv2.putText(canvas,f"{lbl} {val:+.0f}",(PAD,y+16),
                            FONT,0.28,bar_col,1,cv2.LINE_AA)
                y+=18
        else:
            cv2.putText(canvas,"no data",(PAD,y+8),FONT,0.27,TXT_L,1,cv2.LINE_AA)
            y+=14

        if grasp_data and side in grasp_data:
            gt,ap,cs_ = grasp_data[side]
            cv2.putText(canvas,gt,(PAD,y+9),FONT_B,0.34,TXT_D,1,cv2.LINE_AA); y+=12
            cv2.putText(canvas,f"ap {ap*100:.1f}cm  {cs_}",
                        (PAD,y+9),FONT,0.27,TXT_G,1,cv2.LINE_AA); y+=13

        if joint_data and side in joint_data:
            fja = joint_data[side]
            cv2.putText(canvas,"joints°",(PAD,y+8),FONT,0.25,TXT_G,1,cv2.LINE_AA); y+=11
            fingers = [
                ("Th",["thumb_mcp","thumb_ip"]),
                ("Ix",["idx_mcp","idx_pip","idx_dip"]),
                ("Md",["mid_mcp","mid_pip","mid_dip"]),
                ("Rg",["ring_mcp","ring_pip","ring_dip"]),
                ("Pk",["pinky_mcp","pinky_pip","pinky_dip"]),
            ]
            bar_full = W-PAD*2
            for abbr,keys in fingers:
                if y >= H-14: break
                cv2.putText(canvas,abbr,(PAD,y+8),FONT_B,0.28,TXT_D,1,cv2.LINE_AA)
                bx = PAD+16; slot_w = (bar_full-16)//len(keys)
                for k in keys:
                    ang = fja.get(k,0.)
                    blen = int(slot_w*min(ang/180.,1.))
                    cv2.rectangle(canvas,(bx,y+1),(bx+slot_w-2,y+6),(220,220,220),-1)
                    cv2.rectangle(canvas,(bx,y+1),(bx+blen,y+6),hcol,-1)
                    cv2.putText(canvas,f"{ang:.0f}",(bx,y+14),FONT,0.22,TXT_G,1,cv2.LINE_AA)
                    bx+=slot_w
                y+=16

        y+=4; cv2.line(canvas,(PAD,y),(W-PAD,y),BORDER,1); y+=8

    # IMU accel sparkline
    if imu_accel_history and len(imu_accel_history)>2 and y<H-20:
        cv2.putText(canvas,"IMU |a|",(PAD,y+8),FONT,0.25,TXT_G,1,cv2.LINE_AA)
        y+=12
        sh = min(20, H-y-4)
        if sh>4:
            hist = np.array(imu_accel_history[-W:])
            mn,mx = hist.min(), hist.max()+1e-9
            pts_x = np.linspace(PAD, W-PAD, len(hist)).astype(int)
            pts_y = (y+sh - (hist-mn)/(mx-mn)*sh).astype(int)
            for i in range(1,len(pts_x)):
                cv2.line(canvas,(pts_x[i-1],pts_y[i-1]),(pts_x[i],pts_y[i]),
                         AX_Z,1,cv2.LINE_AA)
            y+=sh+4

    return canvas


# ── Panel: 3D hand ────────────────────────────────────────────────────────────

def draw_hand_panel(W, H, current_hands, rpy_data, cube_half=0.12):
    canvas = np.full((H,W,3),255,np.uint8)
    R = R3D
    pad_l=int(W*0.15); pad_b=int(H*0.15)
    CX=pad_l+(W-pad_l)//2; CY=(H-pad_b)//2+int(H*0.12)
    SC=min(W-pad_l,H-pad_b)/(cube_half*1.8)
    _draw_grid(canvas,R,CX,CY,SC,cube_half)
    _draw_axes(canvas,R,CX,CY,SC,cube_half)
    offsets={"Left":np.array([-0.04,0,0]),"Right":np.array([0.04,0,0])}
    for lms_sc,label in current_hands:
        col=L_BGR if label=="Left" else R_BGR
        off=offsets.get(label,np.zeros(3)); lm=lms_sc+off
        bone_z=sorted([((_zd((lm[a]+lm[b])/2,R)),a,b) for a,b in HAND_CONN])
        for _,a,b in bone_z:
            cv2.line(canvas,_p3(lm[a],R,CX,CY,SC),_p3(lm[b],R,CX,CY,SC),BONE3D,1,cv2.LINE_AA)
        for _,i in sorted((_zd(lm[i],R),i) for i in range(21)):
            pt=_p3(lm[i],R,CX,CY,SC); c=FCOL[LM_FINGER[i]]
            r=7 if i==0 else (6 if i in TIPS else 4)
            cv2.circle(canvas,pt,r,c,-1,cv2.LINE_AA)
        segs=[(0,1),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),
              (9,10),(10,11),(11,12),(13,14),(14,15),(15,16),(17,18),(18,19),(19,20)]
        for a,b in segs:
            mid=(lm[a]+lm[b])/2; frm=_local_frame(lm[a],lm[b])
            _draw_orient(canvas,mid,frm,R,CX,CY,SC,length=0.012)
    ly=12
    for lbl,col in [("Left",L_BGR),("Right",R_BGR)]:
        cv2.circle(canvas,(W-70,ly),4,col,-1,cv2.LINE_AA)
        cv2.putText(canvas,lbl,(W-62,ly+4),FONT,0.32,TXT_D,1,cv2.LINE_AA); ly+=14
    for lbl,col in [("X",AX_X),("Y",AX_Y),("Z",AX_Z)]:
        cv2.line(canvas,(W-70,ly),(W-60,ly),col,2,cv2.LINE_AA)
        cv2.putText(canvas,lbl,(W-56,ly+4),FONT,0.28,col,1,cv2.LINE_AA); ly+=12
    return canvas


# ── Panel: body ───────────────────────────────────────────────────────────────

def draw_body_panel(W, H, pose_arr, hands_at_wrist):
    canvas = np.full((H,W,3),255,np.uint8)
    R = R3D
    sk = {k:v.copy() for k,v in TPOSE.items()}
    if pose_arr is not None:
        ls,rs = sk["ls"],sk["rs"]
        sk["le"] = ls+_sv(pose_arr[P_LS],pose_arr[P_LE],_UA)
        sk["re"] = rs+_sv(pose_arr[P_RS],pose_arr[P_RE],_UA)
        sk["lw"] = sk["le"]+_sv(pose_arr[P_LE],pose_arr[P_LW],_FA)
        sk["rw"] = sk["re"]+_sv(pose_arr[P_RE],pose_arr[P_RW],_FA)
    all_pts = np.array(list(sk.values()))
    cx3,cy3 = all_pts[:,0].mean(),all_pts[:,1].mean()
    span = max(np.ptp(all_pts[:,0]),np.ptp(all_pts[:,1]))*0.7+0.15
    pad_l,pad_b = int(W*0.14),int(H*0.14)
    CX=pad_l+(W-pad_l)//2; CY=(H-pad_b)//2
    SC=min(W-pad_l,H-pad_b)/(span*2.1)
    offset=np.array([cx3,cy3,0.])
    sk_c={k:v-offset for k,v in sk.items()}
    _draw_grid(canvas,R,CX,CY,SC,span)
    _draw_axes(canvas,R,CX,CY,SC,span)
    for a_nm,b_nm in SKEL_BONES:
        cv2.line(canvas,_p3(sk_c[a_nm],R,CX,CY,SC),
                 _p3(sk_c[b_nm],R,CX,CY,SC),BODY_B,2,cv2.LINE_AA)
    for a_nm,b_nm in [("ls","le"),("le","lw"),("rs","re"),("re","rw")]:
        mid=(sk_c[a_nm]+sk_c[b_nm])/2; frm=_local_frame(sk_c[a_nm],sk_c[b_nm])
        _draw_orient(canvas,mid,frm,R,CX,CY,SC,length=0.04)
    for nm,pt in sk_c.items():
        sz=8 if nm=="nose" else 5
        cv2.circle(canvas,_p3(pt,R,CX,CY,SC),sz,BODY_J,-1,cv2.LINE_AA)
        cv2.circle(canvas,_p3(pt,R,CX,CY,SC),sz,(255,255,255),1,cv2.LINE_AA)
    hand_sc=(_FA*0.80)/CUBE_H
    wrist_map={"Left":sk_c["lw"],"Right":sk_c["rw"]}
    hcol={"Left":L_BGR,"Right":R_BGR}
    for side,lms_sc in hands_at_wrist.items():
        w3=wrist_map[side]; col=hcol[side]
        hpts=np.column_stack([lms_sc[:,0]*hand_sc+w3[0],
                               lms_sc[:,1]*hand_sc+w3[1],
                               lms_sc[:,2]*hand_sc+w3[2]])
        bone_z=sorted([((_zd((hpts[a]+hpts[b])/2,R)),a,b) for a,b in HAND_CONN])
        for _,a,b in bone_z:
            cv2.line(canvas,_p3(hpts[a],R,CX,CY,SC),_p3(hpts[b],R,CX,CY,SC),col,1,cv2.LINE_AA)
        for _,i in sorted((_zd(hpts[i],R),i) for i in range(21)):
            pt=_p3(hpts[i],R,CX,CY,SC)
            r=6 if i in TIPS else (7 if i==0 else 4)
            cv2.circle(canvas,pt,r,col,-1,cv2.LINE_AA)
            cv2.circle(canvas,pt,r,(255,255,255),1,cv2.LINE_AA)
    ly=12
    for lbl,col in [("Body",BODY_B),("L hand",L_BGR),("R hand",R_BGR)]:
        cv2.rectangle(canvas,(6,ly-6),(18,ly+2),col,-1)
        cv2.putText(canvas,lbl,(22,ly+2),FONT,0.30,TXT_D,1,cv2.LINE_AA); ly+=14
    return canvas


# ── Compositor ────────────────────────────────────────────────────────────────

def build_frame(frame, nlms, hness,
                cur_hands, rpy_data, grasp_data, joint_data,
                pose_arr, haw,
                out_w, out_h,
                macro_task, micro_step, step_idx, total_steps,
                t0, tc, nl_caption, env, scene, oph,
                imu_roll=0., imu_pitch=0., imu_yaw=0.,
                imu_accel_history=None,
                step_method="equal_split") -> np.ndarray:

    INFO_W = 190
    cam_w  = int(out_w * 0.42)
    plot_w = out_w - cam_w - INFO_W - 2
    plot_h = out_h // 2

    left = draw_left(frame, nlms, hness, cam_w, out_h,
                     macro_task, micro_step, step_idx, total_steps,
                     t0, tc, nl_caption, env, scene, oph,
                     imu_roll, imu_pitch, imu_yaw, step_method)
    info = draw_info_panel(INFO_W, out_h, rpy_data, grasp_data,
                           joint_data, imu_accel_history)
    top_r = draw_hand_panel(plot_w, plot_h, cur_hands, rpy_data)
    bot_r = draw_body_panel(plot_w, plot_h, pose_arr, haw)

    right = np.vstack([top_r, bot_r])
    cv2.line(right,(0,plot_h),(plot_w,plot_h),BORDER,1)
    div = np.full((out_h,1,3),BORDER,np.uint8)

    card = np.hstack([left, div, info, div, right])
    card = cv2.resize(card, (out_w, out_h))
    cv2.rectangle(card,(0,0),(card.shape[1]-1,card.shape[0]-1),BORDER,2)
    return card
