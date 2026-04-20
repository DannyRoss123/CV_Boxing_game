"""
Boxing Training Hub — Pygame

Mini-game hub with 4 games:
  Memory      — Simon Says with punches
  Speed Bag   — most punches in 30 seconds
  Reaction    — show a punch, clock how fast you throw it
  Power Shot  — hit a target power zone on the velocity bar

Controls:
  Arrow keys  — navigate hub
  SPACE/ENTER — launch game
  ESC         — back to hub / quit
"""

import collections
import json
import math
import os
import random
import sys
import threading
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pygame

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH  = os.path.join(ROOT, "pose_landmarker_lite.task")
RF_PATH     = os.path.join(ROOT, "models", "random_forest.joblib")
SCALER_PATH = os.path.join(ROOT, "models", "rf_scaler.joblib")
SCORES_PATH = os.path.join(ROOT, "data", "boxing_hub_scores.json")

# ── CV constants ───────────────────────────────────────────────────────────────
SEQ_LEN = 60;  N_LM = 33;  F_PER_LM = 4
L_HIP=23*4; R_HIP=24*4; L_SHO=11*4; R_SHO=12*4
RWX=16*4; RWY=16*4+1; LWX=15*4; LWY=15*4+1
REX=14*4; REY=14*4+1; LEX=13*4; LEY=13*4+1
BUF_SIZE=120; VEL_THRESH=0.013; MOTION_START=3
MOTION_STOP=8; LEAD_IN=18; TAIL=12; COOLDOWN=0.6; MIN_CONF=0.30

# ── Display ────────────────────────────────────────────────────────────────────
W, H = 1280, 720;  FPS = 60
PUNCH_ACTIONS = ["jab", "cross", "hook", "uppercut"]
COLORS = {
    "jab": (255,155,50), "cross": (255,55,55),
    "hook": (255,215,50), "uppercut": (50,220,255),
    "block_high": (50,130,255), "idle": (130,130,130),
}
BG   = (10, 10, 18);   WHITE = (255,255,255); GREY = (130,130,140)
DIM  = (40, 40, 52);   DIM2  = (62, 62, 74)
GREEN= (55,215,85);    RED   = (220,45,45);   GOLD = (255,210,50)
BLUE = (50,160,255)

# ── Scores ─────────────────────────────────────────────────────────────────────

def load_scores():
    try:
        with open(SCORES_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_scores(data):
    os.makedirs(os.path.dirname(SCORES_PATH), exist_ok=True)
    with open(SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# ── CV helpers ─────────────────────────────────────────────────────────────────

def lm_to_arr(lms):
    r = []
    for lm in lms: r.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(r, dtype=np.float32)

def normalize(fr):
    fr = fr.copy()
    cx=(fr[L_HIP]+fr[R_HIP])/2; cy=(fr[L_HIP+1]+fr[R_HIP+1])/2; cz=(fr[L_HIP+2]+fr[R_HIP+2])/2
    sx=(fr[L_SHO]+fr[R_SHO])/2; sy=(fr[L_SHO+1]+fr[R_SHO+1])/2; sz=(fr[L_SHO+2]+fr[R_SHO+2])/2
    t=max(math.sqrt((sx-cx)**2+(sy-cy)**2+(sz-cz)**2),0.01)
    for i in range(N_LM):
        idx=i*F_PER_LM; fr[idx]=(fr[idx]-cx)/t; fr[idx+1]=(fr[idx+1]-cy)/t; fr[idx+2]=(fr[idx+2]-cz)/t
    return fr

def extract_stats(frames):
    X=np.array(frames,dtype=np.float32)
    if X.shape[0]<SEQ_LEN:
        d=SEQ_LEN-X.shape[0]; pb=d//2; pa=d-pb
        parts=[]
        if pb: parts.append(np.tile(X[:1],(pb,1)))
        parts.append(X)
        if pa: parts.append(np.tile(X[-1:],(pa,1)))
        X=np.vstack(parts)
    elif X.shape[0]>SEQ_LEN:
        s=(X.shape[0]-SEQ_LEN)//2; X=X[s:s+SEQ_LEN]
    return np.concatenate([np.std(X,0),np.max(X,0)-np.min(X,0),X[-1]-X[0]])

def compute_vel(a,b):
    if a is None or b is None: return 0.0
    t=0.0
    for ix,iy in [(RWX,RWY),(LWX,LWY),(REX,REY),(LEX,LEY)]:
        t+=math.sqrt((b[ix]-a[ix])**2+(b[iy]-a[iy])**2)
    return t/4

# ── CV Thread ──────────────────────────────────────────────────────────────────

class CVThread(threading.Thread):
    def __init__(self, rf, scaler):
        super().__init__(daemon=True)
        self.rf=rf; self.scaler=scaler; self.running=True
        self._lock=threading.Lock()
        self._punches=[]       # (action, conf, peak_vel) — classified punches
        self._triggers=[]      # timestamps of motion-start events (for speed bag)
        self._vel=0.0; self._mstate="IDLE"; self._frame=None

    def pop_punches(self):
        with self._lock: q=self._punches[:]; self._punches.clear(); return q

    def pop_triggers(self):
        with self._lock: q=self._triggers[:]; self._triggers.clear(); return q

    @property
    def velocity(self):
        with self._lock: return self._vel
    @property
    def motion_state(self):
        with self._lock: return self._mstate
    @property
    def latest_frame(self):
        with self._lock: return self._frame

    def run(self):
        cap=cv2.VideoCapture(0)
        if not cap.isOpened(): print("ERROR: Cannot open webcam."); return
        opts=PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        lmk=PoseLandmarker.create_from_options(opts)
        ts=0; buf=collections.deque(maxlen=BUF_SIZE); prev=None
        ms="IDLE"; mc=sc=mi=0; last_t=0.0; peak=0.0

        while self.running:
            ok,frame=cap.read()
            if not ok: break
            frame=cv2.flip(frame,1)
            with self._lock: self._frame=frame.copy()

            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
            ts+=33; res=lmk.detect_for_video(img,ts); has=bool(res.pose_landmarks)

            v=0.0
            if has:
                raw=lm_to_arr(res.pose_landmarks[0]); norm=normalize(raw)
                buf.append(norm); v=compute_vel(prev,norm); prev=norm
            else:
                prev=None

            now=time.time()
            if ms=="IDLE":
                if v>VEL_THRESH:
                    mc+=1
                    if mc>=MOTION_START:
                        mi=max(0,len(buf)-mc-LEAD_IN); ms="MOTION"; sc=0; peak=0.0
                        with self._lock: self._triggers.append(now)
                else: mc=0
            elif ms=="MOTION":
                peak=max(peak,v)
                if v<VEL_THRESH:
                    sc+=1
                    if sc>=MOTION_STOP:
                        self._classify(buf,mi,len(buf),peak)
                        last_t=now; ms="COOLDOWN"; mc=sc=0
                else:
                    sc=0
                    if len(buf)-mi>90:
                        self._classify(buf,mi,len(buf),peak)
                        last_t=now; ms="COOLDOWN"; mc=sc=0
            elif ms=="COOLDOWN":
                if now-last_t>COOLDOWN: ms="IDLE"; mc=0

            with self._lock: self._vel=v; self._mstate=ms

        lmk.close(); cap.release()

    def _classify(self,buf,si,ei,peak):
        bl=list(buf); ei=min(len(bl),ei+TAIL)
        seg=ei-si
        if seg<SEQ_LEN:
            d=SEQ_LEN-seg; eb=d//2; ea=d-eb
            ns=max(0,si-eb); ne=min(len(bl),ei+ea)
            ab=si-ns; aa=ne-ei
            if ab<eb: ne=min(len(bl),ne+(eb-ab))
            elif aa<ea: ns=max(0,ns-(ea-aa))
            si,ei=ns,ne
        seg=bl[si:ei]
        if len(seg)<10: return
        feats=extract_stats(seg)
        fs=self.scaler.transform(feats.reshape(1,-1))
        act=self.rf.predict(fs)[0]; prob=self.rf.predict_proba(fs)[0]; conf=float(np.max(prob))
        ranked=sorted(zip(self.rf.classes_,prob),key=lambda x:-x[1])
        print(f"  >> {act.upper():>12s} ({conf*100:.0f}%)  "+"  ".join(f"{a}:{p*100:.0f}%" for a,p in ranked))
        if act=="idle" or conf<MIN_CONF: return
        with self._lock: self._punches.append((act,conf,peak))

# ── Shared render helpers ──────────────────────────────────────────────────────

def make_fonts():
    pygame.font.init()
    names=["Segoe UI","Arial","Helvetica",None]
    def get(size,bold=False):
        for n in names:
            try: return pygame.font.SysFont(n,size,bold=bold)
            except: pass
        return pygame.font.Font(None,size)
    return {"title":get(80,True),"big":get(60,True),"med":get(36,True),
            "sm":get(24),"xs":get(17)}

def blit_c(surf,font,text,color,cx,cy,alpha=255):
    s=font.render(text,True,color)
    if alpha<255: s=s.copy(); s.set_alpha(alpha)
    surf.blit(s,(cx-s.get_width()//2,cy-s.get_height()//2))

def rounded_rect(surf,color,rect,radius=12,border=0,bcol=None,alpha=255):
    r=pygame.Rect(rect); tmp=pygame.Surface((r.w,r.h),pygame.SRCALPHA)
    pygame.draw.rect(tmp,(*color,alpha),tmp.get_rect(),border_radius=radius)
    if border and bcol:
        pygame.draw.rect(tmp,(*bcol,alpha),tmp.get_rect(),border,border_radius=radius)
    surf.blit(tmp,r.topleft)

_last_cam_surf = None  # cached last good frame surface

def cam_surf(frame, size=(W,H)):
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    rgb=cv2.resize(rgb,size)
    return pygame.surfarray.make_surface(rgb.swapaxes(0,1))

def draw_bg(screen, cv_thread):
    global _last_cam_surf
    frame=cv_thread.latest_frame
    if frame is not None:
        try:
            s=cam_surf(frame)
            _last_cam_surf=s
        except Exception:
            s=_last_cam_surf
    else:
        s=_last_cam_surf
    if s is not None:
        screen.blit(s,(0,0))
        ov=pygame.Surface((W,H),pygame.SRCALPHA); ov.fill((0,0,0,172)); screen.blit(ov,(0,0))
    else:
        screen.fill(BG)

def draw_vel_bar(screen, cv_thread, F):
    v=cv_thread.velocity; ms=cv_thread.motion_state
    bw,bh=420,14; bx=W//2-bw//2; by=H-22
    pygame.draw.rect(screen,(25,25,35),(bx,by,bw,bh),border_radius=7)
    fill=int(min(v/(VEL_THRESH*3),1.0)*bw)
    if fill>0:
        col=(215,50,50) if ms=="MOTION" else (50,195,75) if ms=="IDLE" else BLUE
        pygame.draw.rect(screen,col,(bx,by,fill,bh),border_radius=7)
    tx=bx+bw//3
    pygame.draw.rect(screen,(0,210,210),(tx,by-2,2,bh+4))
    sc={"IDLE":(70,195,75),"MOTION":(215,50,50),"COOLDOWN":BLUE}
    blit_c(screen,F["xs"],ms,sc.get(ms,GREY),bx-48,by+bh//2)

def draw_hud(screen, F, left="", center="", right=""):
    hud=pygame.Surface((W,50),pygame.SRCALPHA); hud.fill((0,0,0,160)); screen.blit(hud,(0,0))
    if left:   blit_c(screen,F["sm"],left,(180,180,255),110,25)
    if center: blit_c(screen,F["sm"],center,GOLD,W//2,25)
    if right:  blit_c(screen,F["sm"],right,(150,255,160),W-110,25)

def particles_update(parts, dt):
    for p in parts[:]:
        p[0]+=p[2]; p[1]+=p[3]; p[3]+=0.18; p[5]-=dt
        if p[5]<=0: parts.remove(p)

def particles_draw(screen, parts):
    for p in parts:
        lr=max(0.0,p[5]/p[6]); r=max(2,int(6*lr))
        pygame.draw.circle(screen,p[4],(int(p[0]),int(p[1])),r)

def spawn_particles(parts, x, y, color, n=22):
    for _ in range(n):
        a=random.uniform(0,math.tau); spd=random.uniform(2,9); life=random.uniform(0.35,0.75)
        parts.append([x,y,spd*math.cos(a),spd*math.sin(a),color,life,life])

# ══════════════════════════════════════════════════════════════════════════════
# GAME 1 — BOXING MEMORY
# ══════════════════════════════════════════════════════════════════════════════

def level_config(lvl):
    return min(2+(lvl-1)//2,7), max(1.5,3.6-lvl*0.14)

class MemoryGame:
    SHOW="SHOW"; READY="READY"; PLAYING="PLAYING"
    FEEDBACK="FEEDBACK"; LEVEL_UP="LEVEL_UP"; DONE="DONE"

    def __init__(self, cv, F, scores):
        self.cv=cv; self.F=F; self.scores=scores
        self.level=1; self.score=0
        self.hi_score=scores.get("memory",{}).get("hi_score",0)
        self.hi_level=scores.get("memory",{}).get("hi_level",1)
        self.sequence=[]; self.progress=[]; self.show_time=3.0
        self.last_act=None; self.state=self.SHOW; self.timer=0.0
        self.flash=None; self.parts=[]; self.floats=[]; self._new_best=False
        self.new_level()

    def new_level(self):
        length,self.show_time=level_config(self.level)
        self.sequence=[random.choice(PUNCH_ACTIONS) for _ in range(length)]
        self.progress=[]; self._enter(self.SHOW)

    def _enter(self,s): self.state=s; self.timer=0.0

    def _slot_rect(self,i,n):
        sw=min(150,(W-120)//n); gap=min(14,sw//8)
        total=n*sw+(n-1)*gap; x0=W//2-total//2
        return pygame.Rect(x0+i*(sw+gap),H-165,sw,75)

    def on_punch(self,action,conf,peak):
        if self.state!=self.PLAYING: return
        idx=len(self.progress); expected=self.sequence[idx]
        correct=(action==expected); self.last_act=(action,correct,conf)
        if correct:
            self.progress.append(action); self.score+=10+self.level*5
            r=self._slot_rect(idx,len(self.sequence))
            spawn_particles(self.parts,r.centerx,r.centery,COLORS[action])
            self.floats.append([action.upper(),COLORS[action],float(r.centerx),float(r.top-20),0.9,0.9])
            self.flash=[50,210,80,130]; self._enter(self.FEEDBACK)
        else:
            new_best=False
            if self.score>self.hi_score or self.level>self.hi_level:
                self.hi_score=max(self.hi_score,self.score)
                self.hi_level=max(self.hi_level,self.level)
                self.scores.setdefault("memory",{})["hi_score"]=self.hi_score
                self.scores.setdefault("memory",{})["hi_level"]=self.hi_level
                save_scores(self.scores); new_best=True
            self._new_best=new_best; self.flash=[210,40,40,200]; self._enter(self.DONE)

    def update(self,dt):
        self.timer+=dt
        particles_update(self.parts,dt)
        for f in self.floats[:]:
            f[3]-=45*dt; f[4]-=dt
            if f[4]<=0: self.floats.remove(f)
        if self.flash:
            self.flash[3]=max(0,self.flash[3]-320*dt)
            if self.flash[3]<=0: self.flash=None
        if self.state==self.SHOW and self.timer>=self.show_time: self._enter(self.READY)
        elif self.state==self.READY and self.timer>=1.1: self._enter(self.PLAYING)
        elif self.state==self.FEEDBACK and self.timer>=0.45:
            if len(self.progress)==len(self.sequence): self._enter(self.LEVEL_UP)
            else: self._enter(self.PLAYING)
        elif self.state==self.LEVEL_UP and self.timer>=2.2:
            self.level+=1
            if self.level>self.hi_level:
                self.hi_level=self.level; self.hi_score=max(self.hi_score,self.score)
                self.scores.setdefault("memory",{})["hi_score"]=self.hi_score
                self.scores.setdefault("memory",{})["hi_level"]=self.hi_level
                save_scores(self.scores)
            self.new_level()

    def draw(self,screen):
        if self.flash and self.flash[3]>0:
            fl=pygame.Surface((W,H),pygame.SRCALPHA); fl.fill((*self.flash[:3],int(self.flash[3]))); screen.blit(fl,(0,0))
        if   self.state==self.SHOW:     self._d_show(screen)
        elif self.state==self.READY:    self._d_ready(screen)
        elif self.state in (self.PLAYING,self.FEEDBACK): self._d_playing(screen)
        elif self.state==self.LEVEL_UP: self._d_levelup(screen)
        elif self.state==self.DONE:     self._d_done(screen)
        draw_hud(screen,self.F,f"LEVEL  {self.level}",f"SCORE  {self.score}",f"BEST  {self.hi_score}")
        draw_vel_bar(screen,self.cv,self.F)
        particles_draw(screen,self.parts)
        for f in self.floats:
            lr=max(0.0,f[4]/f[5]); blit_c(screen,self.F["med"],f[0],f[1],int(f[2]),int(f[3]),int(255*lr))

    def _d_show(self,screen):
        t=self.timer/self.show_time
        alpha=int(255*(t/0.12)) if t<0.12 else int(255*(1-(t-0.85)/0.15)) if t>0.85 else 255
        alpha=max(0,min(255,alpha))
        blit_c(screen,self.F["med"],f"LEVEL {self.level}  —  MEMORISE THIS COMBO",GREY,W//2,105,alpha)
        n=len(self.sequence); sw=min(160,(W-120)//n); gap=min(16,sw//7)
        total=n*sw+(n-1)*gap; x0=W//2-total//2
        for i,p in enumerate(self.sequence):
            x=x0+i*(sw+gap); col=COLORS[p]; rect=pygame.Rect(x,H//2-55,sw,110)
            rounded_rect(screen,col,rect,radius=14,border=3,bcol=WHITE,alpha=int(alpha*0.85))
            blit_c(screen,self.F["med"],p.upper(),WHITE,rect.centerx,rect.centery-8,alpha)
            blit_c(screen,self.F["xs"],str(i+1),GREY,rect.x+14,rect.y+12,alpha)

    def _d_ready(self,screen):
        blit_c(screen,self.F["big"],"GO!",WHITE,W//2,H//2-30,min(255,int(255*self.timer/0.25)))
        self._d_slots(screen)

    def _d_playing(self,screen):
        idx=len(self.progress)
        if idx<len(self.sequence):
            blit_c(screen,self.F["med"],f"Punch {idx+1} of {len(self.sequence)}",GREY,W//2,H//2-30)
            ms=self.cv.motion_state
            if ms=="MOTION":
                a=int(180+75*math.sin(time.time()*8))
                blit_c(screen,self.F["sm"],"● CLASSIFYING...",(a,80,80),W//2,H//2+25)
            elif ms=="COOLDOWN":
                blit_c(screen,self.F["sm"],"● COOLDOWN",BLUE,W//2,H//2+25)
        self._d_slots(screen)

    def _d_slots(self,screen):
        n=len(self.sequence)
        for i in range(n):
            r=self._slot_rect(i,n)
            if i<len(self.progress):
                punch=self.progress[i]; col=COLORS[punch]
                rounded_rect(screen,col,r,radius=10); pygame.draw.rect(screen,WHITE,r,2,border_radius=10)
                blit_c(screen,self.F["sm"],punch.upper(),(0,0,0),r.centerx,r.centery)
                blit_c(screen,self.F["xs"],"✓",(0,0,0),r.x+12,r.y+10)
            elif i==len(self.progress) and self.state in (self.PLAYING,self.READY):
                pulse=0.5+0.5*math.sin(time.time()*6); target=self.sequence[i]; col=COLORS[target]
                rounded_rect(screen,DIM,r,radius=10,border=max(2,int(3+2*pulse)),bcol=col,alpha=255)
                blit_c(screen,self.F["med"],"?",col,r.centerx,r.centery)
            else:
                rounded_rect(screen,DIM,r,radius=10,border=2,bcol=DIM2,alpha=255)
        blit_c(screen,self.F["xs"],f"{len(self.progress)} / {len(self.sequence)}",GREY,W//2,H-72)

    def _d_levelup(self,screen):
        a=min(255,int(255*self.timer/0.2))
        blit_c(screen,self.F["big"],f"LEVEL {self.level} COMPLETE!",GREEN,W//2,H//2-70,a)
        blit_c(screen,self.F["med"],f"Score:  {self.score}",GOLD,W//2,H//2+10,a)
        blit_c(screen,self.F["sm"],f"Next: {level_config(self.level+1)[0]} punches",GREY,W//2,H//2+65,a)

    def _d_done(self,screen):
        a=min(255,int(255*self.timer/0.2))
        blit_c(screen,self.F["big"],"GAME OVER",RED,W//2,H//2-130,a)
        if self.last_act:
            action,_,conf=self.last_act; expected=self.sequence[len(self.progress)]
            blit_c(screen,self.F["med"],f"Threw  {action.upper()}  —  Expected  {expected.upper()}",(210,210,210),W//2,H//2-50,a)
        blit_c(screen,self.F["med"],f"Level {self.level}     Score  {self.score}",GOLD,W//2,H//2+15,a)
        if self._new_best:
            p=int(180+75*math.sin(time.time()*4))
            blit_c(screen,self.F["med"],"★  NEW HIGH SCORE  ★",(p,220,80),W//2,H//2+75,a)
        else:
            blit_c(screen,self.F["sm"],f"Best:  Level {self.hi_level}   Score {self.hi_score}",GREY,W//2,H//2+75,a)
        if self.timer>1.0:
            p2=int(160+95*math.sin(time.time()*2.5))
            blit_c(screen,self.F["med"],"PRESS ESC TO RETURN",(p2,p2,p2),W//2,H//2+150)

# ══════════════════════════════════════════════════════════════════════════════
# GAME 2 — COMBO RUSH  (beat-the-clock combo trainer)
# ══════════════════════════════════════════════════════════════════════════════

COMBO_ROUNDS = 8
COMBO_POOL = [
    ["jab","cross"], ["cross","hook"], ["jab","uppercut"],
    ["hook","cross"], ["uppercut","jab"], ["cross","jab"],
    ["jab","cross","hook"], ["jab","jab","cross"],
    ["hook","uppercut","jab"], ["cross","hook","uppercut"],
    ["jab","hook","cross"], ["uppercut","cross","jab"],
]

def _rand_combo(prev=None):
    choices=[c for c in COMBO_POOL if c!=prev]
    return random.choice(choices)

class ComboRushGame:
    COUNTDOWN="COUNTDOWN"; THROWING="THROWING"; RESULT="RESULT"; DONE="DONE"

    def __init__(self, cv, F, scores):
        self.cv=cv; self.F=F; self.scores=scores
        self.hi=scores.get("combo_rush",{}).get("hi_score",0)
        self.state=self.COUNTDOWN; self.timer=0.0
        self.round=0; self.score=0; self.results=[]
        self.combo=[]; self.progress=[]; self.combo_start=None
        self.last_result=None; self.flash=None; self.parts=[]; self._new_best=False
        self._next_combo(None)

    def _next_combo(self, prev):
        self.combo=_rand_combo(prev); self.progress=[]; self.combo_start=None

    def on_punch(self,action,conf,peak):
        if self.state!=self.THROWING: return
        idx=len(self.progress); expected=self.combo[idx]
        if self.combo_start is None: self.combo_start=time.time()  # start clock on first punch
        if action==expected:
            self.progress.append(action)
            r=self._slot_rect(idx,len(self.combo))
            spawn_particles(self.parts,r.centerx,r.centery,COLORS[action])
            if len(self.progress)==len(self.combo):
                # completed!
                elapsed_ms=int((time.time()-self.combo_start)*1000)
                pts=max(0, 500-elapsed_ms//10)
                self.score+=pts; self.last_result=("ok",elapsed_ms,pts)
                self.flash=[50,210,80,160]
                self._finish_round()
        else:
            # wrong punch
            elapsed_ms=int((time.time()-self.combo_start)*1000) if self.combo_start else 0
            self.last_result=("fail",elapsed_ms,action,expected)
            self.flash=[210,40,40,180]
            self._finish_round()

    def _finish_round(self):
        self.state=self.RESULT; self.timer=0.0

    def update(self,dt):
        self.timer+=dt
        particles_update(self.parts,dt)
        if self.flash:
            self.flash[3]=max(0,self.flash[3]-380*dt)
            if self.flash[3]<=0: self.flash=None
        if self.state==self.COUNTDOWN and self.timer>=3.0:
            self.state=self.THROWING; self.timer=0.0
        elif self.state==self.RESULT and self.timer>=1.8:
            prev=self.combo; self.round+=1
            if self.round>=COMBO_ROUNDS:
                self._new_best=self.score>self.hi
                if self._new_best:
                    self.hi=self.score
                    self.scores.setdefault("combo_rush",{})["hi_score"]=self.hi
                    save_scores(self.scores)
                self.state=self.DONE; self.timer=0.0
            else:
                self._next_combo(prev); self.state=self.THROWING; self.timer=0.0

    def _slot_rect(self,i,n):
        sw=min(160,(W-120)//n); gap=16
        total=n*sw+(n-1)*gap; x0=W//2-total//2
        return pygame.Rect(x0+i*(sw+gap),H//2+20,sw,80)

    def draw(self,screen):
        if self.flash and self.flash[3]>0:
            fl=pygame.Surface((W,H),pygame.SRCALPHA); fl.fill((*self.flash[:3],int(self.flash[3]))); screen.blit(fl,(0,0))
        if   self.state==self.COUNTDOWN: self._d_countdown(screen)
        elif self.state in (self.THROWING,self.RESULT): self._d_throwing(screen)
        elif self.state==self.DONE: self._d_done(screen)
        draw_hud(screen,self.F,"COMBO RUSH",f"SCORE  {self.score}",f"BEST  {self.hi}")
        draw_vel_bar(screen,self.cv,self.F)
        particles_draw(screen,self.parts)

    def _d_countdown(self,screen):
        n=max(1,3-int(self.timer))
        cols={3:(220,60,60),2:(220,180,50),1:(60,220,80)}
        blit_c(screen,self.F["title"],str(n),cols.get(n,WHITE),W//2,H//2-40)
        blit_c(screen,self.F["sm"],"Throw the combo in order — as fast as possible!",GREY,W//2,H//2+60)

    def _d_throwing(self,screen):
        blit_c(screen,self.F["sm"],f"Round {self.round+1} / {COMBO_ROUNDS}  —  throw this combo:",GREY,W//2,H//2-80)
        n=len(self.combo)
        for i,p in enumerate(self.combo):
            r=self._slot_rect(i,n); col=COLORS[p]
            if i<len(self.progress):
                rounded_rect(screen,col,r,radius=10); pygame.draw.rect(screen,WHITE,r,2,border_radius=10)
                blit_c(screen,self.F["sm"],p.upper(),(0,0,0),r.centerx,r.centery-6)
                blit_c(screen,self.F["xs"],"✓",(0,0,0),r.x+10,r.y+8)
            elif i==len(self.progress):
                pulse=0.5+0.5*math.sin(time.time()*6)
                rounded_rect(screen,DIM,r,radius=10,border=max(2,int(3+2*pulse)),bcol=col,alpha=255)
                blit_c(screen,self.F["med"],p.upper(),col,r.centerx,r.centery)
            else:
                rounded_rect(screen,DIM,r,radius=10,border=2,bcol=DIM2,alpha=255)
                blit_c(screen,self.F["sm"],p.upper(),GREY,r.centerx,r.centery)

        # timer or result feedback
        if self.state==self.RESULT and self.last_result:
            if self.last_result[0]=="ok":
                _,ms,pts=self.last_result
                blit_c(screen,self.F["med"],f"✓  {ms}ms   +{pts} pts",GREEN,W//2,H//2+140)
            else:
                _,ms,wrong,exp=self.last_result
                blit_c(screen,self.F["med"],f"✗  threw {wrong.upper()} — wanted {exp.upper()}",RED,W//2,H//2+140)
        elif self.combo_start is not None:
            elapsed=int((time.time()-self.combo_start)*1000)
            col=RED if elapsed>2000 else GOLD if elapsed>1000 else GREEN
            blit_c(screen,self.F["big"],f"{elapsed}ms",col,W//2,H//2-140)
        else:
            ms=self.cv.motion_state
            if ms=="MOTION":
                a=int(180+75*math.sin(time.time()*8))
                blit_c(screen,self.F["sm"],"● CLASSIFYING...",(a,80,80),W//2,H//2-140)
            else:
                pulse=int(160+95*math.sin(time.time()*3))
                blit_c(screen,self.F["med"],"THROW!",(pulse,pulse,pulse),W//2,H//2-140)

    def _d_done(self,screen):
        a=min(255,int(255*self.timer/0.2))
        blit_c(screen,self.F["big"],"COMBO RUSH DONE!",WHITE,W//2,H//2-110,a)
        blit_c(screen,self.F["title"],str(self.score),GOLD,W//2,H//2-30,a)
        blit_c(screen,self.F["sm"],"points",GREY,W//2,H//2+55,a)
        if self._new_best:
            p=int(180+75*math.sin(time.time()*4))
            blit_c(screen,self.F["med"],"★  NEW HIGH SCORE  ★",(p,220,80),W//2,H//2+110,a)
        else:
            blit_c(screen,self.F["sm"],f"Best:  {self.hi}",GREY,W//2,H//2+110,a)
        if self.timer>1.0:
            p2=int(160+95*math.sin(time.time()*2.5))
            blit_c(screen,self.F["med"],"PRESS ESC TO RETURN",(p2,p2,p2),W//2,H//2+165,a)

# ══════════════════════════════════════════════════════════════════════════════
# GAME 3 — DEFENSE DRILL  (read the attack, counter or block)
# ══════════════════════════════════════════════════════════════════════════════

# Valid counters per incoming attack. block_high always works but scores less.
# Perfect counter = 100pts, block = 40pts, wrong/miss = 0pts + lose HP
COUNTER_MAP = {
    "jab":      {"cross":100,     "hook":80,      "block_high":40},
    "cross":    {"hook":100,      "uppercut":80,  "block_high":40},
    "hook":     {"uppercut":100,  "cross":80,     "block_high":40},
    "uppercut": {"jab":100,       "cross":80,     "block_high":40},
}
# Flavour arrows shown with each attack to hint direction (no counter spoilers)
ATTACK_ARROW = {"jab":"→","cross":"→→","hook":"↗","uppercut":"↑"}
ATTACK_WINDOW = 3.5   # seconds to respond before taking damage
DEFENSE_ROUNDS = 10
DEFENSE_HP = 5

class DefenseDrillGame:
    GUIDE="GUIDE"; WAITING="WAITING"
    JUDGING="JUDGING"; KO="KO"; DONE="DONE"

    def __init__(self, cv, F, scores):
        self.cv=cv; self.F=F; self.scores=scores
        self.hi=scores.get("defense",{}).get("hi_score",0)
        self.state=self.GUIDE; self.timer=0.0
        self.round=0; self.score=0; self.hp=DEFENSE_HP
        self.attack=None; self.last_result=None
        self.flash=None; self.parts=[]; self._new_best=False
        self.results=[]   # ("hit"/"block"/"miss"/"wrong", pts)

    def _next_attack(self):
        self.attack=random.choice(list(COUNTER_MAP.keys()))
        self.state=self.WAITING; self.timer=0.0

    def on_key(self, key):
        if self.state==self.GUIDE and key==pygame.K_SPACE:
            self._next_attack()

    def on_punch(self,action,conf,peak):
        if self.state!=self.WAITING: return
        pts=COUNTER_MAP[self.attack].get(action,0)
        if pts>0:
            tag="hit" if pts>=80 else "block"
            self.score+=pts
            self.flash=[50,210,80,160] if pts>=80 else [255,200,50,130]
            spawn_particles(self.parts,W//2,H//2-60,COLORS[action],30)
        else:
            tag="wrong"; self.hp-=1
            self.flash=[210,40,40,200]
        self.last_result=(tag,action,pts,self.attack)
        self.results.append((tag,pts))
        self.state=self.JUDGING; self.timer=0.0

    def update(self,dt):
        self.timer+=dt
        particles_update(self.parts,dt)
        if self.flash:
            self.flash[3]=max(0,self.flash[3]-360*dt)
            if self.flash[3]<=0: self.flash=None

        if self.state==self.WAITING and self.timer>=ATTACK_WINDOW:
            # timeout — took the hit
            self.hp-=1; self.last_result=("miss",None,0,self.attack)
            self.results.append(("miss",0))
            self.flash=[210,40,40,200]
            self.state=self.JUDGING; self.timer=0.0
        elif self.state==self.JUDGING and self.timer>=1.2:
            if self.hp<=0:
                self._finish(ko=True)
            else:
                self.round+=1
                if self.round>=DEFENSE_ROUNDS:
                    self._finish(ko=False)
                else:
                    self._next_attack()

    def _finish(self,ko):
        self._new_best=self.score>self.hi
        if self._new_best:
            self.hi=self.score
            self.scores.setdefault("defense",{})["hi_score"]=self.hi
            save_scores(self.scores)
        self.state=self.KO if ko else self.DONE; self.timer=0.0

    def draw(self,screen):
        if self.flash and self.flash[3]>0:
            fl=pygame.Surface((W,H),pygame.SRCALPHA); fl.fill((*self.flash[:3],int(self.flash[3]))); screen.blit(fl,(0,0))
        if   self.state==self.GUIDE:                     self._d_countdown(screen)
        elif self.state in (self.WAITING,self.JUDGING): self._d_waiting(screen)
        elif self.state in (self.DONE,self.KO):         self._d_done(screen)
        draw_hud(screen,self.F,"DEFENSE DRILL",
                 f"Round {min(self.round+1,DEFENSE_ROUNDS)} / {DEFENSE_ROUNDS}",
                 f"BEST  {self.hi}")
        self._d_hp(screen)
        draw_vel_bar(screen,self.cv,self.F)
        particles_draw(screen,self.parts)

    def _d_hp(self,screen):
        for i in range(DEFENSE_HP):
            x=W//2-DEFENSE_HP*22//2+i*22; y=H-50
            col=RED if i<self.hp else (40,40,50)
            pygame.draw.circle(screen,col,(x,y),8)
            if i<self.hp: pygame.draw.circle(screen,(255,120,120),(x,y),8,2)

    def _d_countdown(self,screen):
        # Counter guide takes up most of screen for full 5s
        blit_c(screen,self.F["big"],"COUNTER GUIDE",WHITE,W//2,60)
        blit_c(screen,self.F["sm"],"Memorise these — you won't be told mid-fight!",GREY,W//2,115)

        rows=[
            ("JAB",       "cross",    "hook",     (255,155,50)),
            ("CROSS",     "hook",     "uppercut", (255,55,55)),
            ("HOOK",      "uppercut", "cross",    (255,215,50)),
            ("UPPERCUT",  "jab",      "cross",    (50,220,255)),
        ]
        row_h=90; y0=165
        for i,(atk,best,alt,col) in enumerate(rows):
            y=y0+i*row_h
            # attack box
            rounded_rect(screen,col,(W//2-540,y,170,70),radius=10,alpha=60)
            blit_c(screen,self.F["med"],atk,col,W//2-455,y+35)
            blit_c(screen,self.F["xs"],"INCOMING",GREY,W//2-455,y+58)
            # arrow
            blit_c(screen,self.F["big"],"→",GREY,W//2-250,y+35)
            # best counter
            bcol=COLORS.get(best,WHITE)
            rounded_rect(screen,bcol,(W//2-195,y,160,70),radius=10,alpha=70)
            blit_c(screen,self.F["med"],best.upper(),bcol,W//2-115,y+28)
            blit_c(screen,self.F["xs"],"100 pts",GREEN,W//2-115,y+52)
            # alt counter
            acol=COLORS.get(alt,WHITE)
            rounded_rect(screen,acol,(W//2+0,y,160,70),radius=10,alpha=50)
            blit_c(screen,self.F["med"],alt.upper(),acol,W//2+80,y+28)
            blit_c(screen,self.F["xs"],"80 pts",GOLD,W//2+80,y+52)
            # block always
            rounded_rect(screen,(50,130,255),(W//2+195,y,175,70),radius=10,alpha=40)
            blit_c(screen,self.F["med"],"BLOCK",BLUE,W//2+283,y+28)
            blit_c(screen,self.F["xs"],"40 pts  (always works)",(100,150,255),W//2+283,y+52)

        p=int(160+95*math.sin(time.time()*2.5))
        blit_c(screen,self.F["med"],"PRESS SPACE WHEN READY",(p,p,p),W//2,H-45)
        blit_c(screen,self.F["xs"],"Wrong punch or timeout = -1 HP",(160,80,80),W//2,H-18)

    def _d_waiting(self,screen):
        col=COLORS[self.attack]
        blit_c(screen,self.F["title"],self.attack.upper(),col,W//2,H//2-60)
        arrow=ATTACK_ARROW[self.attack]
        blit_c(screen,self.F["big"],arrow,col,W//2-200,H//2-60)

        # countdown bar
        if self.state==self.WAITING:
            frac=max(0.0,1.0-self.timer/ATTACK_WINDOW)
            bw,bh=500,16; bx=W//2-bw//2; by=H//2+50
            pygame.draw.rect(screen,(25,25,35),(bx,by,bw,bh),border_radius=8)
            fc=RED if frac<0.35 else GOLD if frac<0.65 else GREEN
            if frac>0: pygame.draw.rect(screen,fc,(bx,by,int(bw*frac),bh),border_radius=8)
            blit_c(screen,self.F["xs"],"TIME TO COUNTER",GREY,W//2,by+30)

        # result feedback
        if self.state==self.JUDGING and self.last_result:
            tag,action,pts,atk=self.last_result
            if tag=="miss":
                blit_c(screen,self.F["big"],"TOO SLOW! -1 HP",RED,W//2,H//2+80)
            elif tag=="wrong":
                blit_c(screen,self.F["med"],f"WRONG!  {action.upper()}  is no good here  -1 HP",RED,W//2,H//2+80)
            elif tag=="block":
                blit_c(screen,self.F["med"],f"BLOCKED!  +{pts} pts  (find the counter for more)",GOLD,W//2,H//2+80)
            else:
                blit_c(screen,self.F["big"],f"PERFECT COUNTER!  +{pts} pts",GREEN,W//2,H//2+80)

        # motion state indicator
        if self.state==self.WAITING:
            ms=self.cv.motion_state
            if ms=="MOTION":
                a=int(180+75*math.sin(time.time()*8))
                blit_c(screen,self.F["sm"],"● CLASSIFYING...",(a,80,80),W//2,H//2+100)

        # round dots
        dot_r=9; spacing=28; total_w=(DEFENSE_ROUNDS-1)*spacing; x0=W//2-total_w//2
        for i in range(DEFENSE_ROUNDS):
            x=x0+i*spacing; y=H//2+155
            if i<len(self.results):
                tag=self.results[i][0]
                c=GREEN if tag=="hit" else GOLD if tag=="block" else RED
                pygame.draw.circle(screen,c,(x,y),dot_r)
            else:
                pygame.draw.circle(screen,DIM2,(x,y),dot_r,2)

    def _d_done(self,screen):
        a=min(255,int(255*self.timer/0.2))
        ko=self.state==self.KO
        title="K.O.!!" if ko else "DRILL COMPLETE!"
        tcol=RED if ko else GREEN
        blit_c(screen,self.F["big"],title,tcol,W//2,H//2-130,a)
        hits=sum(1 for t,_ in self.results if t=="hit")
        blocks=sum(1 for t,_ in self.results if t=="block")
        misses=sum(1 for t,_ in self.results if t in ("miss","wrong"))
        blit_c(screen,self.F["med"],f"Counters: {hits}   Blocks: {blocks}   Misses: {misses}",
               WHITE,W//2,H//2-60,a)
        blit_c(screen,self.F["title"],str(self.score),GOLD,W//2,H//2+10,a)
        blit_c(screen,self.F["sm"],"points",GREY,W//2,H//2+80,a)
        if self._new_best:
            p=int(180+75*math.sin(time.time()*4))
            blit_c(screen,self.F["med"],"★  NEW HIGH SCORE  ★",(p,220,80),W//2,H//2+115,a)
        else:
            blit_c(screen,self.F["sm"],f"Best:  {self.hi}",GREY,W//2,H//2+115,a)
        if self.timer>1.0:
            p2=int(160+95*math.sin(time.time()*2.5))
            blit_c(screen,self.F["med"],"PRESS ESC TO RETURN",(p2,p2,p2),W//2,H//2+165,a)

# ══════════════════════════════════════════════════════════════════════════════
# HUB SCREEN
# ══════════════════════════════════════════════════════════════════════════════

GAMES_META = [
    {"key":"memory",    "name":"MEMORY",    "sub":"Simon Says with punches",  "color":(255,155,50)},
    {"key":"combo_rush","name":"COMBO RUSH","sub":"Beat-the-clock combos",    "color":(50,220,255)},
    {"key":"defense",   "name":"DEFENSE",   "sub":"Read the attack & counter","color":(255,215,50)},
]

TW, TH, GAPX = 320, 200, 50

def _tile_positions():
    total = len(GAMES_META)*TW + (len(GAMES_META)-1)*GAPX
    ox = W//2 - total//2
    oy = 240
    return [(ox+i*(TW+GAPX), oy) for i in range(len(GAMES_META))]

class Hub:
    def __init__(self, screen, cv, F, scores):
        self.screen=screen; self.cv=cv; self.F=F; self.scores=scores
        self.sel=0; self.timer=0.0

    def best_str(self,key):
        s=self.scores
        if key=="memory":      return f"Best score: {s.get('memory',{}).get('hi_score',0)}"
        if key=="combo_rush":  return f"Best score: {s.get('combo_rush',{}).get('hi_score',0)}"
        if key=="defense":     return f"Best score: {s.get('defense',{}).get('hi_score',0)}"
        return ""

    def tile_at(self, mx, my):
        """Return game index (0-3) if (mx,my) is inside a tile, else None."""
        for i, (x,y) in enumerate(_tile_positions()):
            if x <= mx <= x+TW and y <= my <= y+TH:
                return i
        return None

    def handle_key(self,key):
        if key in (pygame.K_LEFT,pygame.K_UP):   self.sel=(self.sel-1)%3
        if key in (pygame.K_RIGHT,pygame.K_DOWN): self.sel=(self.sel+1)%3
        if key in (pygame.K_SPACE,pygame.K_RETURN): return self.sel
        return None

    def handle_mouse_move(self, mx, my):
        idx = self.tile_at(mx, my)
        if idx is not None:
            self.sel = idx

    def handle_click(self, mx, my):
        return self.tile_at(mx, my)

    def update(self,dt): self.timer+=dt

    def draw(self):
        self.screen.fill(BG)
        blit_c(self.screen,self.F["title"],"BOXING TRAINING HUB",GOLD,W//2,90)
        blit_c(self.screen,self.F["sm"],"Click a game or use arrow keys  ·  ESC to quit",GREY,W//2,155)

        mx,my=pygame.mouse.get_pos()
        positions=_tile_positions()
        for i,(meta,pos) in enumerate(zip(GAMES_META,positions)):
            x,y=pos; col=meta["color"]
            hovered=(x<=mx<=x+TW and y<=my<=y+TH)
            selected=(i==self.sel)
            active=selected or hovered
            pulse=0.55+0.45*math.sin(self.timer*4) if active else 0.0
            border_w=4 if active else 2
            bg_alpha=int(55+40*pulse) if active else 30
            rounded_rect(self.screen,col,(x,y,TW,TH),radius=16,alpha=bg_alpha)
            bcol=WHITE if active else col
            rounded_rect(self.screen,(0,0,0),(x,y,TW,TH),radius=16,border=border_w,bcol=bcol,alpha=0)

            blit_c(self.screen,self.F["med"],meta["name"],WHITE if active else col,x+TW//2,y+55)
            blit_c(self.screen,self.F["xs"],meta["sub"],GREY,x+TW//2,y+100)
            blit_c(self.screen,self.F["xs"],self.best_str(meta["key"]),(150,220,150),x+TW//2,y+135)

            if active:
                label="▶ CLICK TO PLAY" if hovered else "▶ PRESS SPACE"
                blit_c(self.screen,self.F["xs"],label,WHITE,x+TW//2,y+162)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def make_game(idx, cv, F, scores):
    if idx==0: return MemoryGame(cv,F,scores)
    if idx==1: return ComboRushGame(cv,F,scores)
    if idx==2: return DefenseDrillGame(cv,F,scores)

DONE_STATES = {"DONE","GAME_OVER"}  # states where ESC returns to hub

def main():
    print("Loading model...")
    rf=joblib.load(RF_PATH); scaler=joblib.load(SCALER_PATH)
    print("Model loaded.\n")
    cv=CVThread(rf,scaler); cv.start()

    pygame.init()
    screen=pygame.display.set_mode((W,H))
    pygame.display.set_caption("Boxing Training Hub")
    clock=pygame.time.Clock(); F=make_fonts()
    scores=load_scores()
    hub=Hub(screen,cv,F,scores)
    active_game=None

    while True:
        dt=clock.tick(FPS)/1000.0

        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                cv.running=False; pygame.quit(); sys.exit()
            if event.type==pygame.KEYDOWN:
                if event.key in (pygame.K_q,pygame.K_ESCAPE):
                    if active_game is None:
                        cv.running=False; pygame.quit(); sys.exit()
                    else:
                        active_game=None
                elif active_game is not None:
                    if hasattr(active_game,'on_key'): active_game.on_key(event.key)
                elif active_game is None:
                    idx=hub.handle_key(event.key)
                    if idx is not None:
                        scores=load_scores()
                        active_game=make_game(idx,cv,F,scores)
            if event.type==pygame.MOUSEMOTION and active_game is None:
                hub.handle_mouse_move(*event.pos)
            if event.type==pygame.MOUSEBUTTONDOWN and event.button==1 and active_game is None:
                idx=hub.handle_click(*event.pos)
                if idx is not None:
                    scores=load_scores()
                    active_game=make_game(idx,cv,F,scores)

        if active_game is None:
            for _,conf,peak in cv.pop_punches(): pass  # drain queue in hub
            cv.pop_triggers()
            hub.scores=scores; hub.update(dt)
            hub.draw()  # hub.draw() fills BG itself — no camera on hub
        else:
            for action,conf,peak in cv.pop_punches():
                active_game.on_punch(action,conf,peak)
            active_game.update(dt)
            draw_bg(screen,cv)
            active_game.draw(screen)
            # auto-return to hub after game over (ESC or after long wait)
            if hasattr(active_game,'state') and active_game.state in DONE_STATES:
                if active_game.timer>8.0:
                    active_game=None

        pygame.display.flip()

if __name__=="__main__":
    main()
