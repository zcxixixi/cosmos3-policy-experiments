"""Render the Chinese experiment explanation from archived evidence.

Needs Pillow, NumPy, edge-tts (Chinese narration), and an existing FFmpeg.
No model inference, synthetic experiment footage, or changes to original data.
Intermediate narration and silent video stay in the supplied work directory.
"""
from __future__ import annotations

import argparse
import asyncio
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 24
BG = "#f3f6fb"
INK = "#17263c"
MUTED = "#637187"
BLUE = "#3168d8"
GREEN = "#19816c"
ORANGE = "#bc651c"
FONT = Path("/home/cenxi/.local/share/fonts/MiSans/MiSans-Normal.ttf")
BOLD = Path("/home/cenxi/.local/share/fonts/MiSans/MiSans-Bold.ttf")
ROOT = Path(__file__).resolve().parents[1]


SCENES = [
    dict(key="00_intro", kind="intro", chapter="01 / 我们在查什么", title="指令和摆放，分别怎样影响它抓谁？", minimum=11,
         voice="我们想查清楚，指令和摆放分别怎样影响机械臂抓谁？这次没有重新训练模型，而是改指令、改摆放，再让仿真机械臂实际执行。"),
    dict(key="01_milk", kind="clip", chapter="02 / 同一起点，只换目标名称", title="说拿牛奶：抓起牛奶", minimum=9,
         source="experiments/09-system-prompt/closed-loop/center_milk_off", target="牛奶", selected="牛奶", prompt="拿牛奶，放进篮子", color=GREEN,
         notes=["牛奶在原来的位置", "没有替换模型", "实际抓起：牛奶"],
         voice="先让它拿牛奶。牛奶在原位置时，机械臂抓起了牛奶。注意，抓起来还不等于放进篮子成功。"),
    dict(key="02_butter", kind="clip", chapter="02 / 同一起点，只换目标名称", title="改说拿黄油：还是抓牛奶", minimum=10,
         source="experiments/09-system-prompt/closed-loop/center_butter_off", target="黄油", selected="牛奶", prompt="拿黄油，放进篮子", color=ORANGE,
         notes=["恢复到同一份初态", "只换指令中的目标", "实际仍抓：牛奶"],
         voice="恢复到同一份初态，只把指令改成拿黄油。它还是抓牛奶。这次换目标名称，没有换对抓取对象。"),
    dict(key="03_far", kind="clip", chapter="03 / 改位置，再看它抓谁", title="牛奶移远15厘米：改抓奶酪盒", minimum=10,
         source="experiments/09-system-prompt/closed-loop/far_milk_off", target="牛奶", selected="奶酪盒", prompt="拿牛奶，放进篮子", color=ORANGE,
         notes=["牛奶沿X轴移远15 cm", "指令仍然是拿牛奶", "实际抓起：奶酪盒"],
         voice="把牛奶移远十五厘米，仍然说拿牛奶。它却改抓奶酪盒。接下来要查，它认的是奶酪，还是那个位置？"),
    dict(key="04_swap", kind="clip", chapter="03 / 改位置，再看它抓谁", title="交换奶酪和黄油：抓旧位置上的黄油", minimum=12,
         source="experiments/10-identity-position/far_cream_butter_swap_milk_off", target="牛奶", selected="黄油", prompt="拿牛奶，放进篮子", color=ORANGE,
         notes=["牛奶保持移远，指令不变", "交换奶酪盒和黄油的位置", "抓了旧奶酪位置上的黄油"],
         voice="牛奶保持移远，指令也不变。我们交换奶酪盒和黄油的位置。它改抓了旧奶酪位置上的黄油。这支持摆放位置在影响它抓谁。"),
    dict(key="05_arrays", kind="arrays", chapter="04 / 去网络里面查真实数字", title="语言确实影响了动作前的内部数组", minimum=16,
         voice="再去网络内部看真实数字。固定相机画面和随机噪声，只换牛奶与黄油两个目标词。动作前的数组确实改变了。这里的百分比是数组差异，不是听懂了多少。"),
    dict(key="06_routes", kind="routes", chapter="04 / 去网络里面查真实数字", title="数字会变，仍不等于抓对了目标", minimum=16,
         voice="我们还分别改了文字直接影响动作，以及经过预测画面影响动作的两种读取方式。两种都能改动作数字，但六条实际执行仍没改抓黄油。参与计算，不等于选对物体。"),
    dict(key="07_goal_input", kind="goal_input", chapter="05 / 找一个正常能换对目标的对照", title="换到碗和酒瓶：从同一初态出发", minimum=10,
         voice="但是，不能说模型根本不使用语言。我们找了一组碗和酒瓶的共同初态，用同一个模型，只换这两条环境原有的指令。"),
    dict(key="08_bowl", kind="clip", chapter="05 / 同一初态，指令改变了抓取对象", title="说放碗：抓起了碗，但最后没放好", minimum=10,
         source="experiments/15-native-goal-targets/policy/closed-loop/bowl", target="碗", selected="碗", prompt="把碗放到柜子上", color=GREEN,
         notes=["实际抓起：碗", "最高抬升28.6 cm", "最后没有通过放置检查"],
         voice="说把碗放到柜子上，它抓起了碗。最高抬升二十八点六厘米。目标抓对了，但最后没有完成放置。"),
    dict(key="09_wine", kind="clip", chapter="05 / 同一初态，指令改变了抓取对象", title="说放酒瓶：抓起酒瓶，但最后没放好", minimum=11,
         source="experiments/15-native-goal-targets/policy/closed-loop/wine", target="酒瓶", selected="酒瓶", prompt="把酒瓶放到柜子上", color=GREEN,
         notes=["实际抓起：酒瓶", "最高抬升30.0 cm", "最后仍夹着，没放好"],
         voice="从同一初态出发，说把酒瓶放到柜子上，它抓起酒瓶。最高抬升三十厘米。但结束时还夹着，没有放好。"),
    dict(key="10_conclusion", kind="conclusion", chapter="06 / 现在知道什么，还差什么", title="有了明确证据，还没有找到唯一根因", minimum=17,
         voice="现在能确认，语言可以起作用，摆放也会影响选择。仍不知道牛奶场景为什么抓错。我们还没证明它在背训练答案，也没定位到为什么选错目标。下一步是只换指定的内部数据，再用实际抓取核验原因。"),
]


@lru_cache(maxsize=40)
def font(size, bold=False):
    return ImageFont.truetype(str(BOLD if bold else FONT), size)


def text(draw, xy, value, size=38, fill=INK, bold=False):
    draw.text(xy, value, font=font(size, bold), fill=fill, anchor="lt")


def wrap(value, size, width):
    lines, line = [], ""
    f = font(size)
    for ch in value:
        if f.getlength(line + ch) > width and line:
            lines.append(line)
            line = ch
        else:
            line += ch
    if line:
        lines.append(line)
    return lines


def paragraph(draw, xy, value, size=36, width=500, fill=INK, gap=16):
    x, y = xy
    for line in wrap(value, size, width):
        text(draw, (x, y), line, size, fill)
        y += size + gap
    return y


def box(draw, bounds, fill="white", outline=None, radius=22):
    draw.rounded_rectangle(bounds, radius=radius, fill=fill, outline=outline, width=2)


def arrow(draw, start, end, color=BLUE, progress=1.0):
    x1, y1 = start
    x2, y2 = end
    draw.line([start, end], fill=color, width=5)
    theta = math.atan2(y2-y1, x2-x1)
    pts = [(x2,y2),(x2-20*math.cos(theta-.5), y2-20*math.sin(theta-.5)),(x2-20*math.cos(theta+.5),y2-20*math.sin(theta+.5))]
    draw.polygon(pts, fill=color)
    p = progress % 1
    px, py = x1+(x2-x1)*p, y1+(y2-y1)*p
    draw.ellipse((px-8, py-8, px+8, py+8), fill=color)


def duration(ffmpeg, path):
    p = subprocess.run([ffmpeg,"-hide_banner","-i",str(path)],capture_output=True,text=True)
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)",p.stderr)
    if not m:
        raise ValueError(f"Cannot read duration: {path}")
    return int(m[1])*3600+int(m[2])*60+float(m[3])


def decode_clip(ffmpeg, path):
    raw = subprocess.check_output([ffmpeg,"-v","error","-i",str(path),"-f","rawvideo","-pix_fmt","rgb24","-vf","scale=512:256","pipe:1"])
    frames = np.frombuffer(raw,dtype=np.uint8).reshape(-1,256,512,3)
    assert len(frames)==129, (path,frames.shape)
    return frames


def evidence():
    raw=json.loads((ROOT/"experiments/07-language-path/result.json").read_text())
    vals=[c for c in raw["contrasts"] if c["kind"]=="language_center"]
    assert len(vals)==3
    first=float(np.mean([c["first_readout_relative"] for c in vals])*100)
    last=float(np.mean([c["last_readout_relative"] for c in vals])*100)
    assert round(first,2)==2.04 and round(last,2)==8.65
    groups=json.loads((ROOT/"experiments/10-identity-position/summary.json").read_text())["groups"]
    assert len(groups)==4 and all(len(g["seeds"])==3 for g in groups)
    goals=json.loads((ROOT/"experiments/15-native-goal-targets/policy/closed-loop/result.json").read_text())["cases"]
    assert len(goals)==2 and all(c["target_selected"] and not c["final_native_target_goal"] for c in goals)
    return dict(first=first,last=last)


async def prepare_voice(work):
    import edge_tts
    sem=asyncio.Semaphore(2)
    async def one(scene):
        audio=work/"narration"/(scene["key"]+".mp3")
        meta=audio.with_suffix(".jsonl")
        text_hash=hashlib.sha256(scene["voice"].encode()).hexdigest()
        marker=audio.with_suffix(".txt")
        if audio.exists() and meta.exists() and marker.exists() and marker.read_text()==text_hash:
            return
        async with sem:
            await asyncio.wait_for(edge_tts.Communicate(scene["voice"],"zh-CN-XiaoxiaoNeural",rate="+5%",boundary="SentenceBoundary").save(str(audio),str(meta)),60)
        marker.write_text(text_hash)
        print("narration:",scene["key"],flush=True)
    (work/"narration").mkdir(parents=True,exist_ok=True)
    await asyncio.gather(*(one(s) for s in SCENES))


def build_timeline(ffmpeg, work):
    time=0
    for scene in SCENES:
        audio=work/"narration"/(scene["key"]+".mp3")
        scene["audio_seconds"]=duration(ffmpeg,audio)
        scene["frames"]=math.ceil(max(scene["minimum"],scene["audio_seconds"]+1.0)*FPS)
        scene["seconds"]=scene["frames"]/FPS
        scene["start"]=time
        meta=audio.with_suffix(".jsonl")
        scene["captions"]=[json.loads(line) for line in meta.read_text().splitlines() if line]
        assert scene["captions"]
        time+=scene["seconds"]
    return time


def caption(scene, seconds):
    for item in scene["captions"]:
        a=item["offset"]/10_000_000
        b=(item["offset"]+item["duration"])/10_000_000
        if a-.08<=seconds<b+.1:
            return item["text"]
    return ""


def base(scene, total, seconds):
    im=Image.new("RGB",(W,H),BG)
    d=ImageDraw.Draw(im)
    text(d,(70,38),"COSMOS3 POLICY / 实验讲解",29,MUTED,bold=True)
    text(d,(1350,38),scene["chapter"],29,BLUE)
    text(d,(70,107),scene["title"],58,bold=True)
    d.line((70,201,1850,201),fill="#dbe4f0",width=2)
    progress=(scene["start"]+seconds)/total
    d.rectangle((0,H-7,int(W*progress),H),fill=BLUE)
    text(d,(70,1026),"现有社区权重 · 当前程序 · 固定场景，非总体成功率 · 原始相机256×256，讲解画布1080p",24,MUTED)
    return im,d


def subtitle(im, scene, seconds):
    value=caption(scene,seconds)
    if value:
        d=ImageDraw.Draw(im)
        box(d,(60,877,1860,998),"#17263c",radius=16)
        lines=wrap(value,37,1690)
        assert len(lines)<=2,(scene["key"],value,lines)
        for i,line in enumerate(lines):
            text(d,((W-font(37).getlength(line))/2,899+i*47),line,37,"white")


class Renderer:
    def __init__(self, ffmpeg, metrics):
        self.metrics=metrics
        self.clips={s["key"]:decode_clip(ffmpeg,ROOT/s["source"]/"actual.mp4") for s in SCENES if s["kind"]=="clip"}
        self.input=Image.open(ROOT/"experiments/09-system-prompt/closed-loop/center_milk_off/input_00.png").convert("RGB")
        self.goal=Image.open(ROOT/"experiments/15-native-goal-targets/inputs/reference/input.png").convert("RGB")

    @lru_cache(maxsize=28)
    def clip_image(self,key,index):
        return Image.fromarray(self.clips[key][index]).resize((1240,620),Image.Resampling.LANCZOS)

    def frame(self,scene,seconds,total):
        im,d=base(scene,total,seconds)
        kind=scene["kind"]
        p=seconds/scene["seconds"]
        if kind=="clip":
            # One complete archived clip, uniformly slowed; no reordered/cut frames.
            idx=min(128,int(p*129))
            im.paste(self.clip_image(scene["key"],idx),(70,249))
            text(d,(82,213),"固定相机",26,MUTED)
            text(d,(709,213),"手腕相机",26,MUTED)
            box(d,(1340,245,1850,858),"white")
            text(d,(1370,279),"这一条的指令",29,MUTED)
            yy=paragraph(d,(1370,336),scene["prompt"],39,450,BLUE)
            d.line((1370,yy+25,1820,yy+25),fill="#dbe4f0",width=2)
            y=yy+59
            for item in scene["notes"]:
                y=paragraph(d,(1370,y),item,34,446,gap=12)+30
            if scene["key"]=="04_swap":
                text(d,(1370,706),"原位时同样交换，仍抓牛奶。",24,MUTED)
                text(d,(1370,743),"四种布局×三个噪声种子。",24,MUTED)
                text(d,(1370,780),"每组抓取对象一致。",24,MUTED)
            text(d,(1380,815),f"原录像记录 {idx:03d} / 128",25,MUTED)
            # Label reflects the currently displayed archived contact state.
            trajectory=self.trajectories[scene["key"]]
            names=trajectory[idx].get("grasped",[])
            if names:
                box(d,(90,271,581,325),"white",radius=8)
                text(d,(110,282),"当前双侧接触："+scene["selected"],28,scene["color"],bold=True)
            text(d,(1347,213),"MuJoCo实际执行 · 放慢播放",25,MUTED)
        elif kind=="intro":
            im.paste(self.input.resize((540,270),Image.Resampling.LANCZOS),(70,321))
            text(d,(77,277),"输入：两路相机＋一句话",34,bold=True)
            box(d,(730,325,1175,590),"white")
            text(d,(790,386),"同一个模型",49,BLUE,bold=True)
            text(d,(790,477),"计算一段动作",38)
            frame=Image.fromarray(self.clips["01_milk"][90]).resize((540,270),Image.Resampling.LANCZOS)
            im.paste(frame,(1310,321))
            text(d,(1317,277),"输出：动作在仿真中执行",34,bold=True)
            arrow(d,(628,450),(711,450),progress=seconds/2)
            arrow(d,(1193,450),(1290,450),progress=seconds/2)
            text(d,(102,686),"我们改变输入，观察它实际抓谁。",49,bold=True)
            text(d,(102,763),"下文录像均为MuJoCo实际执行；流程动画为讲解示意。",32,MUTED)
        elif kind=="arrays":
            box(d,(70,255,680,830),"white")
            text(d,(108,295),"怎么比较？",42,bold=True)
            for y,value in [(393,"① 画面保持相同"),(481,"② 起始随机噪声相同"),(569,"③ 只换牛奶／黄油"),(657,"④ 保存并比较真实数组")]:
                text(d,(112,y),value,35,BLUE if y==569 else INK)
            text(d,(750,270),"动作头前的数组，相对变化",39,bold=True)
            for j,(label,value) in enumerate([("第1次计算",self.metrics["first"]),("第30次计算",self.metrics["last"]) ]):
                y=365+j*175
                text(d,(754,y),label,35)
                d.rounded_rectangle((983,y,1760,y+70),radius=10,fill="#e4ebf7")
                fraction=min(1,max(0,(seconds-j*2-1)/3))
                width=777*value/10*fraction
                if width>1:d.rounded_rectangle((983,y,983+width,y+70),radius=10,fill=BLUE)
                text(d,(983,y+87),f"{value:.2f}%",41,BLUE,bold=True)
            box(d,(752,733,1835,837),"#fff0df")
            text(d,(785,763),"数组变了 ≠ 听懂了这么多",40,ORANGE,bold=True)
            text(d,(755,693),"同一段动作预测；原位，三个种子平均；实验07",27,MUTED)
        elif kind=="routes":
            box(d,(100,401,430,596),"white")
            text(d,(177,463),"文字数据",46,BLUE,bold=True)
            box(d,(1360,401,1810,596),"white")
            text(d,(1418,448),"动作内部数据",43,bold=True)
            text(d,(1444,519),"数字会改变",36,BLUE)
            arrow(d,(451,447),(1338,447),progress=seconds/2)
            text(d,(731,352),"直接读取文字",36,BLUE)
            box(d,(670,647,1220,768),"#e4edff")
            text(d,(700,682),"预测画面的内部数据",40,BLUE)
            arrow(d,(430,570),(650,705),progress=seconds/2)
            arrow(d,(1240,705),(1390,602),progress=seconds/2)
            text(d,(470,270),"两种读取方式，都能影响动作数字",46,bold=True)
            text(d,(119,816),"六条实际执行：仍未改抓黄油。",39,ORANGE,bold=True)
            text(d,(117,231),"信息读取示意；并非两个独立模型，效应不能相加。",27,MUTED)
        elif kind=="goal_input":
            im.paste(self.goal.resize((1140,570),Image.Resampling.LANCZOS),(70,274))
            box(d,(1250,278,1850,833),"white")
            text(d,(1290,319),"两次起点完全相同",40,bold=True)
            paragraph(d,(1290,414),"第一次：把碗放到柜子上",39,510,BLUE)
            paragraph(d,(1290,566),"第二次：把酒瓶放到柜子上",39,510,BLUE)
            paragraph(d,(1290,736),"动起来后，各自拍新照片继续算动作。",28,510,MUTED)
            text(d,(78,221),"真正送入模型的共同初始画面",32,MUTED)
        elif kind=="conclusion":
            box(d,(70,269,935,751),"white")
            box(d,(985,269,1850,751),"white")
            text(d,(110,313),"已经观察到",45,GREEN,bold=True)
            text(d,(1025,313),"仍然没有证明",45,ORANGE,bold=True)
            left=["摆放会影响这组测试抓谁", "语言会影响内部计算", "特定场景能按指令换对目标"]
            right=["它背了某条训练答案", "哪一层是唯一出错原因", "能在各种场景稳定完成任务"]
            for j,(a,b) in enumerate(zip(left,right)):
                y=422+j*101
                if seconds>j*1.8:
                    text(d,(113,y),str(j+1)+". "+a,35)
                    text(d,(1028,y),str(j+1)+". "+b,35)
            text(d,(93,791),"下一步：只换指定内部数据，再看实际抓谁。",43,BLUE,bold=True)
        subtitle(im,scene,seconds)
        return im


def timestamp(seconds):
    ms=round(seconds*1000)
    h,ms=divmod(ms,3_600_000);m,ms=divmod(ms,60_000);s,ms=divmod(ms,1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def render(args, work, total):
    metrics=evidence()
    rr=Renderer(args.ffmpeg,metrics)
    rr.trajectories={s["key"]:json.loads((ROOT/s["source"]/"trajectory.json").read_text()) for s in SCENES if s["kind"]=="clip"}
    assert all(len(x)==129 for x in rr.trajectories.values())
    silent=work/"silent.mp4"
    log=(work/"render.log").open("w")
    proc=subprocess.Popen([args.ffmpeg,"-y","-v","warning","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r",str(FPS),"-i","pipe:0","-an","-c:v","libx264","-preset","veryfast","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(silent)],stdin=subprocess.PIPE,stderr=log)
    emitted=0
    for scene in SCENES:
        for f in range(scene["frames"]):
            im=rr.frame(scene,f/FPS,total)
            proc.stdin.write(im.tobytes())
            emitted+=1
            if f==round(scene["frames"]*.55):
                im.save(work/(scene["key"]+".png"))
            if emitted%(FPS*5)==0:print(f"render {emitted/FPS:.0f}/{total:.1f}s",flush=True)
    proc.stdin.close();assert proc.wait()==0
    log.close()
    # Decode/pad each narration independently; silence fills the scene tail.
    cmd=[args.ffmpeg,"-y","-v","warning","-i",str(silent)]
    for scene in SCENES:cmd += ["-i",str(work/"narration"/(scene["key"]+".mp3"))]
    filters=[]
    for i,scene in enumerate(SCENES):
        filters.append(f"[{i+1}:a]apad,atrim=duration={scene['seconds']:.9f},asetpts=PTS-STARTPTS[a{i}]")
    filters.append("".join(f"[a{i}]" for i in range(len(SCENES)))+f"concat=n={len(SCENES)}:v=0:a=1[aout]")
    cmd += ["-filter_complex",";".join(filters),"-map","0:v:0","-map","[aout]","-c:v","copy","-c:a","aac","-b:a","128k","-movflags","+faststart","-t",str(total),str(args.output)]
    subprocess.run(cmd,check=True)
    srt=[];n=0
    for scene in SCENES:
        for cap in scene["captions"]:
            n+=1
            a=scene["start"]+cap["offset"]/10_000_000
            b=a+cap["duration"]/10_000_000
            srt.append(f"{n}\n{timestamp(a)} --> {timestamp(b)}\n{cap['text']}\n")
    args.output.with_suffix(".srt").write_text("\n".join(srt),encoding="utf-8")
    sources=[ROOT/"experiments/07-language-path/result.json", ROOT/"experiments/10-identity-position/summary.json", ROOT/"experiments/15-native-goal-targets/policy/closed-loop/result.json"]
    sources += [ROOT/s["source"]/"actual.mp4" for s in SCENES if s["kind"]=="clip"]
    report={"seconds":total,"frames":emitted,"canvas":[W,H],"fps":FPS,"original_camera_resolution":[256,256],"narration":"Synthetic Chinese voice; subtitles follow sentence timing.","sources":[{"path":str(p.relative_to(ROOT)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],"scenes":SCENES,"scope":"Animation explains archived observations, not a new trial or a claimed root cause. Real clips replay all 129 frames in order, uniformly slowed. Diagram motion is explanatory, not latent coordinates.","output_sha256":hashlib.sha256(args.output.read_bytes()).hexdigest()}
    (work/"render_complete.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("complete",str(args.output),f"{total:.2f}s",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--ffmpeg",required=True)
    parser.add_argument("--work",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=ROOT/"docs/media/experiment-summary.mp4")
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    args.work.mkdir(parents=True,exist_ok=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    asyncio.run(prepare_voice(args.work))
    total=build_timeline(args.ffmpeg,args.work)
    print("planned duration",round(total,2),flush=True)
    if not args.prepare_only:render(args,args.work,total)


if __name__=="__main__":main()
