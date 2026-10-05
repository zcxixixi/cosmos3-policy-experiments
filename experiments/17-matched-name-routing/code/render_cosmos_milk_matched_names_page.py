"""Render the latest completed experiment using its audited, actual data."""
from pathlib import Path
import ast
import hashlib
import html
import json
import re
import shutil
import struct
import zipfile

REPO = Path('/home/cenxi/Documents/cosmos3-policy-experiments')
EXP = REPO / 'experiments/17-matched-name-routing'
LINK = 'https://github.com/zcxixixi/cosmos3-policy-experiments/blob/main/experiments/17-matched-name-routing/'
GOALS = {'milk_box': '牛奶盒', 'cream_cheese': '奶酪盒'}
ARMS = {'native': '原生网络', 'all_allowed': '连接全保留对照', 'full_hardmask': '前9层切边'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_exact(source, target):
    if target.exists():
        assert digest(source) == digest(target), str(target)
    else:
        shutil.copyfile(source, target)
    assert digest(source) == digest(target)


def boolean_array(archive, key):
    raw = archive.read(key + '.npy')
    assert raw[:6] == b'\x93NUMPY'
    version = tuple(raw[6:8])
    offset = 10 if version == (1, 0) else 12
    length = struct.unpack('<H' if offset == 10 else '<I', raw[8:offset])[0]
    header = ast.literal_eval(raw[offset:offset + length].decode())
    assert header == {'descr': '|b1', 'fortran_order': False, 'shape': (120, 120)}
    data = raw[offset + length:]
    assert len(data) == 14400
    return data


def main():
    analysis = json.loads((EXP / 'matched-names-analysis/analysis.json').read_text())
    complete = json.loads((EXP / 'matched-names-analysis/complete.json').read_text())
    assert complete['state'] == 'complete' and complete['CUDA_initialized'] is False
    for name, expected in complete['files_sha256'].items():
        assert digest(EXP / 'matched-names-analysis' / name) == expected
    rows = analysis['physical_cases']
    assert len(rows) == 24 and analysis['physical_counts']['actual_model_records'] == 192
    assert analysis['physical_counts']['actual_converted_actions'] == 3072
    assert analysis['primary']['prediction'] == 'rejected'
    media = REPO / 'docs/media'
    for row in rows:
        copy_exact(EXP / 'future-matched-names-execution/closed-loop' / row['case'] / 'actual.mp4',
                   media / ('matched-' + row['case'] + '.mp4'))
    for scene in ('x12', 'x15'):
        copy_exact(EXP / 'future-matched-names-execution/closed-loop' / f'{scene}_milk_box_V195_A195_native/input_00.png',
                   media / f'matched-{scene}-input.png')
    for name in ('L35-offline-target-reading.png', 't0-language-versus-scene.png'):
        copy_exact(EXP / 'matched-names-analysis' / name, media / ('matched-' + name))

    # The blue cells represent measured byte equality, not invented activity.
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="940" height="470" viewBox="0 0 940 470" role="img">',
           '<rect width="940" height="470" fill="white"/><g font-family="sans-serif" fill="#253143">',
           '<text x="24" y="34" font-size="22">同一句指令：末层文字后缀 K/V 是否改变？</text>',
           '<text x="24" y="62" font-size="15">每格＝一次真实去噪，与该指令的第一份捕获逐字节比较。蓝色＝完全相同。</text>']
    with zipfile.ZipFile(EXP / 'matched-names-analysis/arrays.npz') as archive:
        for gi, goal in enumerate(GOALS):
            values = boolean_array(archive, goal + '_same_name_suffix_allpair_KV_byte_equal')
            assert all(values), 'Conclusion must follow the full actual matrix.'
            for ci, (scene, seed) in enumerate((('12厘米', 195), ('12厘米', 198), ('15厘米', 195), ('15厘米', 198))):
                y = 90 + gi * 168 + ci * 34
                svg.append(f'<text x="24" y="{y + 20}" font-size="15">{GOALS[goal]} · {scene} · 噪声{seed}</text>')
                for step in range(30):
                    equal = bool(values[ci * 30 + step])
                    color = '#2862ab' if equal else '#ca4c32'
                    svg.append(f'<rect x="{330 + step * 18}" y="{y}" width="15" height="24" fill="{color}"><title>{html.escape(goal)}，{scene}，噪声{seed}，去噪{step}，K/V逐字节相同={equal}</title></rect>')
    svg.extend(['<text x="330" y="398" font-size="14">去噪0</text><text x="815" y="398" font-size="14">去噪29</text>',
                '<text x="24" y="433" font-size="17">牛奶盒：120/120相同；奶酪盒：120/120相同。两条不同指令之间的K/V则不同。</text>',
                '<text x="24" y="457" font-size="14">仅检验L35保存的74行UND后缀，不代表全部网络或全部语言表示。</text></g></svg>'])
    (media / 'matched-language-supply.svg').write_text(''.join(svg))
    style = re.search(r'<style>(.*?)</style>', (REPO / 'docs/index.html').read_text(), re.S)[1]
    cases = {r['case']: dict(scene=r['scene'], goal=GOALS[r['goal']], V=r['V'], arm=ARMS[r['arm']],
                            selected='、'.join({'milk_1': '牛奶', 'cream_cheese_1': '奶酪盒'}.get(x, x) for x in r['selected_objects']) or '未抓起',
                            start=r['first_selection_step']) for r in rows}
    detail_rows = ''.join(f'<tr><td>{r["scene"][1:]}厘米</td><td>{GOALS[r["goal"]]}</td><td>{r["V"]}</td><td>{ARMS[r["arm"]]}</td><td>{cases[r["case"]]["selected"]}</td><td>{r["first_selection_step"]}–{r["first_selection_step"] + 4}</td></tr>' for r in rows)
    page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cosmos3：文字状态还在，为什么仍抓错？</title><style>{style}</style></head><body><main>
<section id="latest"><p class="muted">2026-10-05 · 最新完成：24条真实执行＋720次内部捕获＋只读数组核验</p>
<h1>文字状态还在，动作却不一定按它选目标。</h1>
<div class="goal"><b>研究问题没有变：</b>同一句“抓牛奶”，为什么牛奶移远后，机械臂转抓奶酪盒？我们要找到具体内部通路，并让它继续跟着牛奶换位置。</div>
<p><b>本轮排除了一个干扰：</b>用“milk box”和“cream cheese”两种真实名称，指令长度和目标词位置完全一样，只改变目标名称。因此，这轮的语言对照没有同时改变token数量与位置。</p>
<div class="status"><b>最新内部发现：</b>同一句话在12／15厘米、两份噪声、30次去噪中，末层保存的文字后缀K/V共120份，<b>逐字节完全相同</b>。抓谁却会改变。出错不能归因于这段文字向量随这些条件消失；读取方式、其他输入或后续计算仍待定位。</div>
<p class="muted">这是社区权重、两个初态与两个配对噪声的结果，不能当作总体成功率。原始“milk”指令在15厘米的失败仍未修好。</p></section>
<section><h2>1. 实际抓谁？</h2><p>照片、机器人初态、动作初始噪声保持配对。表中先看未经内部干预的网络。</p>
<div class="table-scroll"><table><thead><tr><th>牛奶位置</th><th>未来生成噪声</th><th>指令：抓牛奶盒</th><th>指令：抓奶酪盒</th></tr></thead><tbody>
<tr><td>移动12厘米</td><td>195</td><td>奶酪盒 ❌</td><td>奶酪盒 ✅</td></tr><tr><td>移动12厘米</td><td>198</td><td>牛奶 ✅</td><td>奶酪盒 ✅</td></tr>
<tr><td>移动15厘米</td><td>195</td><td>奶酪盒 ❌</td><td>奶酪盒 ✅</td></tr><tr><td>移动15厘米</td><td>198</td><td>牛奶 ✅</td><td>奶酪盒 ✅</td></tr></tbody></table></div>
<p><b>“完全不读语言”解释不了全部结果。</b>噪声198下，只换名称就会换抓取对象。但换成195，同样的指令又抓错；语言没有稳定控制目标。</p>
<p>还复测了前9层“未来视觉→当前视觉”切边：195下，它能让12厘米条件抓回牛奶，却救不了15厘米条件。因此，之前的内部改动不是通用修复。连接全保留对照的对象分类与原生全部一致。</p>
<div class="video-picker"><label>牛奶位置<select id="scene"><option value="x12">移动12厘米</option><option value="x15" selected>移动15厘米</option></select></label><label>指令目标<select id="goal"><option value="milk_box">牛奶盒</option><option value="cream_cheese">奶酪盒</option></select></label><label>网络条件<select id="arm"><option value="native">原生网络</option><option value="all_allowed">连接全保留对照</option><option value="full_hardmask">前9层切边</option></select></label><label>未来生成噪声<select id="seed"><option value="195">195</option><option value="198">198</option></select></label></div>
<video id="execution" controls preload="metadata" playsinline src="media/matched-x15_milk_box_V195_A195_native.mp4"></video><p id="video-caption" class="muted" aria-live="polite"></p>
<p class="muted">一次只播放一条MuJoCo录像。“抓起”要求双侧指垫接触、抬高超过2厘米、连续5条记录；不等于完成放篮任务。所有24条件都执行，没有挑最好的一条。</p>
<details><summary>全部24条结果</summary><div class="table-scroll"><table><thead><tr><th>位置</th><th>指令</th><th>噪声</th><th>内部条件</th><th>实际抓起</th><th>合格5条窗口</th></tr></thead><tbody>{detail_rows}</tbody></table></div></details></section>
<section id="functional"><h2>2. 这次在网络里面查到了什么？</h2>
<div class="flow"><span>输入目标名称</span><i>→</i><span>末层文字K/V<br>同指令120份相同</span><i>→</i><span>视觉／动作状态读取它<br>具体问题待定位</span><i>→</i><span>动作输出与实际抓取</span></div>
<p>K/V可以先理解成：<b>供其他状态读取的文字数组</b>。这段存的是文字侧状态，物体在哪里还需要结合视觉与动作侧的计算。相同指令的这段数组保持固定，符合这条文字支路的结构；不能据此说整个网络没变。</p>
<figure><img src="media/matched-language-supply.svg" alt="实际240份末层文字K/V核验：牛奶盒和奶酪盒各120份，相同指令跨位置、噪声、去噪逐字节相同"><figcaption>直接来自真实字节比较。蓝色表示完全相同，不是活动强弱，也不是人工画出来的语言脑区。</figcaption></figure>
<div class="result-note"><b>这条结果怎样缩小问题？</b>不能指望把同一句话的“好案例文字K/V”搬给坏案例来修复这一接口——这里本来就是相同数组。下一步应改变它被读取的方式或下游响应，而不是继续重复“语言向量变了多少”。</div>
<details><summary>看真实各层差异与末层读取图</summary><figure><img loading="lazy" src="media/matched-t0-language-versus-scene.png" alt="真实首个去噪时刻，37处文字、当前视觉和动作状态的语言对照与位置对照RMS"><figcaption>左列换目标名称，右列把牛奶从12厘米移到15厘米。纵轴是实际数组的绝对RMS差；不能当成理解程度或跨层信息保留率。</figcaption></figure>
<figure><img loading="lazy" src="media/matched-L35-offline-target-reading.png" alt="末层32个注意力头对两个目标词的离线读取估计"><figcaption>由保存的真实Q/K用Float64重新计算，展示首个去噪时刻、16个动作位置×32个head。它不是记录下来的BF16注意力权重，也不是因果重要性。名称信息还会进入目标词之后的文字状态，不能只看两个词的权重就断言不读语言。</figcaption></figure>
<p>核验覆盖720份真实内部边界、720份末层dispatch片段、240份原生语言后缀、192份实际执行预测与3072条控制命令。末层为零编号L35；后缀为UND第48～121行，共74行。不同目标名称的K/V确实不同；相同名称跨条件保持一致。目标词之前的完整K/V只捕获了首个去噪时刻，因此不把后缀结论扩展到未核验部分。</p>
<p><a href="{LINK}matched-names-analysis/arrays.npz">真实NPZ数组</a> · <a href="{LINK}matched-names-analysis/analysis.json">完整分析</a> · <a href="{LINK}matched-names-analysis/complete.json">分析封存</a></p></details></section>
<section><h2>3. 下一项局部实验：末层怎样处理读进来的目标信息？</h2>
<p><b>猜想：</b>动作状态读到另一目标的信息后，末层MLP的更新可能把它拉回原来的动作方向。要检验这个猜想，就比较“换入另一指令的文字K/V”和“同样换入，但固定住MLP对这次替换的响应”。</p>
<p>准备做双向替换：牛奶盒→奶酪盒、奶酪盒→牛奶盒，并加入自身替换与相同强度随机方向对照。只改末层供16个动作位置读取的部分，再把真实动作交给MuJoCo。</p>
<div class="status"><b>这项尚未运行。</b>只使用噪声198中原生能按两个名称抓对目标的配对案例。即使成功，也只是定位该配对中的局部作用；原始milk／15厘米、噪声195的失败，以及撤去刺激后能否维持，仍必须回测。</div>
<p><b>目前结论：</b>排除了输入长度混杂和这段语言后缀随位置／噪声丢失的解释；还没有定位具体出错head或MLP，也没证明模型背了某条训练动作。</p></section>
<footer><p>仅研究社区权重 <a href="https://huggingface.co/fwd4xl/cosmos3-nano-policy-liberoall-5k">cosmos3-nano-policy-liberoall-5k</a>，未新增训练。<a href="https://github.com/zcxixixi/cosmos3-policy-experiments/tree/main/experiments/17-matched-name-routing">本轮代码、数据和全部录像</a>。</p>
<details><summary>复现范围与方法依据</summary><p>公开全部24条物理轨迹、192份真实执行预测；全部240份原生末层稀疏Q/K/V与动作注意力输出片段；4份15厘米原生首时刻的完整逐层边界；以及全部720个末层动作状态的原始BF16位模式NPZ。其他完整内部PT保留A6000，发布索引与SHA。输入目录符号链接未发布，也未伪造合并输入目录。</p>
<p><a href="{LINK}publication-manifest.json">发布文件清单与SHA</a> · <a href="{LINK}code/probe_cosmos_milk_matched_names.py">内部捕获代码</a> · <a href="{LINK}code/run_cosmos_milk_matched_names.py">实际执行代码</a> · <a href="{LINK}code/analyze_cosmos_milk_matched_names.py">只读数组核验代码</a>。</p>
<p>局部替换及对照参考 <a href="https://arxiv.org/abs/2309.16042">Towards Best Practices of Activation Patching</a>、<a href="https://arxiv.org/abs/2404.15255">How to use and interpret activation patching</a>；这些文章提供方法依据，没有证明本模型的机制。</p>
<p>实际指令：<code>pick up the milk box and place it in the basket</code>／<code>pick up the cream cheese and place it in the basket</code>。均11个prompt token，完整UND122／GEN266／joint388；只有两个目标token改变，其他输入排列保持配对。未来噪声195／198，动作与运行时195。首轮30次去噪输出16步动作；后续7轮按各自指令正常推理。</p></details></footer></main>
<script>const cases={json.dumps(cases, ensure_ascii=False)};
const scene=document.getElementById('scene'),goal=document.getElementById('goal'),arm=document.getElementById('arm'),seed=document.getElementById('seed'),video=document.getElementById('execution'),caption=document.getElementById('video-caption');
function show(change){{const key=`${{scene.value}}_${{goal.value}}_V${{seed.value}}_A195_${{arm.value}}`,r=cases[key];if(change){{video.pause();video.src=`media/matched-${{key}}.mp4`;video.load();}}caption.textContent=`牛奶移动${{r.scene.slice(1)}}厘米；指令：抓${{r.goal}}；${{r.arm}}；噪声${{r.V}}；实际抓起：${{r.selected}}（合格窗口${{r.start}}–${{r.start+4}}）。这是MuJoCo实际执行。`;video.setAttribute('aria-label',caption.textContent);}}
for(const select of [scene,goal,arm,seed])select.addEventListener('change',()=>show(true));show(false);</script></body></html>'''
    (REPO / 'docs/index.html').write_text(page)
    print('Rendered 24 real videos and the audited latest page.')


if __name__ == '__main__':
    main()
