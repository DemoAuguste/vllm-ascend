# SAGE 鍙傝€冨疄鐜帮紙experiment/SAGE-k3锛?
- 璁烘枃锛?SAGE: Pushing 2-Bit Vector-Quantized LLMs toward Real-World Usability*锛坅rXiv:2610.14745v1, SJTU锛?- 瀹樻柟瀹炵幇锛歨ttps://github.com/Zincwolf/sage 锛圙PL-3.0锛孡inux + NVIDIA + CUDA 12.8锛?- 瀹氫綅锛?*SAGE 鐨勭嫭绔嬪弬鑰冨疄鐜帮紙NPU 鍙嬪ソ锛?*锛屼緵 Qwen3.8-27B 娣峰悎绮惧害閲忓寲锛圫AGE 浣庝綅妗?+ GPTQ-INT 楂樹綅妗?+ SLQ 鍒嗛厤锛夊鐢ㄣ€傛柟娉曞彲琛屾€ц [`../SAGE-feasibility.md`](../../docs/SAGE-feasibility.md)銆?
> 瀹樻柟瀹炵幇渚濊禆 `paroquant` 瀛愭ā鍧?+ 5 涓?git patch + CUDA kernel锛屾棤娉曠洿鎺ョЩ妞嶅埌 Ascend銆?> 鏈疄鐜版寜璁烘枃 Algorithm 1 / 搂3 鍏紡涓庡畼鏂规簮鐮佽涔?*浠庨浂閲嶅啓**锛屼繚鎸佺畻娉曠瓑浠凤紝骞跺湪涓嬫枃鏍囨敞涓庡畼鏂圭殑宸紓銆?
---

## 1. 涓€鍙ヨ瘽缁撹

SAGE = 鐞冨潗鏍囪В鑰?VQ锛堢鍙锋樉寮?+ 闈炶礋鏂瑰悜鐮佹湰 + 鍏变韩鍗婂緞缃戞牸锛壝?**ESG 鑳介噺鍒嗗眰鍒嗙粍** 脳 **ABS 鑷€傚簲浣嶅垏鍒?* 脳 **blockwise 璇樊琛ュ伩**銆?缂栫爜杞借嵎 **2.25 bpw**锛坄(v+log鈧侹+V)/v`锛夈€傛湰瀹炵幇鍚?CPU demo/鍗曟祴涓?NPU 鍐掔儫鑴氭湰锛涙湰鍦?CPU 宸查獙璇?13 椤规祴璇曢€氳繃銆乥lockwise 浣跨鍒扮 MSE 闄嶄綆绾?43%銆?
---

## 2. 鏂规硶 鈫?瀹炵幇鏄犲皠

| 璁烘枃瑕佺礌 | 鍏紡 | 瀹炵幇浣嶇疆 |
|---|---|---|
| 鐞冨潗鏍囧垎瑙?`w 鈫?(s, p, r)` | 搂3.1, Eq.1 | [`sage/spherical.py`](sage/spherical.py) `decompose` / `reconstruct_vectors` |
| 闈炶礋鍗曚綅鐮佹湰瑙掑弬鏁板寲 | Eq.12-13 | [`sage/spherical.py`](sage/spherical.py) `angles_to_codewords` / `codewords_to_angles` |
| 鍏变韩鍗婂緞缃戞牸閲忓寲 | Eq.2/14 | [`sage/spherical.py`](sage/spherical.py) `radius_grid_quantize` |
| ESG 鑳介噺鍒嗗眰浜ら敊缃崲 | Eq.5 | [`sage/grouping.py`](sage/grouping.py) `esg_permutation` / `energy_weights` |
| 鍔犳潈鐞冮潰 k-means | Eq.6 | [`sage/codebook.py`](sage/codebook.py) `weighted_spherical_kmeans` / `weighted_assignment` |
| ABS 閫愮煩闃甸€夊垏鍒?| Eq.7-9 | [`sage/quantizer.py`](sage/quantizer.py) `SAGEQuantizer.prepare` |
| blockwise 璇樊琛ュ伩 | 搂3.4, Eq.10 | [`sage/blockwise.py`](sage/blockwise.py) `blockwise_optimize` |
| bpp 瑙ｆ瀽 | Eq.3/15 | [`sage/quantizer.py`](sage/quantizer.py) `SAGEQuantizer.bpp` |

鍏抽敭鍏紡锛?```
姣忓瓙鍚戦噺 bit:  B = v + log鈧侹 + V            (v=8, 榛樿 (K,V)=(64,4) 鈫?18 bit)
payload bpw:   BPW = B / v = 1 + (log鈧侹+V)/8   鈫?2.25 bpw
鍗婂緞缃戞牸:       碌_g = min r虄,  未_g = (max r虄 鈭?碌_g)/(2^V 鈭?1),  r虃 = 碌_g + q路未_g
鏂瑰悜璧嬪€?       z = argmax_k (危 蠅 p c)虏 / (危 蠅 c虏)      (蠅=1 鏃堕€€鍖栦负浣欏鸡)
鍔犳潈:           蠅_i 鈭?e_i^纬  (纬=0.75, 瀛愬悜閲忓唴鍗曚綅鍧囧€?
```

---

## 3. 鐩綍缁撴瀯

```
SAGE-k3/
  sage/
    config.py     SAGEConfig / Split锛圓BS 鍊欓€変笌瓒呭弬锛?    utils.py      device 瑙ｆ瀽銆佸垎鍧?argmax銆乻oftplus 閫嗐€佹棩蹇楋紙NPU 鍙嬪ソ锛?    spherical.py  鐞冨潗鏍囧垎瑙?閲嶅缓銆佽鍙傛暟鍖栥€佸叡浜崐寰勭綉鏍?    grouping.py   閫氶亾鑳介噺浼拌 + ESG 浜ら敊缃崲 + 鍔犳潈
    codebook.py   鍔犳潈鐞冮潰 k-means锛堥潪璐熸浜よ薄闄愶級
    quantizer.py  SAGEQuantizer锛氭嫙鍚?/ ABS / 缂栬В鐮?/ bpp
    blockwise.py  閫愬眰 blockwise 璇樊琛ュ伩锛圫TE锛?  tests/test_sage.py   13 椤瑰崟娴嬶紙鍚ā鍨嬬骇闆嗘垚锛?  demo.py              CPU 鏈€灏忓彲楠岃瘉 demo锛圓鈥揊锛?  model_quant.py       鍏ㄦā鍨?fake-quant锛圕PU/NPU锛?  ppl_eval.py          WikiText-2 PPL 瀵规瘮
  npv_verify.py        NPU 鏁板€间竴鑷存€?+ 鎬ц兘
  README.md            鏈枃浠?```

---

## 4. 蹇€熷紑濮?
```bash
# 渚濊禆锛歱ython>=3.9, torch(CPU 鍗冲彲), numpy锛涳紙妯″瀷绾ц剼鏈彟闇€ transformers銆乨atasets锛?cd experiment/SAGE-k3

# 1) 鍗曞厓娴嬭瘯锛圕PU锛?python -m pytest -q            # 鏈熸湜 13 passed

# 2) 鏁板€?Demo锛圕PU锛?python demo.py

# 3) 鍏ㄦā鍨?fake-quant 鍐掔儫锛圕PU锛岀ず渚嬶級
python model_quant.py --device cpu --model Qwen/Qwen3-0.6B \
    --calib-seqs 4 --calib-len 256 --kmeans-iters 8 --epochs 5

# 4) NPU 鐜楠岃瘉锛圢PU 鏈嶅姟鍣級
ASCEND_RT_VISIBLE_DEVICES=5 python npv_verify.py

# 5) NPU 涓婅窇 0.6B PPL锛堝浐瀹?2.0bpw 鍗曞垏鍒?/ ABS 涓夊垏鍒嗭級
ASCEND_RT_VISIBLE_DEVICES=5 python ppl_eval.py --device npu --splits 64,2 --epochs 20
ASCEND_RT_VISIBLE_DEVICES=5 python ppl_eval.py --device npu --splits 64,4\;128,3\;256,2
```

`model_quant.py` 甯哥敤寮€鍏筹細`--no-esg`锛堟秷铻嶏級銆乣--no-adaptive-bits`锛堝浐瀹氬垏鍒嗭級銆乣--sequential`锛堝簭璐鐢熻緭鍏ワ級銆乣--only mlp|attn`锛堝垎缁勬秷铻嶏級銆乣--limit-layers N`锛堝揩閫熷啋鐑燂級銆乣--calib-text <file>`锛堟湰鍦版枃鏈牎鍑嗭紝鍏?datasets锛夈€?
---

## 5. NPU 瀹炵幇涓庢€ц兘璁捐

瀵归綈 `AGENTS.md` 鐨?NPU 鎬ц兘瑕佹眰锛?
1. **璁惧鏃犲叧 + 鎯版€?`torch_npu`**锛歚sage/utils.py:resolve_device` 浠呭湪鎸囧畾 NPU 鏃舵墠瀵煎叆 `torch_npu`锛汣PU 鐜闆朵緷璧栵紝渚夸簬鏈湴寮€鍙?CI銆?2. **瑙勯伩 `.item()` 鐑矾寰勫悓姝?*锛氱紪鐮併€乲-means銆佸崐寰勯噺鍖栧叏绋嬪紶閲忓寲锛涙敹鏁涘垽鎹瘡杞嚦澶氫竴娆℃爣閲忓悓姝ワ紙k-means 20 杞級銆侾CDVQ 涓?NPU 涓?2^a 娆?`.item()` 鐏鹃毦"鐨勯棶棰樺湪鏈疄鐜颁笉瀛樺湪銆?3. **澶х煩闃靛垎鍧?*锛氭柟鍚戣祴鍊?`[N,8]@[8,K]` 鎸?N 鍒嗗潡锛坄chunk`锛夛紝闃?27B 鍗曞眰锛圢鈮?1M锛変腑闂寸煩闃?OOM锛沗npv_verify.py` 宸茶鐩?27B MLP 瑙勬ā鍚炲悙娴嬭瘯銆?4. **鐮佹湰灏哄鍙帶**锛欿鈮?56 甯搁┗灏忓紶閲忥紝璐磋繎瀹樻柟 shared-memory 鏌ヨ〃瑙ｇ爜褰㈡€併€?5. **blockwise 鍏ㄥ紶閲忓寲**锛氬崐寰?STE銆佺爜鏈銆佺綉鏍奸噸绠楀潎鍚戦噺鍖栵紱姣?epoch 鍙仛涓€娆＄鏁ｈ瘎浼帮紙`discrete_mse`锛屾棤姊害锛夈€?6. **鏄惧瓨鍥炴敹**锛氶€愬眰澶勭悊鍚?`torch.npu.empty_cache()`锛涙牎鍑嗘縺娲讳笌鍏ㄧ簿搴﹁緭鍑虹紦瀛樺湪 **CPU**锛岄€愬眰鎼埌璁惧澶勭悊鍚庡啀閲婃斁銆?7. **鍙祴鎬ц兘**锛歚npv_verify.py` 鎶ュ憡缂栫爜鍚炲悙锛圡 瀛愬悜閲?s锛夈€丄BS 鎷熷悎鑰楁椂銆乥lockwise 姣忓眰鑰楁椂銆?
---

## 6. 涓庡畼鏂瑰疄鐜扮殑宸紓锛堥噸瑕侊級

| 椤?| 瀹樻柟 | 鏈疄鐜?| 褰卞搷 |
|---|---|---|---|
| 浼樺寲妗嗘灦 | `paroquant` 瀛愭ā鍧?+ git patch | 鐙珛 Python 閲嶅啓 | 鏃?CUDA 渚濊禆锛屽彲璇绘€?鍙Щ妞嶆€т紭鍏?|
| blockwise 瀛︾敓杈撳叆 | 搴忚疮锛堝墠搴忛噺鍖栧潡杈撳嚭锛?| 榛樿 FP 鎹曡幏锛沗--sequential` 搴忚疮 | 榛樿鏇村揩锛涘簭璐洿璐磋鏂囷紝鎴愭湰 O(L) 鍓嶅悜 |
| 鏁板€肩簿搴?| FP16 缃戞牸浠跨湡 + AMP | FP32 鍙傝€冨疄鐜?| 鏇寸ǔ锛岄潪閫愪綅涓€鑷?|
| 鍗婂緞缃戞牸 | 璁粌鏈?FP16 缃戞牸鍙傛暟 | FP32 缃戞牸 | bpp 璁拌处涓€鑷达紙鎸?FP16 璁″紑閿€锛?|
| codebook | 闈炶礋姝ｄ氦璞￠檺锛坰ign 澶栨彁锛?| 鍚?| 涓€鑷?|
| 瀵煎嚭 | 瀵嗛泦 QDQ + 鎵撳寘 CUDA 鏍煎紡 | 浠呭瘑闆?fake-quant checkpoint | 鎵撳寘/kernel 灞為儴缃插伐浣滐紝瓒呭嚭鏈弬鑰冭寖鍥?|
| 鏀寔妯″瀷 | qwen3 / qwen3_moe / qwen3_5 / Ministral-3 | 浠绘剰 HF `nn.Linear`锛堝彲杩囨护锛?| 鏇撮€氱敤 |

**鏈疄鐜扮殑瀹樻柟鑳藉姏**锛堟寜鏈换鍔¤寖鍥磋鍓級锛欳UDA 鎵撳寘鏍煎紡涓庤В鐮?kernel銆丮oE 涓撳鍒嗙墖銆佸垎甯冨紡锛堝鍗★級鎷熷悎銆佸帇缂╁鍑恒€?
---

## 7. 涓?PCDVQ-k3 鐨勮鎺?
| | PCDVQ-k3 | SAGE-k3锛堟湰瀹炵幇锛?|
|---|---|---|
| 鏂瑰悜鐮佹湰 | 鍥哄畾 E8 鏍?+ 璐績 FPS | 鍙涔狅紙婵€娲诲姞鏉冪悆闈?k-means锛?|
| 骞呭€?鍗婂緞 | chi-Lloyd-Max 鍏ㄥ眬鐮佹湰 | 閫愮粍鍏变韩鍧囧寑缃戞牸锛圴 bit锛?|
| 鏁版嵁鎰熺煡 | 鏃狅紙妯″瀷鏃犲叧锛?| ESG 閫氶亾鑳介噺 + 閫愮煩闃?ABS |
| 璇樊琛ュ伩 | **鏃?*锛堢函 fake-quant锛屽鑷?PPL 宕╂簝锛?| **blockwise**锛堣緭鍑?MSE + STE锛?|
| 浣嶆。 | a=14/15/16 脳 b=2 | (K,V)鈭坽(64,4),(128,3),(256,2)} 绛?|

SAGE 涓?PCDVQ 鍚屽睘"鏂瑰悜+骞呭€?褰㈢姸-澧炵泭鍒嗚В锛?*鎺ュ彛涓€鑷达紙閫愬眰 fake-quant 鏉冮噸锛?*锛屽彲鐩存帴鏇挎崲 PCDVQ 浣滀负娣峰悎绮惧害鏂规鐨?*浣庝綅妗ｇ畻瀛?*锛屽苟琛ラ綈鍏剁己澶辩殑璇樊琛ュ伩銆?
---

## 8. 鏈湴宸查獙璇佺粨鏋滐紙CPU锛孭ython 3.10 / torch 2.9.1锛?
```
python -m pytest -q        鈫?13 passed
python demo.py锛堝叧閿緭鍑猴級
  A. 鐞冨潗鏍囧垎瑙ｉ噸寤烘渶澶х粷瀵硅宸?3.7e-09
  B. 瑙掑弬鏁板寲寰€杩斿亸宸?3.5e-06锛涚爜鏈潪璐熷崟浣?  C. 鍗婂緞缃戞牸 V=2/3/4 鐩稿璇樊 0.124/0.053/0.025锛涢浂姝ラ暱缁勬纭€€鍖?  D. ESG 缃崲鍚堟硶锛涘瓙鍚戦噺鍐呰兘閲忓彉寮傦細鐩搁偦鍒嗙粍 0.022 鈫?ESG 0.964
  E. ABS 涓夊垏鍒嗭紙鍧?2.25bpw锛夐€夋縺娲诲姞鏉冭宸渶灏忚€咃紱total bpp 2.562
  F. 涓ゅ眰 MLP 绔埌绔?MSE锛氭棤琛ュ伩 7.51e-03 鈫?blockwise 4.29e-03锛堥檷浣?42.9%锛?```

F 椤瑰嵆"**鏈?blockwise 璇樊琛ュ伩鍚?2bit 绾?VQ 涓嶅啀宕╂簝**"鐨勬満鍒惰瘉鎹€斺€旀湰瀹炵幇鐩稿 PCDVQ 鐨勬牳蹇冨鐩娿€?
---

## 9. 灞€闄愪笌鍚庣画

- **NPU 瀹炴祴缁撴灉锛?026-10-16锛岃 [../progress.md](../../logs/progress-2026-10-16_10-19.md) 闃舵 5锛?*锛氭湰瀹炵幇宸插湪 Ascend 910B3 涓婅窇閫?`npv_verify.py` / `model_quant.py` / `ppl_eval.py`銆?  - 鏍稿績绠楁硶姝ｇ‘锛氬悎鎴愰珮鏂笌鐪熷疄鏉冮噸锛堟棤 ESG锛?64,4)/(256,2) 鐨?rel鈮?.31/0.27銆佹柟鍚戜綑寮︹増0.95/0.97銆?  - **绾?fake-quant锛堟棤寰皟锛変笅 Qwen3-0.6B 鍦?2.0鈥?.75bpw 鍏ㄩ儴宕╂簝**锛圥PL 135鈥?.2e6锛夛紱Qwen3-4B锛堣鏂囬獙璇佽妯★級no-esg 2.25bpw 涓?PPL 13.20鈫?6.66锛?178%锛夈€?  - **ESG 鍦?WikiText 灏忔牎鍑嗕笅鏈夊**锛坮el 0.31鈫?.63锛夛紝涓荤嚎寤鸿鍏?`--no-esg`銆?- **鏍″噯鏁版嵁**锛氬畼鏂圭敤鐩爣妯″瀷鑷噰鏍凤紙self-rollout锛?920脳2048锛夛紱鏈疄鐜伴粯璁?WikiText-2锛岃妯¤繙灏忎簬瀹樻柟 鈥斺€?闇€鍔犺嚜閲囨牱浠ュ鐜板畼鏂圭簿搴︺€?- **鍧楃骇搴忚疮**锛歚--sequential` 涓洪€愬眰閲嶅墠鍚戠殑鏈寸礌瀹炵幇锛屾垚鏈?O(L)锛涘彲浼樺寲涓哄崟娆′紶鎾€?- **閮ㄧ讲 kernel**锛欰scend 鏃?SAGE VQ 瑙ｇ爜绠楀瓙锛涚湡瀹炲帇缂╁眰灞炲悗缁伐浣滐紙绮惧害楠岃瘉璧?fake-quant锛夈€?- **浣嶅鑼冨洿**锛歋AGE 涓?鈮?bit 鏋佷綆浣嶆柟娉曪紝宸ヤ綔鐐圭害 2.0鈥?.5 bpw锛涢珮浣嶆。鐢?GPTQ-INT 鎵挎媴锛堣 [`../SAGE-feasibility.md`](../../docs/SAGE-feasibility.md) 搂4锛夈€?