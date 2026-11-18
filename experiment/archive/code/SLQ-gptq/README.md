# SLQ + GPTQ 娣峰悎绮惧害閲忓寲锛坋xperiment/SLQ-gptq锛?
- 瀹氫綅锛氫富绾块噺鍖栨柟妗堬紙闃舵 7锛夈€?*GPTQ** 浣滀负閫愬眰浣嶅閲忓寲绠楀瓙锛圚essian 浜岄樁璇樊琛ュ伩锛夛紝**SLQ**锛圫hapley/绾挎€ф晱鎰熷害 + ILP锛夊仛鍏ㄥ眬娣峰悎绮惧害浣嶅鍒嗛厤銆?- GPTQ 绠楀瓙绉绘骞舵硾鍖栬嚜鏄囪吘 [AMCT](https://gitcode.com/cann/amct)锛圓pache-2.0锛宍amct_pytorch/algorithms/quant/gptq.py`锛孒uawei Technologies 2026锛夛紝浣嶅鐢?`{4,8}` 鎵╁睍鍒颁换鎰?int銆?- 鑳屾櫙锛氶樁娈?5 璇佸疄绾?fake-quant 鐨?VQ锛圥CDVQ/SAGE锛夊湪浣庝綅宕╂簝锛堢己璇樊琛ュ伩锛夛紱GPTQ 鐨?Hessian 琛ュ伩姝ｆ槸缂哄け鐜妭銆傝 [`../progress.md`](../../logs/progress-2026-10-16_10-19.md)銆?
## 1. 鐩綍缁撴瀯

```
SLQ-gptq/
  gptq/            GPTQ 绠楀瓙
    quant.py       INT 瀵圭О閲忓寲-鍙嶉噺鍖栵紙per-channel / per-group锛屼换鎰忎綅瀹斤級
    core.py        GPTQ锛圚essian 绱Н + 鍒楀紡浜岄樁琛ュ伩锛汣PU/NPU 閫氱敤锛?  slq/             SLQ
    sensitivity.py 绾挎€ф晱鎰熷害锛堥€愬眰鍗曠偣鏇挎崲娴?NLL锛?    ilp.py         ILP 浣嶅鍒嗛厤锛堟媺鏍兼湕鏃ュ鍋朵簩鍒?+ scipy.milp 绮剧‘瑙ｏ級
  model_quant.py   鍏ㄦā鍨?GPTQ fake-quant锛堢粺涓€浣嶅 / 閫愬眰 bit_map锛?  ppl_eval.py      WikiText-2 PPL锛堝熀绾?vs 閲忓寲锛?  slq_allocate.py  SLQ 椹卞姩锛氶€愬眰澶氭。浣嶉噺鍖?鈫?鏁忔劅搴?鈫?ILP 鈫?bit_map.json
  tests/test_gptq.py
```

## 2. 蹇€熷紑濮?
```bash
cd experiment/SLQ-gptq

# 鍗曟祴锛圕PU锛?python -m pytest tests/test_gptq.py -q

# 鈶?缁熶竴浣嶅 GPTQ PPL锛圢PU锛?ASCEND_RT_VISIBLE_DEVICES=5 python ppl_eval.py --device npu \
    --model Qwen/Qwen3-0.6B --bits 4 --group-size 128

# 鈶?SLQ 娣峰悎绮惧害锛氬垎閰?bit_map 鍚庤瘎娴?ASCEND_RT_VISIBLE_DEVICES=5 python slq_allocate.py --device npu \
    --model Qwen/Qwen3-0.6B --bits 3,4,8 --target-avg 3.5 \
    --group-size 128 --out bits_06b.json
ASCEND_RT_VISIBLE_DEVICES=5 python ppl_eval.py --device npu \
    --model Qwen/Qwen3-0.6B --bit-map bits_06b.json
```

## 3. 宸查獙璇佺粨鏋?
- **GPTQ 鍗曟祴**锛? passed锛堜綅瀹芥硾鍖?{2,3,4,6,8}銆佽宸ˉ鍋夸紭浜?RTN銆乨ead 鍒楀鐞嗐€乸er-group锛夈€?- **AMCT 鍙傝€冪畻瀛愪竴鑷存€?*锛氫笌 AMCT `gptq.py` 鍚屾瀯锛圚essian 绱Н `sqrt(2/n)`銆乨amp銆丆PU Cholesky 姹傞€嗐€?28 鍒楀垎鍧椼€侀€愬垪浜岄樁琛ュ伩锛夈€?- **Qwen3-0.6B GPTQ int4 g128**锛歐ikiText-2 PPL **20.0419 鈫?22.8689锛?14.11%锛?*锛屽钩鍧?rel 0.1748锛堝姣?SAGE 2.25bpw 鐨勫穿婧?PPL鈮?102锛夈€?
## 4. 涓?AMCT 鐨勫樊寮?
| 椤?| AMCT | 鏈疄鐜?|
|---|---|---|
| 浣嶅 | int4 / int8 | 浠绘剰 int锛坽2..8}锛?|
| 鍒嗙粍 | per-channel锛坕nt锛夈€丠iF4 鎸?64 鍒楃粍 | per-channel 鎴?per-group 鍙厤 |
| 渚濊禆 | amct_pytorch 鍏ㄦ爤锛坥nnx 绛夊彲閫夛級 | 浠?torch锛屽彲鐙珛杩愯 |
| 姹傝В缂撳瓨 | 姣?bit 閲嶆柊瑙傚療 | Hessian/缃崲/姹傞€嗚法浣嶅澶嶇敤锛圫LQ 澶氭。浣嶉珮鏁堬級 |

> 浠呭瘑闆?fake-quant锛堢簿搴﹂獙璇佺敤锛夛紱鐪熷疄鍘嬬缉 kernel/packing 灞為儴缃插伐浣溿€?