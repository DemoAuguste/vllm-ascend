# -*- coding: UTF-8 -*-
"""GPTQ 权重伪量化算子（移植自昇腾 AMCT，泛化到任意 int 位宽）。"""
from .core import GPTQ
from .quant import int_qparams, quant_dequant_int, quant_dequant_weight

__all__ = ["GPTQ", "int_qparams", "quant_dequant_int", "quant_dequant_weight"]
