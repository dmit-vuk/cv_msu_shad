from typing import List, Tuple

import numpy as np
import torch


# Task 1 (1 point)
class QuantizationParameters:
    def __init__(
        self,
        scale: np.float64,
        zero_point: np.int32,
        q_min: np.int32,
        q_max: np.int32,
    ):
        self.scale = scale
        self.zero_point = zero_point
        self.q_min = q_min
        self.q_max = q_max

    def __repr__(self):
        return f"scale: {self.scale}, zero_point: {self.zero_point}"


def compute_quantization_params(
    r_min: np.float32,
    r_max: np.float32,
    q_min: np.int32,
    q_max: np.int32,
) -> QuantizationParameters:
    # your code goes here \/
    scale = np.float64((r_max - r_min) / (q_max - q_min))
    zero_point = np.int32(round((r_max*q_min - q_max*r_min) / (r_max - r_min)))
    params = QuantizationParameters(scale=scale, zero_point=zero_point, q_min=q_min, q_max=q_max)
    return params
    # your code goes here /\


# Task 2 (0.5 + 0.5 = 1 point)
def quantize(r: np.ndarray, qp: QuantizationParameters) -> np.ndarray:
    # your code goes here \/
    return np.clip(np.round(r/qp.scale + qp.zero_point), qp.q_min, qp.q_max).astype(np.int8)
    # your code goes here /\

def dequantize(q: np.ndarray, qp: QuantizationParameters) -> np.ndarray:
    # your code goes here \/
    return qp.scale*(q.astype(np.int32) - qp.zero_point)
    # your code goes here /\


# Task 3 (1 point)
class MinMaxObserver:
    def __init__(self):
        self.min = np.finfo(np.float32).max
        self.max = np.finfo(np.float32).min

    def __call__(self, x: torch.Tensor):
        # your code goes here \/
        self.min = min(np.float32(x.min()), self.min)
        self.max = max(np.float32(x.max()), self.max)
        # your code goes here /\


# Task 4 (1 + 1 = 2 points)
def quantize_weights_per_tensor(
    weights: np.ndarray,
) -> Tuple[np.array, QuantizationParameters]:
    # your code goes here \/
    min_max = MinMaxObserver()
    min_max(weights)
    board = max(abs(min_max.min), min_max.max)
    qp = compute_quantization_params(-board, board, np.int32(-127), np.int32(127))
    return quantize(weights, qp), qp
    # your code goes here /\


def quantize_weights_per_channel(
    weights: np.ndarray,
) -> Tuple[np.array, List[QuantizationParameters]]:
    # your code goes here \/
    qps = []
    quantize_weights = []
    for channel in weights:
        min_max = MinMaxObserver()
        min_max(channel)
        board = max(abs(min_max.min), min_max.max)
        qp = compute_quantization_params(-board, board, np.int32(-127), np.int32(127))
        quantize_weights.append(quantize(channel, qp))
        qps.append(qp)
    return np.array(quantize_weights), qps
    # your code goes here /\


# Task 5 (1 point)
def quantize_bias(
    bias: np.float32,
    scale_w: np.float64,
    scale_x: np.float64,
) -> np.int32:
    # your code goes here \/
    return np.round(bias/(scale_w*scale_x)).astype(np.int32)
    # your code goes here /\


# Task 6 (2 points)
def quantize_multiplier(m: np.float64) -> [np.int32, np.int32]:
    # your code goes here \/
    for n in range(-1000, 1000):
        m0 = m * 2**n 
        if m0 < 1 and m0 >= 0.5:
            break
    m0 *= 2**31
    return np.int32(n), np.int32(round(m0))
    # your code goes here /\


# Task 7 (2 points)
def multiply_by_quantized_multiplier(
    accum: np.int32,
    n: np.int32,
    m0: np.int32,
) -> np.int32:
    # your code goes here \/
    res = np.multiply(accum, m0, dtype=np.int64)
    point = 33 - n
    last_sign = int(np.binary_repr(np.right_shift(res, 64 - point - 1))[-1])
    res = np.right_shift(res, 64-point).astype(np.int32) + np.int32(last_sign)
    return res
    # your code goes here /\
