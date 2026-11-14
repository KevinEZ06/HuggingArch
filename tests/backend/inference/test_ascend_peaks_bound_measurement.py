"""An Ascend sheet cell is an upper bound on what the hardware reached.

A roofline divides by the cell, so a cell BELOW the hardware makes every
consumer read above 100% of peak — the defect that put 1,439 RTX-4090 rows
over peak (`gpu_specs` module docstring). These are the figures a probe
reached on one 910C die; the sheet must stay at or above each. They are a
floor on the cell, never a replacement for it: a kernel reaching 66% of a
cell says nothing about whether the cell is the vendor's peak.

Measured 2026-11-14 on an Atlas A3 node, one die, inside the
`quay.io/ascend/vllm-ascend:v0.30.0-a3` container (torch_npu 2.10.0.post7),
at 8192^3 for the matmuls and a 4 GiB device-to-device copy:

    torch.matmul bf16        248.5 TFLOPs   (4096^3; 227.7 at 8192^3)
    npu_quant_matmul int8    392.3 TOPs     (8192^3)
    copy_ bytes/s           1262 GB/s       (1 GiB; 1249 at 4 GiB)
    HBM capacity              64 GiB        (npu-smi -t memory: 65536 MiB)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.gpu_specs import get_gpu  # noqa: E402

#: The best each probe reached on one die. A cell at or above these is a
#: bound; a cell below one is a sheet the hardware has already beaten.
REACHED = {"bf16_tflops": 248.5, "fp16_tflops": 248.5, "int8_tops": 392.3,
           "mem_bw_gbps": 1262.0}
DIE_HBM_GIB = 64.0


class TestAscendCellsBoundTheProbe(unittest.TestCase):

    def test_every_compute_and_bandwidth_cell_is_at_least_what_ran(self):
        for sku in ("Ascend-910C-A3", "Ascend-910C-CM384"):
            card = get_gpu(sku)
            self.assertIsNotNone(card, sku)
            for column, reached in REACHED.items():
                cell = getattr(card, column)
                self.assertIsNotNone(cell, f"{sku}.{column}")
                self.assertGreaterEqual(cell, reached, f"{sku}.{column}")

    def test_the_die_states_the_capacity_npu_smi_reports(self):
        for sku in ("Ascend-910C-A3", "Ascend-910C-CM384"):
            self.assertEqual(get_gpu(sku).mem_gib, DIE_HBM_GIB, sku)

    def test_no_fp8_or_fp4_peak_is_claimed(self):
        # The probe ran no FP8 or FP4 matmul because the hardware has no
        # such MMA; a cell here would price a launch nothing runs.
        for sku in ("Ascend-910C-A3", "Ascend-910C-CM384"):
            card = get_gpu(sku)
            self.assertIsNone(card.fp8_tflops, sku)
            self.assertIsNone(card.fp4_tflops, sku)


if __name__ == "__main__":
    unittest.main()
