# CUPTI hook (milestone M4)

`ptxforge-tune` (the in-process `CudaRunner`) deliberately ships **without** CUPTI so M0–M3 work on a free Colab T4.
When you reach M4, add counters one at a time, and keep each one *validated* against Nsight Compute (`ncu`).

## Which CUPTI API

| GPU | API | Notes |
| --- | --- | --- |
| Turing/Volta and older (T4, P100, V100) | **CUPTI Event / Metric API** (`cuptiEventGroupSets…`, `cuptiMetricGetValue`) | Deprecated in newer CUDA but still works on T4. |
| Ampere+ (A100, L4, H100) | **CUPTI Profiler Host + Range Profiler** (`cupti_profiler_target.h`, `cupti_range_profiler.h`) | The event API is gone on these. Check `CUPTI_API_VERSION` for your toolkit. |

Colab/Kaggle often block hardware counters (`CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`). If so, run
`ncu --metrics ... --csv` on a machine you control and have the tuner parse the CSV instead.

## Metrics worth wiring up first

| Want | ncu metric (use as ground truth) |
| --- | --- |
| Achieved occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Shared-memory bank conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`, `..._op_st.sum` |
| DRAM bytes (-> measured arithmetic intensity) | `dram__bytes_read.sum`, `dram__bytes_write.sum` |
| L2 hit rate | `lts__t_sector_hit_rate.pct` |
| FP32 FLOPs executed | `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum` (x2) + fadd/fmul |

## Why this matters for the roofline

Until M4 the studio plots **analytic** arithmetic intensity (FLOPs / modelled DRAM bytes).
After M4 it should plot **measured** AI = FLOPs / `dram__bytes`. Keep both on the chart —
the gap between them is a finding (L2 reuse, uncoalesced loads) and good interview material.
