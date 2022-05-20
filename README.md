# CMANet
CMANet is a network dedicated to high accuracy, high resolution, and fast inference. Currently it is used in Season Depth  Challenge.  

The rest will be updated later.

# Season Depth

Season Depth  Challenge is an open-source monocular challenge that runs through February-May 2022. Please refer to the [website](http://seasondepth-challenge.org/index/index.html#introduction) for more detailed information.

# Requirements

```
torch                   1.7.0
timm                    0.4.12
opencv-python           4.1.2.30
python                  3.6.9
einops                  0.4.1
```

# Compare

## 1. Basic Information

| Methods | Resolution | Flops | Params | Inference Time |
| ------- | ---------- | ----- | ------ | -------------- |
| CMANet  | 640x640    | 273G  | 103M   | **21ms**       |
| DPT     | 640x640    | 306G  | 123M   | 140ms          |

Note: On the RTX2080ti, CMANet can train images with a resolution of 768x1024.

## 2. Season Depth Benchmark

| Methods | **Mean AbsRel** | **Mean a1** |
| ------- | --------------- | ----------- |
| CMANet  | 0.140           | 0.818       |
| DPT     | 0.152           | 0.790       |

