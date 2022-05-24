# CMANet
CMANet is a network dedicated to high accuracy, high resolution, and fast inference. Currently it is used in Season Depth  Challenge.  

updata 2022/5/24
Our CMANet took third place in the Season Depth Challenge.

![幻灯片4](https://user-images.githubusercontent.com/52912893/169947620-5d1ad043-4bbd-45ca-9f43-fb35b7fda099.PNG)
![幻灯片5](https://user-images.githubusercontent.com/52912893/169947641-f2a3d299-a925-4591-93ad-b34670050c1b.PNG)

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
# Test on Season Dataset
CMANet weights on Season Dataset in [here](https://drive.google.com/file/d/1NJYFBAgBXmlKElhzKliw0qkI6vdGQxNv/view?usp=sharing).
```
python test.py --model_set  --test_data
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

