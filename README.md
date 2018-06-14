# WeChat Jump DDPG
用 DDPG 算法玩微信跳一跳

# Principle
本项目需安装 PyTorch/OpenCV/ADB

* Actor 输出动作范围 (-1, 1) 缩放到 400 ms ~ 1200 ms
* Critic 最后一层用 Linear 输出 Q 值
* OpenCV 的模板匹配提取分数和获取结束状态
* adb 调试控制手机

# Train
因为使用真机训练几秒钟才能跳一次，样本的收集速度是比较慢的，
经过测试需要训练好几天

* 测试环境为一加5
* 分辨率1920 * 1080
* 暂时未兼容其它分辨率
```
python train.py
```

# Test
models 中有预训练一天的模型文件

```
python infer.py
```

# Reference
https://github.com/ikostrikov/pytorch-ddpg-naf
https://github.com/floodsung/wechat_jump_end_to_end

