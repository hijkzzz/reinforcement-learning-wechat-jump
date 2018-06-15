# WeChat Jump DDPG
用 DDPG 算法玩微信跳一跳

## Tricks
* Actor 用 Tanh 输出动作 (-1, 1) 缩放到 400 ms ~ 1200 ms (根据机型设置)
* Critic 最后一层用 Linear 输出 Q 值
* BATH_SIZE = 16, 去除了 Batch Normalization 层
* 噪音为标准差等于 0.2 的正态分布
* OpenCV 的模板匹配提取分数和获取结束状态
* adb 调试控制手机

## Train
**本项目需安装 PyTorch/OpenCV/ADB**

因为使用真机训练几秒钟才能跳一次，样本的收集速度是比较慢的，
可能需要训练好几天的时间

* 测试环境为一加5
* 分辨率1920 * 1080
* 暂未兼容其它分辨率
```
python train.py
```

## Test
models 中有在一加5上训练了5000个样本(10小时收集)的模型
已经能跳几百分了

```
python infer.py
```

## Reference
* [Continuous control with deep reinforcement learning](http://xueshu.baidu.com/s?wd=paperuri:(3752bdb69e8a3f4849ecba38b2b0168f)&filter=sc_long_sign&sc_ks_para=q%3DContinuous+control+with+deep+reinforcement+learning&tn=SE_baiduxueshu_c1gjeupa&ie=utf-8&sc_us=5932345815760573065)
* https://github.com/ikostrikov/pytorch-ddpg-naf
* https://github.com/floodsung/wechat_jump_end_to_end
