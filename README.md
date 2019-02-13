# Reinforcement Learning Wechat Jump
End-to-end training Wechat-Jump AI using DDPG algorithm

## Environment
* PyTorch
* PyOpenCV
* Android Device(1920 * 1080)
* ADB Tools

## Detail
* Using screenshot as neural network input
* The `actor` uses tanh as the activation function to output the action
* `Critic` uses a linear layer to output Q values
* Noise is a normal distribution with a `std=0.2`
* Get game state with template matching of OpenCV

## Train
```
python train.py
```

## Infer
```
python infer.py
```

## Reference
* [Continuous control with deep reinforcement learning](http://xueshu.baidu.com/s?wd=paperuri:(3752bdb69e8a3f4849ecba38b2b0168f)&filter=sc_long_sign&sc_ks_para=q%3DContinuous+control+with+deep+reinforcement+learning&tn=SE_baiduxueshu_c1gjeupa&ie=utf-8&sc_us=5932345815760573065)
* https://github.com/ikostrikov/pytorch-ddpg-naf
* https://github.com/floodsung/wechat_jump_end_to_end
