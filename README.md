# Using Resnet architecture in the contextual bandit framework for financial asset trading

## Abstract:

Machine learning techniques have long been a theme of great interest in financial market literature. More recently, reinforcement learning and deep learning methods have been applied to the asset trading task achieving outstanding performances compared with the classical benchmarks. This article proposes a different view on the reinforcement learning approaches for the trading problem using the contextual bandit technique as a framework. We study the Thompson sampling technique's performance and suggest an adaptation of the Resnet architecture to propose the Resnet-LSTM actor (RSLSTM-A) to overcome the problems related to high noised finance fit the dynamic problem of financial asset trading. Furthermore, in this work, we show the increase of performance achieved by the classical Resnet architecture's modifications. We compare classical and recent approaches using the full reinforcement learning techniques such as recurrent reinforcement learning, deep q-learning, and advantage actor-critic. To perform the test, we simulated a financial market environment with the price time-series of the Bitcoin, Litecoin, Ethereum, Monero, and Dash cryptocurrencies. Finally, we show that our approach has a better overall performance when considering all the cryptocurrencies and present some analyzes of the features extracted from the Resnet neural networks.


## Requirements

- Python 3.6
- [gym](https://github.com/openai/gym)
- [Keras 2.4.3](https://pypi.org/project/Keras/)
- [TensorFlow 1.12.0](https://pypi.org/project/tensorflow/)
- [ta 0.5.25](https://pypi.org/project/ta/)
- [Empyrical 0.5.5](https://pypi.org/project/empyrical/)
- [Scikit-learn 0.20.0](https://pypi.org/project/scikit-learn/)
- [Pytorch 1.7.0](https://pytorch.org/)

## Usage

First, install prerequisites

```
$ pip install gym
$ pip install keras==2.4.3
$ pip install ta
$ pip install empyrical
$ pip install -U scikit-learn
```

Check pytorch address for the compatible version

To train all the models besides the RSLSTM-A run the file: [single-run.py](run/single-run.py)

To change the asset you need to change directly on the python code the data source

```python
input_data_file = path+'/data/Poloniex_DASHUSD_1h.csv'
```

To execute the RSLSTM-A, run the file: [ResnetCB.py](run/ResnetCB.py)

## Plotting Results

This is a example of a plot when comparing all the results of the four models proposed in the article

<p align="center">
    <img src="https://raw.githubusercontent.com/leokan92/Contextual-bandit-Resnet-trading/main/images/test_btc.png?token=AINPHV254E7JCKAETMAPYVK72FHK6" width="640"\>
</p>


## References

[Contextual Bandit Git](https://github.com/david-cortes/contextualbandits)

[Resnet for Time-Series Classification](https://github.com/hfawaz/dl-4-tsc)
	

If you re-use this work, please cite:

```
@article{Felizardo2020,
  Title                    = {Using Resnet architecture in the contextual bandit framework for financial asset trading},
  Author                   = {Felizardo, Leonardo},
  journal                  = {},
  Year                     = {2020},
  volume                   = {},
  number                   = {},
  pages                    = {},
}
```







