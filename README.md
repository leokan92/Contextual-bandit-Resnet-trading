# Outperforming algorithmic trading reinforcement learning systems: a supervised approach in the cryptocurrency market

## Abstract:

The interdisciplinary relationship between machine learning and financial markets has long been a theme of great interest among both research communities. Recently, reinforcement learning and deep learning methods gained prominence in the active asset trading task, aiming to achieve outstanding performances compared with classical benchmarks, such as the Buy and Hold strategy. This paper explores both supervised learning and reinforcement learning approaches applied to active asset trading, drawing attention to the benefits of both approaches. This work extends the comparison between the supervised approach and reinforcement learning by using state-of-the-art strategies in both techniques. We propose adopting the Resnet architecture, one of the best deep learning approaches for time-series classification, into the Resnet-LSTM actor (RSLSTM-A). We compare RSLSTM-A against classical and recent reinforcement learning techniques, such as recurrent reinforcement learning, deep Q-network, and advantage actor-critic. We simulated a currency exchange market environment with the price time-series of the Bitcoin, Litecoin, Ethereum, Monero, and Dash cryptocurrencies to run our tests. We show that our approach achieves better overall performance, confirming that supervised learning can outperform reinforcement learning for trading. We also present a graphic representation of the features extracted from the Resnet neural network to identify which type of characteristics each residual block generates.


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

Check pytorch library for the compatible version

To train all the models besides the RSLSTM-A run the file: [single-run.py](run/single-run.py)

To change the asset you need to change directly on the python code the data source

```python
input_data_file = path+'/data/Poloniex_DASHUSD_1h.csv'
```

To execute the RSLSTM-A, run the file: [ResnetCB.py](run/ResnetCB.py)


## Plotting Results

This is a example of a plot when comparing all the results of the four models proposed in the article.
All the plots and table generator are presented in the `folder plots and tables`

<p align="center">
    <img src="https://raw.githubusercontent.com/leokan92/Contextual-bandit-Resnet-trading/main/images/test_btc.png?token=AINPHV254E7JCKAETMAPYVK72FHK6" width="640"\>
</p>


## References

[Contextual Bandit Git](https://github.com/david-cortes/contextualbandits)

[Resnet for Time-Series Classification](https://github.com/hfawaz/dl-4-tsc)
	

## Citing the Project

If you re-use this work, please cite:

```bibtex
@misc{Felizardo2021,
  author = {Charles, P.W.D.},
  title = {Outperforming algorithmic trading reinforcement learning systems: a supervised approach in the cryptocurrency market},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/leokan92/Contextual-bandit-Resnet-trading}},
  commit = {2452a2b786af53d210da3c4e42544d25ae06c1f6}
}
```







