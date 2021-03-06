# Gym-Dofbot
## OpenAI Gym environment for low-cost Yahboom [Dofbot](https://category.yahboom.net/products/dofbot-jetson_nano)

Gym-Dofbot is an reinforcement-learning friendly Gym environment that is powered by Pybullet. There are certain features that make this package somewhat useful:

## Features

- Re-modelled camera module that supports real-time streaming of images in the front
- Allows precise position control for the servo group
- Potential for multi-agent learning and object interaction

### Single agent
![gif](https://github.com/CharlesXu1124/gym-dofbot/blob/main/Demo/dofbot-gym.gif?raw=true)

### Multi-agent (in progress)
![gif](https://github.com/CharlesXu1124/gym-dofbot/blob/main/Demo/dofbot-ma.gif?raw=true)

## Installation

Install the numpy, pybullet and gym.
Install via pip:
```
pip install gym-dofbot==0.1.1
```
Install via Github repository:
```sh
cd gym_dofbot
pip install -e .
```

## Development

Want to contribute? Great!

Code is straightforward enough, feel free to fork and add more content to it.


## License

GPL
