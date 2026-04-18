# ☁️ Diffusion-Based-Nowcasting-of-Cloud-Motion-from-Satellite-Image-Sequences 🛰️

```
           \  |  /
          -  [O]  -    <-- Extremely expensive satellite
           /  |  \
             |
             | (taking pics)
             v
          .-~~~-.
  .- ~ ~-(       )_ _    <-- The clouds (they are moving, allegedly)
 /                     ~ -.
|      O          O        \  <-- Clouds judging your GPU limits
 \            ^            .'
   ~- . _____________ . -~
           >>> WHOOSH >>>
```

## 🌪️ What is this?

Welcome to **DBNCMSIS** (an acronym so long it sounds like a rare bone disease). 

Are you tired of meteorologists just *guessing* where the clouds will go? Do you think standard optical flow models are too fast, lightweight, and computationally efficient? Perfect. You are in the right place.

We use state-of-the-art **Diffusion Models** to predict ("nowcast") cloud motion from sequences of satellite images. Because why just look out the window to see if it's going to rain when you can melt an RTX 4090 trying to iteratively denoise the sky?

## ✨ Features

* **Over-engineered Predictions:** We replace simple physics with a diffusion process. We take clouds, add Gaussian noise until they look like TV static, and then ask a neural network to guess where the clouds went. 
* **"Nowcasting":** A fancy meteorological term for "predicting the next 30 minutes." Note: Inference might take 45 minutes, meaning we successfully predict the past.
* **Satellite Image Digestion:** Feeds on high-res satellite image sequences and outputs incredibly convincing (but occasionally hallucinated) weather patterns.

## 🚀 Getting Started

### 1. Prerequisites
You will need:
* Python 3.8+
* PyTorch
* A GPU that requires its own zip code and power grid.
* A healthy disrespect for traditional meteorology.

### 2. Installation
```bash
git clone [[https://github.com/yourusername/cloud-diffusion-nowcasting.git](https://github.com/yourusername/cloud-diffusion-nowcasting.git)](https://github.com/nandininema07/Diffusion-Based-Nowcasting-of-Cloud-Motion-from-Satellite-Image-Sequences.git)
cd Diffusion-Based-Nowcasting-of-Cloud-Motion-from-Satellite-Image-Sequences
pip install -r requirements.txt
```
*(Grab a coffee, `requirements.txt` is going to download half of the internet.)*

## 🎮 Usage

Want to predict if that cloud is going to ruin your picnic in 10 minutes? Run:

```bash
python nowcast.py --input data/satellite_seq.npy --timesteps 1000
```

**What happens next:**
1. The script loads the satellite sequence.
2. The diffusion model goes *brrrrrrrr*.
3. Your fan speed hits 100%.
4. You get an output sequence of where the clouds *might* be, assuming the model didn't hallucinate a hurricane over Nebraska.

## 🤝 Contributing

Found a bug? The clouds moving backwards? The model accidentally predicting the heat death of the universe instead of a mild drizzle? 

Please open an issue! Pull requests are welcome, especially if you know how to make this run faster than the actual weather happens.
