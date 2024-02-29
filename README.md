
<div style="border-bottom:none;">
	<div align="center">
		<img src="https://upload.wikimedia.org/wikipedia/commons/8/89/Universit%C3%A4t_Z%C3%BCrich_logo.svg" width="600">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/ETH_Z%C3%BCrich_Logo.svg/2560px-ETH_Z%C3%BCrich_Logo.svg.png" width="600">
		<h1><b>Spiking Neural Networks Stimulus Encoding in Neuroprosthetics</b></h1>
		<h3>Institute of Neuroinformatics - Sensors Group</h3>
	</div>
</div>


## Abstract

Sensory neuroprostheses are increasingly becoming viable technology for restoring lost sensory capabilities. 
Recently, several architectures have been proposed to find the optimal electrode stimulation strategy to elicit the desired biological response. 
The most promising of such architectures is the end-to-end optimization of a deep learning autoencoder using a reconstruction loss. 
Although these approaches provide promising results, the stimulation encoder model is energy-inefficient and impractical for a wearable edge device. 
To address this issue, we propose using a spiking neural network to train an energy-efficient stimulation encoder model. 
We show that our model performs on par with non-spiking models while reducing the necessary computations our benchmarks dataset MNIST. 
Overall, this approach provides an important step in developing efficient deep learning-based models that can be used for neuroprosthetic devices.

## Structure of this Repository
```
+-- _config                               | configurations for running our experiments
|   +-- other                             | other configurations (not included in the report)
|   +-- MNIST_spike_E2E.yaml              | configuration for the E2E autoencoder
|   +-- MNIST_spike_SNN_Enc_ANN_Dec.yaml  | configuration for the SNN encoder / ANN decoder
|   +-- MNIST_spike_SNN_full.yaml         | configuration for the SNN autoencoder without phosphene simulator
+-- _Datasets                             | datasets used in the report (ADEK20 and Characters excluded)
|   +-- MNIST                             | the MNIST dataset
+-- Out                                   | output directory
|   +-- spike_E2E                         | training output for E2E autoencoder
|   +-- spike_SNN_Enc_ANN_Dec             | training output for SNN endoer / ANN decoder 
|   +-- spike_SNN_full                    | training output for SNN autoencoder without phosphene simulator
+-- init_training.py                      | training setup
+-- local_datasets.py                     | dataset setup
+-- model.py                              | all models
+-- plotter.ipynb                         | plots of the models
+-- README.md                             | this README file
+-- training.py                           | training loops
+-- utils.py                              | some utility functions
```

