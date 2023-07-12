# SSABN
## Spatial-spectral Attention Bilateral Network for Hyperspectral Unmixing
The code in this toolbox implements the "Spatial-spectral Attention Bilateral Network for Hyperspectral Unmixing". 

## System-specific notes
The code was tested in the environment of `Python 3.8` and `tensorflow==2.4.1`.

Install `requirements.txt` dependency package (environment)
```python
pip install -r requirements.txt
```
The environment can be quickly installed on the target machine

## Run the code
Directly run `SSABN_demo.ipynb` to reproduce the results on the Samson data.

If you want to run the code on your own data, you can change the input accordingly, place the data under the Dataset folder, and tune the parameters. It is important to pay attention to the shape of the input matrix.
