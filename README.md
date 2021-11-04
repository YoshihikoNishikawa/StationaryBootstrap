

# Stationary bootstrap method

![license](https://img.shields.io/badge/license-GPLv3-brightgreen)
## The aim of the code
The stationary bootstrap ([Politis, D. N. and Romano, J. P.](https://www.jstor.org/stable/2290993)) 
is a bootstrap method for estimating statistical properties of samples with time correlation.

The Python code `Python/stationary_bootstrap.py` implements the stationary bootstrap method and calculates the standard error and the confidence interval, using 
[the bootstrap percentile](https://www.taylorfrancis.com/books/mono/10.1201/9780429246593/introduction-bootstrap-bradley-efron-tibshirani), 
[the bias-corrected](https://projecteuclid.org/journals/statistical-science/volume-11/issue-3/Bootstrap-confidence-intervals/10.1214/ss/1032280214.full), 
or [the bootstrap-t](https://projecteuclid.org/journals/statistical-science/volume-11/issue-3/Bootstrap-confidence-intervals/10.1214/ss/1032280214.full) methods, 
with an optimal choice of the parameter 
([Politis, D.N. and White, H.](http://www.tandfonline.com/doi/abs/10.1081/ETC-120028836), [Patton, A., Politis, D.N. and White, H.](http://www.tandfonline.com/doi/abs/10.1080/07474930802459016)) from a single time series. 


## Authors
[Yoshihiko Nishikawa (Tohoku University)](mailto:yoshihiko.nishikawa.a7@tohoku.ac.jp), [Jun Takahashi (University of New Mexico)](https://github.com/JunGitef17), and [Takashi Takahashi (University of Tokyo)](https://github.com/takashi-takahashi)



## Requirements
- numpy
- scipy
- matplotlib
- argparse
- numba

You can install them by `pip3 install -r requirements.txt`.


## Usage
As the simplest use, typing from terminal
```shell
python3 stationary_bootstrap.py ../data/timeseries.dat
```
in the `Python/` directory will output three PDFs,
```
mean_timeseries.pdf
susceptibility_timeseries.pdf
Binder_parameter_timeseries.pdf
```
and three DATs,
```
mean_timeseries.dat
susceptibility_timeseries.dat
Binder_timeseries.dat
```
In the PDFs, the standard error and the confidence interval are plotted as functions of the timeseries length. The output DATs have descriptions in them what each column means.  

Type 
```shell
python3 stationary_bootstrap.py -h
``` 
for further help.

### Extension to other quantities
If you wish to use the code for physical quantities apart from the simple mean, the susceptibility, and the kurtosis, change the function 
```
mean_suscep_kurtosis(list_bsamples)
```
so that the quantities in interest are calculated.


## Contributing
If you wish to contribute, please submit a pull request.

If you find an issue or a bug, please contact us or raise an issue. 