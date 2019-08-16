# Locally Linear Regression for graph-linked data

Python package implementing a compression based version for fitting and predicting responses using Locally Linear Regression(LLR) on graph-linked data.

## Getting Started

Following steps discribe the installation process, as well as few examples using LLR. 

### Prerequisites

The packege requires you to have Python version 3.3 +. To check your Python version please follow the directions below.

Checking the Python version:
```
python --version
```

### Installation

```
pip
```

## Documentation

### Class 

```
LLR()
```
Locally Linear Regression model.

### Methods
```
LLR.fit(X, y, mu, v, perm_size = 50, var=None, Graph=None):
```
Full fit of the model. 

**Parameters:**

&nbsp;&nbsp;&nbsp;&nbsp; **X**: *array_like*
&nbsp;&nbsp;&nbsp;&nbsp; An n by p array with n observations and p features. 

&nbsp;&nbsp;&nbsp;&nbsp; **y**: *array_like*
&nbsp;&nbsp;&nbsp;&nbsp; 1-d array of response variable.

&nbsp;&nbsp;&nbsp;&nbsp; **mu**: *int*
&nbsp;&nbsp;&nbsp;&nbsp; Tuning parameter affecting the y-intercept . 

&nbsp;&nbsp;&nbsp;&nbsp; **v**: *int*
&nbsp;&nbsp;&nbsp;&nbsp; Tuning parameter affecting the regression coefficients. 

&nbsp;&nbsp;&nbsp;&nbsp; **perm_size**: *int*
&nbsp;&nbsp;&nbsp;&nbsp; Number of data points to keep in compression. 

&nbsp;&nbsp;&nbsp;&nbsp; **var**: *double*
&nbsp;&nbsp;&nbsp;&nbsp; Variance of Gaussian Kernel, needed if Graph parameter is None. 

&nbsp;&nbsp;&nbsp;&nbsp; **Graph**: *array_like*
&nbsp;&nbsp;&nbsp;&nbsp; Graph of data points passed as an adjacency matrix. 

```
LLR.predict(X_new):
```
Returns the predicted values of the response variable as a 1-d array. 

**Parameters:**

&nbsp;&nbsp;&nbsp;&nbsp; **X_new**: *array_like*
&nbsp;&nbsp;&nbsp;&nbsp; New array of data points, with m observations and p features. 

&nbsp;&nbsp;&nbsp;&nbsp; **Graph**: *array_like*
&nbsp;&nbsp;&nbsp;&nbsp; Graph of the data points in X_new passed as an adjacency matrix. 


## Running the examples

The examples in the /examples folder describe ways to fit the LLR model to the data, predict responses and choose tuning parameters using cross validation. 


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.