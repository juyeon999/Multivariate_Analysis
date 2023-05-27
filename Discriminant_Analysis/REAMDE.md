# Discriminant Analysis
## Goal
1. To describe, either graphically or algebraically, the differential features of objects from several unknown populations.
We try to find "discriminants whose numerical values are such that the populations are seperated as much as possible.

2. To sort observations into two or more labeled classes.

## Gaussian Assumption
Knowing true pdf is impossible, we assume the populations folllow multivariate normal.  
$$X_1 \sim N_p(\mu_1, \Sigma_1), \quad X_2 \sim N_p(\mu_2, \Sigma_2)$$

### LDA (Linear Discriminant Analysis)
Main assumption is  
**$\Sigma_1 = \Sigma_2 = \Sigma$ (등분산가정)**
등분산 가정하면 posterior probability가 quadratic term이 없어져서 Linear in X가 된다.  
Here, we
$$\hat{\mu_1} = \bar{X_1}, \quad \hat{\mu_2} = \bar{X_2}, \quad \hat{\Sigma} = S_p = \frac{n_1 - 1}{n_1 + n_2 -2}S_1 + \frac{n_2 - 1}{n_1 + n_2 -2}S_2$$
### QDA (Quadraatic Discriminant Anaysis)  
Main assumption is  
**$\Sigma_1 = \Sigma_2 = \Sigma$ (등분산가정)**
$$\hat{\mu_1} = \bar{X_1}, \quad \hat{\mu_2} = \bar{X_2}, \quad \hat{\Sigma_1} = S_1, \quad \hat{\Sigma_2} = S_2$$

## Procedure
#### 1. Check the Homoscedasticity 
 - Use Python's `statsmodels.stats.multivaritate`
```python
## 등분산 검정(homogeneity of variance test) 
# Covariance matrix for each group
classes, n_groups = np.unique(y, return_counts=True)
for i, name in enumerate(classes):
    globals()['cov' + str(i+1)] = X[y == name].cov()

# Testing
test = mv.test_cov_oneway([cov1, cov2, cov3], n_groups)
print(f'Chi-Square Test stat: {test.statistic_chi2 :.5f}, Pr > ChiSq: {test.pvalue_chi2 :.5f}')
print('Reject H0')
```
Ho(null hypothesis) is the group is homoscedastic which means the group variances are same.  

#### 2. Discriminant Analysis
- If satisfy homoscedasticity, do LDA. Else QDA.
