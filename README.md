# C3O: Collaborative Cluster Configuration Optimization in Public Clouds

## Repository Content

- The [C3O runtime predictor](/RuntimePrediction)
- The [C3O cluster configurator](/ClusterConfiguration) as a prototype
- The [Spark jobs runtime data](/data) which originates from a [previous paper](https://github.com/dos-group/c3o-experiments)
- An [evaluation of the C3O runtime predictor](/evaluation) in single-user, collaborative and low data availability scenarios

## C3O Prototype 

### Dependencies

- Python >= 3.6
- Libraries: [scipy](https://pypi.org/project/scipy/), [scikit-learn](https://pypi.org/project/scikit-learn/), [numpy](https://pypi.org/project/numpy/), [pandas](https://pypi.org/project/pandas/)

### Testing the prototype

The prototype can be tested by executing `c3o.py`.  
The file `c3o_cc_examples.sh` contains usage examples. On systems that have [bash](https://en.wikipedia.org/wiki/Bash_\(Unix_shell\)), it can be executed directly after making it executable.


### Example Execution

```
$ python c3o.py 'Page Rank' 330 2000000 3000000 0.0007

Configuring cluster to execute a Page Rank job in 330s with a confidence of 0.95
Execution context for Page Rank:
    links: 2000000
    pages: 3000000
    convergence_criterion: 0.0007
Estimated mean runtime prediction error: 0.69s, standard deviation: 12.99s
Required tolerance to reach the deadline in 95.0% of cases: 22.06s
Estimated optimal cluster configuration: 6 x r4.2xlarge with estimated runtime: 299.66s
```
