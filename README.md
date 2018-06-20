#POC recommender
Proof of concept (POC) of a recommender of time series categorical data.
We use Tensorboard and nmslib for visualization and nearest neighbor queries, respectively.

The virtual conda environment can be setup via the included setup.sh script.

Pull-requests are welcome.


## prior data preparation
Prepare data with integer numbers as in test/train_data_small which is used as input.
Each line is one unser interaction history ordered w.r.t. time (see attached data), e.g.:

```bash
0,_
1,a
2,b
3,c
4,d
5,e
6,f
...
```

A map to string is needed (see attached data), e.g.:

```bash
15,8,34,15,76,77,78,38,76,79,40,76,8,50
15,15,92,15,92,15,92,15,92
15,35,38,37,39,36,40,41,35,15,35,37,39,36,38,40,41,35
29,30
68,68,15
15,6,15,6,3,4,5,15,6,15,6,15
... 
```

## overview
Naming convention:
* interaction: one single integer in the history given in the data
* event: one event contains features and targets beside having an occurence count.

We use the data to generate events. 
The events are sparse one-hot encoded according to the integer given in the data file.
Thus, make sure, the user interactions are mapped to unique sequential integer values! 
For the translation, we use a mapping, defined in the interaction_map file for easier interpretation.



## profiling
Some performance metrics are plotted with Tensorboard.

![tensorboard_example](doc/figures/tensorboard_example.png)

The (final) json file produced can be viewed via a chrome browser:
    chrome://tracing

![json_profiling_example](doc/figures/json_profiling.png)


# REST API (WIP)

## starting
use gunicorn in virtual environment (see environment.yml)

Start server (set timeout to let it create the index):
    gunicorn things:app --timeout=1500
    
## profiling
install:
    sudo apt install apache2-utils
    
profile via, e.g.:
    ab -c 10 -n 100000 http://127.0.0.1:8000/recos?url=aa