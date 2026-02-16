# Bonsai_ABCD

Small repo for parsing and plotting Bonsai VR behavioural sessions in the style of ABC/ABCD tasks.

## Installation (tested compatibility for use in Win and Mac OS)
I recommend using the environment.yml file to create a new conda environment 

```console
conda env create -f environment.yml
```

to activate the environment, run

```console
conda activate bonsai_abcd
```

### Alternatively, 
create a blank environment and install python 3.12:

```console
conda create --name bonsai_abcd python=3.12
```

activate it:

```console
conda activate bonsai_abcd
```

then use the requirements.txt file to pip install the necessary packages  

```console
pip install -r requirements.txt
```


## Use

The main output script is the jupyter notebook Cohort3_daily_summary.ipynb. 
To analyse the behaviour of one session, change [mouse_id] and [date] to the session you want, then run the cells. 

To keep the script tidy, it imports the central functions from parse_bonsai_functions.py, which is a collection of all parsing and plotting operations. 
To add new functionality, write a function in that script and call it in the notebook. 

There will be some analyses/plots that only make sense under certain task conditions (e.g. stable world vs. random world). 
As the scripts grow, let's try and make sure it is always indicated which conditions need to be met.
