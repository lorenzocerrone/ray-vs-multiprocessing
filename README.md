# ray-vs-multiprocessing
Basic experiment to benchmark ray against python multiprocessing

## install environment
```
conda create -n ray-vs-mp -c conda-forge numpy matplotlib tqdm scikit-image memory_profiler
pip install ray[all] 
```

## run with profiler
Total memory
```
mprof run --include-children main.py 
mprof plot
```
Process by process memory
```
mprof run --multiprocessing main.py 
mprof plot
```