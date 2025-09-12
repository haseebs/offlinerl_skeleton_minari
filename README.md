# Offline RL skeleton code


All the run statistics are logged to a MySQL database server. The schema can be found in [`configs/schema/default-schema.yaml`](configs/schema/default-schema.yaml)
## How to run
0. This codebase contains some features that are only available in Python3.10+

1. Install requirements:
``` bash
pip install -r requirments.txt
``` 

2. Add the correct db credentials to [`configs/db/credentials.yaml`](configs/db/credentials.yaml). You can log the results to the local database, to a google cloud server (recommended) or get a database server on Compute Canada by following these [instructions](https://docs.computecanada.ca/wiki/Database_servers). I would recommend making `credentials-local.yaml` and adding it to `.gitignore`.

3. Set `db_prefix` to your username in [`configs/config.yaml`](configs/config.yaml). This is especially required if hosting on CC because they only allow you to make databases that start with your CC username. 

4. Run the following command for a sweep of 2400 runs as defined in
   [`configs/config.yaml`](configs/config.yaml). The `args.run` argument
is the run id for the first experiment in the sweep. Other experiments
are automatically assigned a run id of args.run + sweep_id in the
database.

``` bash
python main.py run=0
```


# Install D4RL Mujoco


## on Ubuntu:
1. download mujoco 210 from https://github.com/google-deepmind/mujoco/releases?expanded=true&page=2&q=2.1

2. follow the instructions here https://github.com/openai/mujoco-py 
> Extract the downloaded mujoco210 directory into ~/.mujoco/mujoco210

3. when installing the requirments it prompts that swig and g++ are missing
```bash
pip install swig
sudo apt-get install g++
```
4. install all dependencies via `pip install requirements.txt`, 
add environmental variable the ~/.bashrc
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lingwei/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```



5. on linux mujoco-py needs additional packages, see https://github.com/openai/mujoco-py/issues/627 for detail
```bash
sudo apt-get install libghc-x11-dev
sudo apt install ligblew-dev
```
the following packages are also needed:
```bash
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
```
Then add your conda environment include to CPATH (put this in your .bashrc to make it permanent):

```bash
export CPATH=$CONDA_PREFIX/include
```
Finally, install patchelf with 
```bash
pip install patchelf
```

6. test it with `import mujoco_py`

## on MAC: 
1. download mujoco 210 from https://github.com/google-deepmind/mujoco/releases?expanded=true&page=2&q=2.1

2. follow the instructions here https://github.com/openai/mujoco-py 
> Extract the downloaded mujoco210 directory into ~/.mujoco/mujoco210


3. because on Mac the downloaded Mujoco is `.dmg` file, move the MuJoCo.app to the Application folder, and then build the link

```bash
ln -s /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework /Path/to/Conda/Env/lib/
```

4. install all dependencies via `pip install requirements.txt`, test them by 
```python
import mujoco_py
```