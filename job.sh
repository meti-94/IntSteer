
#!/bin/bash
#PBS -q gpuhopper
#PBS -P hn98
#PBS -l walltime=05:00:00
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=80GB
#PBS -l jobfs=100GB
#PBS -l storage=gdata/hn98
#PBS -l wd
#PBS -N intsteer

echo "Job ${PBS_JOBID} on $(hostname) started at $(date)"
export OPENAI_API_KEY=''
source ../my_env/bin/activate
python src/sae_ts/baselines/analysis.py


