# Job Name
#$ -N UM_Test
# Notifications
#$ -M b.gower-winter@uu.nl
# When notified (b : begin, e : end, s : error)
#$ -m bes
# Execute from the current directory, and pass environment:
#$ -cwd # use current directory
#$ -V  # pass my complete environment to the job
# define output file name
#$ -o ./output.o
# define error file name
#$ -e ./error_output.o
#cd <PATH TO YOUR PYTHON SCRIPT> #

echo 'Start of Job:'
# Set up the environment
source ./venv/bin/activate
# Install all needed modules:
# Examples:
# python -m pip install requirements.txt

python3 -m ./src/main.py -n 50 -f 1.0 --tau 1000 --config ./configs/self_defeating/default.json -o 0001.json --debug
echo 'Finished computation.'
