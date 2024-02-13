# What part of the code will be run?
code2run = 'get_haloes' # Produce file with haloes, including number per mass bin
#code2run = 'get_sample'  # Make a cut in one property to generate (shuffled) samples
#code2run = 'get_params'
#code2run = 'run_HOD'

# Name of the simulation and work environment
simtype = 'BAHAMAS'
env = 'arilega'

# Simulations and redshifts
sims = ['HIRES/AGN_RECAL_nu0_L100N512_WMAP9']
snaps = [31]

# Path for output
dirout = '/users/arivgonz/output/Junk/'

# Consider haloes with more than npmin particles
npmin = 20

# Define the halo mass bin size for <HODs>, shuffling and biasf
dm0 = 0.057  #dex log10(Msun/h)

# Halo mass to be considered
mhnom = 'FOF/Group_M_Crit200'

# Target number densities in log10(n/vol) and the properties to be used
# String with values separated by a space
ndtarget = '-2.5 -3.5'
propname = 'Subhalo/Mass_030kpc'

# Are test plots to be produced? 
tet_plots = True

verbose = True
Testing = True

# Use slurm queues?
use_slurm = False
partition = 'test'   # test or compute
time = '09:30:00'    #Format: d-hh:mm:ss
nodes = 1
ntasks = 1
cputask = 1

#--------------End of input parameters-------------------
import os
from src.h2s_io import get_file_name

# Executable full path
path2program = os.getcwd()+'/src/'

counter = 0
for sim in sims:
    for snap in snaps:
        match code2run:
            case 'get_haloes':
                program = path2program+'h2s_gethaloes.py'
                args = simtype+' '+sim+' '+env+' '+str(snap)+' '+mhnom+' '+str(npmin)+' '+str(dm0)
                args = args+' '+dirout+' '+str(verbose)+' '+str(Testing)
                
            case 'get_sample':
                program = path2program+'h2s_getsample.py'

                filenom = get_file_name(simtype,sim,snap,dirout,filetype='sample')
                args = ' --listsim '+filenom
                args = args+' --listnd '+ndtarget+' --listprop '+propname
                args = args+' --listbool '+str(verbose)+' '+str(Testing)
                
            case 'get_params':
                program = path2program+'h2s_getparams.py'
                args = simtype+' '+sim+' '+env+' '
                
            case 'run_HOD':
                program = path2program+'h2s_runHOD.py'
                args = simtype+' '+sim+' '+env+' '
                
            case other:
                print(f'Code to run not recognised: {code2run} ({sim}, z={zz})')

        # Submit the jobs
        if not use_slurm:
            os.system(f'python3 {program} {args}')
        else:
            import time as tt

            log_dir = dirout+'logs/'
            if not os.path.exists(log_dir): os.mkdir(log_dir)

            # Name of the submission script
            now = tt.localtime()
            script = f"{log_dir}{code2run}_{now.tm_hour:02d}{now.tm_min:02d}_{counter}.sh"
            counter += 1
            # For checking if the module is loaded
            script_content1 = 'module_to_check="apps/anaconda3/2023.03/bin"'
            script_content2 = 'if [[ $module_list_output != *"$module_to_check"* ]]; then'

            # Write sumbission script
            with open(script,'w') as fh:
                fh.write("#!/bin/bash \n")
                fh.write("\n")
                fh.write("#SBATCH --job-name=job.{}.%J \n".format(code2run))
                fh.write("#SBATCH --output={}out.{}.%J \n".format(log_dir,code2run))
                fh.write("#SBATCH --error={}err.{}.%J \n".format(log_dir,code2run))
                fh.write("#SBATCH --time={} \n".format(time))
                fh.write("#SBATCH --nodes={} \n".format(str(nodes)))
                fh.write("#SBATCH --ntasks={} \n".format(str(ntasks)))
                fh.write("#SBATCH --cpus-per-task={} \n".format(str(cputask)))
                fh.write("#SBATCH --partition={} \n".format(partition))
                fh.write("\n")
                fh.write("flight env activate gridware \n")
                fh.write("\n")
                fh.write("{} \n".format(script_content1))
                fh.write("module_list_output=$(module list 2>&1) \n")
                fh.write("{} \n".format(script_content2))
                fh.write("  module load $module_to_check \n")
                fh.write("fi \n")
                fh.write("\n")
                fh.write("python3 {} {}\n".format(program, args))

            print("Run {}".format(script))
            #os.system("sbatch {}".format(script))
