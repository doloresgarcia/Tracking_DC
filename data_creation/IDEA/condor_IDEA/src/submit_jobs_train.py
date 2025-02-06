#!/usr/bin/env python
import os, sys, subprocess
import glob
import argparse
import time
import math
import sys
# ____________________________________________________________________________________________________________


# ____________________________________________________________________________________________________________
def absoluteFilePaths(directory):
    files = []
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            files.append(os.path.abspath(os.path.join(dirpath, f)))
    return files


# _____________________________________________________________________________________________________________
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        help="output directory ",
        default="/eos/experiment/fcc/ee/simulation/ClicDet/test/",
    )

    parser.add_argument("--njobs", help="max number of jobs", default=2)
    
    parser.add_argument("--type", help="simulation type", default="Pythia")
     
    parser.add_argument("--config", help="Pythia configuration card name", default="")

    parser.add_argument("--detectorVersion", help="Detector Version", default=3)
    
    parser.add_argument(
        "--queue",
        help="queue for condor",
        choices=[
            "espresso",
            "microcentury",
            "longlunch",
            "workday",
            "tomorrow",
            "testmatch",
            "nextweek",
        ],
        default="longlunch",
    )

    args = parser.parse_args()
    
    queue = args.queue
    outdir = os.path.abspath(args.outdir)
    
    njobs = int(args.njobs)
    sim_type = args.type
    config = args.config
    detectorVersion=int(args.detectorVersion)

    os.system("mkdir -p {}/{}/{}".format(outdir,sim_type,config))
    storage_path = "{}/{}/{}".format(outdir,sim_type,config)

    # find list of already produced files:
    list_of_outfiles = []
    for name in glob.glob("{}/{}/{}/*.root".format(outdir,sim_type,config)):
        list_of_outfiles.append(name)

    script = ""
    if sim_type == "Pythia":
        script = "src/run_sequence_global.sh"
        
    if sim_type == "gun":
        
        #TO-DO
        sys.exit(0)

    jobCount = 0

    cmdfile = """# here goes your shell script
executable    = {}

# here you specify where to put .log, .out and .err files
output                = std/condor.$(ClusterId).$(ProcId).out
error                 = std/condor.$(ClusterId).$(ProcId).err
log                   = std/condor.$(ClusterId).log

+AccountingGroup = "group_u_FCC.local_gen"
+JobFlavour    = "{}"
""".format(
        script, queue
    )


    for job in range(njobs):
        
        discard_events = math.ceil(0.07*njobs)
        
        if discard_events < 1:
            discard_events = 1
        if njobs <= 2:
            discard_events = -1
        
        if job>discard_events:
            
            seed = str(job + 1)
            basename = config + "_graphs_" + seed + ".root"
            outputFile = storage_path + "/" + basename

            # print outdir, basename, outputFile
            if not outputFile in list_of_outfiles:
                print("{} : missing output file ".format(outputFile))
                jobCount += 1

                argts = "{} {} {} {} {}".format(outdir,sim_type,config,detectorVersion,seed)

                cmdfile += 'arguments="{}"\n'.format(argts)
                cmdfile += "queue\n"

                # cmd = "rm -rf job*; ./{} {}".format(script, argts)
                # if jobCount == 1:
                #     print("")
                #     print(cmd)

    gun_name = "gun/{}_{}.sub".format(sim_type,config)
    with open(gun_name, "w") as f:
        f.write(cmdfile)

    ### submitting jobs
    if jobCount > 0:
        print("")
        print("[Submitting {} jobs] ... ".format(jobCount))
        os.system("condor_submit {}".format(gun_name))


# _______________________________________________________________________________________
if __name__ == "__main__":
    main()
