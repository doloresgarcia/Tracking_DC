import os
from Gaudi.Configuration import *

# Loading the input SIM file
from Configurables import k4DataSvc, PodioInput
from k4FWCore.parseArgs import parser

parser.add_argument("--inputFile", default="output_IDEA_DIGI.root", help="InputFile")
parser.add_argument("--outputFile", default="out_tracker_algorithm.root", help="OutputFile")
args = parser.parse_args()

evtsvc = k4DataSvc("EventDataSvc")
evtsvc.input = args.inputFile
inp = PodioInput("InputReader")


# pattern recognition over digitized hits
from Configurables import GGTF_tracking_dbscan_eval

GGTF_tracking = GGTF_tracking_dbscan_eval(
    "GGTF_tracking_dbscan_eval",
    modelPath="/afs/cern.ch/user/a/adevita/public/workDir/k4RecTracker/model_multivector_input_011124_v2.onnx",
    inputHits_CDC="CDCHDigis",
    inputHits_VTXD="VTXDDigis",
    inputHits_VTXIB="VTXIBDigis",
    inputHits_VTXOB="VTXOBDigis",
    clustering_space="output_model",
    OutputLevel=INFO,
)

################ Output
from Configurables import PodioOutput

out = PodioOutput("out", OutputLevel=INFO)
out.outputCommands = ["keep *"]

out.filename = args.outputFile

# CPU information
from Configurables import AuditorSvc, ChronoAuditor

chra = ChronoAuditor()
audsvc = AuditorSvc()
audsvc.Auditors = [chra]
out.AuditExecute = True

from Configurables import ApplicationMgr

ApplicationMgr(
    TopAlg=[
        inp,
        GGTF_tracking,
        out,
    ],
    EvtSel="NONE",
    ExtSvc=[evtsvc, audsvc],
    StopOnSignal=True,
)