import os

from Gaudi.Configuration import INFO, WARNING

from Configurables import AuditorSvc, ChronoAuditor
from Configurables import EventDataSvc
from Configurables import HiveSlimEventLoopMgr, HiveWhiteBoard, AvalancheSchedulerSvc


################ parser
from k4FWCore.parseArgs import parser

parser.add_argument("--inputFile", default="output_IDEA_DIGI.root", help="InputFile")
parser.add_argument("--outputFile", default="out_tracker.root", help="OutputFile")
args = parser.parse_args()

################ input & output
from k4FWCore import IOSvc

io_svc = IOSvc("IOSvc")
io_svc.input = args.inputFile
io_svc.output = args.outputFile

# pattern recognition over digitized hits
from Configurables import GGTF_tracking_dbscan

GGTF = GGTF_tracking_dbscan(
    "GGTF_tracking_dbscan",
    inputHits_CDC=["CDCHDigis"],
    inputHits_VTXD=["VTXDDigis"],
    inputHits_VTXIB=["VTXIBDigis"],
    inputHits_VTXOB=["VTXOBDigis"],
    outputTracks=["CDCHTracks"],
    OutputLevel=INFO,
)
GGTF.modelPath = "/afs/cern.ch/user/a/adevita/public/workDir/k4RecTracker/model_multivector_input_011124_v2.onnx"

################ Application
from k4FWCore import ApplicationMgr
chra = ChronoAuditor()
audsvc = AuditorSvc()
audsvc.Auditors = [chra]
ApplicationMgr(
    TopAlg=[GGTF],
    EvtSel="NONE",
    ExtSvc=[EventDataSvc("EventDataSvc"), audsvc],
    StopOnSignal=True,
    EvtMax=-1,
    OutputLevel=INFO,
)
    
    