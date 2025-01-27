from Gaudi.Configuration import INFO, WARNING
from k4FWCore import ApplicationMgr
from Configurables import k4DataSvc, PodioOutput, PodioInput
from Configurables import AuditorSvc, ChronoAuditor
from k4FWCore.parseArgs import parser
from Configurables import HiveSlimEventLoopMgr, HiveWhiteBoard, AvalancheSchedulerSvc

import os

################ parser

parser.add_argument("--inputFile", default="out_tracker.root", help="InputFile")
parser.add_argument("--outputFile", default="out_efficiency.root", help="OutputFile")
args = parser.parse_args()

################ input & output

from k4FWCore import IOSvc
io_svc = IOSvc("IOSvc")
io_svc.input = args.inputFile
io_svc.output = args.outputFile

################ eff
from Configurables import GGTF_efficiency

trackin_eff = GGTF_efficiency(
    "GGTF_efficiency",
    InputCollectionTracks=["CDCHTracks"],
    InputCollectionParticles=["MCParticles"],
    DC_associations=["CDCHDigisAssociation"],
    VTXD_links=["VTXD_links"],
    VTXIB_links=["VTXIB_links"],
    VTXOB_links=["VTXOB_links"],
    
    costheta_mc=["costheta_mc"],
    pt_mc=["pt_mc"],
    vertex_mc=["vertex_mc"],
    deltaMC_mc=["deltaMC_mc"],
    purity_mc=["purity_mc"],
    isReco_mc=["isReco_mc"],
    isMatched_mc=["isMatched_mc"],
    matched_track_index=["matched_track_index"],
    fakeTracks_index=["fakeTracks_index"],
    efficiency_mc=["efficiency_mc"],
    tracking_eff=["tracking_eff"],
    matched_butNoReco_particles_index=["matched_butNoReco_particles_index"],
    OutputLevel=INFO,
)

################ Application
from Configurables import EventDataSvc

chra = ChronoAuditor()
audsvc = AuditorSvc()
audsvc.Auditors = [chra]

ApplicationMgr(
    TopAlg=[trackin_eff],
    EvtSel="NONE",
    ExtSvc=[EventDataSvc("EventDataSvc"), audsvc],
    StopOnSignal=True,
    EvtMax=-1,
    OutputLevel=INFO,
)