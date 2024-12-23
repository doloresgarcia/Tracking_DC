import os

from Gaudi.Configuration import *

# Loading the input SIM file
from Configurables import k4DataSvc, PodioInput

evtsvc = k4DataSvc("EventDataSvc")
evtsvc.input = "out_sim_edm4hep.root"
inp = PodioInput("InputReader")

################## Simulation setup
# Detector geometry
from Configurables import GeoSvc

geoservice = GeoSvc("GeoSvc")
path_to_detector = os.environ.get("K4GEO", "")
print(path_to_detector)
detectors_to_use = [
    "/afs/cern.ch/work/m/mgarciam/private/k4geo/FCCee/IDEA/compact/IDEA_o1_v02/IDEA_o1_v02.xml"
]
# prefix all xmls with path_to_detector
geoservice.detectors = [
    os.path.join(path_to_detector, _det) for _det in detectors_to_use
]
geoservice.OutputLevel = INFO

# digitize vertex hits
from Configurables import VTXdigitizer
import math

innerVertexResolution_x = 0.003  # [mm], assume 5 µm resolution for ARCADIA sensor
innerVertexResolution_y = 0.003  # [mm], assume 5 µm resolution for ARCADIA sensor
innerVertexResolution_t = 1000  # [ns]
outerVertexResolution_x = 0.050 / math.sqrt(
    12
)  # [mm], assume ATLASPix3 sensor with 50 µm pitch
outerVertexResolution_y = 0.150 / math.sqrt(
    12
)  # [mm], assume ATLASPix3 sensor with 150 µm pitch
outerVertexResolution_t = 1000  # [ns]

vtxib_digitizer = VTXdigitizer(
    "NewVTXIBCollection",
    inputSimHits="NewVTXIBCollection",
    outputDigiHits="VTXIBDigis",
    detectorName="Vertex",
    readoutName="VTXIBCollection",
    xResolution=innerVertexResolution_x,  # mm, r-phi direction
    yResolution=innerVertexResolution_y,  # mm, z direction
    tResolution=innerVertexResolution_t,
    forceHitsOntoSurface=False,
    OutputLevel=INFO,
)

vtxob_digitizer = VTXdigitizer(
    "NewVTXOBCollection",
    inputSimHits="NewVTXOBCollection",
    outputDigiHits="VTXOBDigis",
    detectorName="Vertex",
    readoutName="VTXOBCollection",
    xResolution=outerVertexResolution_x,  # mm, r-phi direction
    yResolution=outerVertexResolution_y,  # mm, z direction
    tResolution=outerVertexResolution_t,  # ns
    forceHitsOntoSurface=False,
    OutputLevel=INFO,
)

vtxd_digitizer = VTXdigitizer(
    "NewVTXDCollection",
    inputSimHits="NewVTXDCollection",
    outputDigiHits="VTXDDigis",
    detectorName="Vertex",
    readoutName="VTXDCollection",
    xResolution=outerVertexResolution_x,  # mm, r direction
    yResolution=outerVertexResolution_y,  # mm, phi direction
    tResolution=outerVertexResolution_t,  # ns
    forceHitsOntoSurface=False,
    OutputLevel=INFO,
)

# digitize drift chamber hits
from Configurables import DCHsimpleDigitizerExtendedEdm

dch_digitizer = DCHsimpleDigitizerExtendedEdm(
    "DCHsimpleDigitizerExtendedEdm",
    inputSimHits="NewCDCHHits",
    outputDigiHits="CDCHDigis",
    outputSimDigiAssociation="CDCHDigisAssociation",
    readoutName="CDCHHits",
    xyResolution=0.1,  # mm
    zResolution=1,  # mm
    OutputLevel=INFO,
)

################ Output
from Configurables import PodioOutput

out = PodioOutput("out", OutputLevel=INFO)
out.outputCommands = ["keep *"]

out.filename = "output_IDEA_DIGI.root"

# CPU information
from Configurables import AuditorSvc, ChronoAuditor

chra = ChronoAuditor()
audsvc = AuditorSvc()
audsvc.Auditors = [chra]
out.AuditExecute = True

# from Configurables import EventCounter

# event_counter = EventCounter("event_counter")
# event_counter.Frequency = 1

from Configurables import ApplicationMgr

ApplicationMgr(
    TopAlg=[
        inp,
        # event_counter,
        vtxib_digitizer,
        vtxob_digitizer,
        vtxd_digitizer,
        dch_digitizer,
        out,
    ],
    EvtSel="NONE",
    EvtMax=2000,
    ExtSvc=[geoservice, evtsvc, audsvc],
    StopOnSignal=True,
)
