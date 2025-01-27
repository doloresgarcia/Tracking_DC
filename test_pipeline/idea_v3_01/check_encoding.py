import ROOT
from podio import root_io
import dd4hep as dd4hepModule
from ROOT import dd4hep

input_file_path = "output_digi.root"

podio_reader = root_io.Reader(input_file_path)
metadata = podio_reader.get("metadata")[0]
cellid_encoding = metadata.get_parameter("DCHCollection__CellIDEncoding")
decoder = dd4hep.BitFieldCoder(cellid_encoding)

available_fields = decoder.fields()
for field in available_fields:
    field_name = field.name()
    
    print(field_name)
    
    
    
for event in podio_reader.get("events"):
    for dc_hit in event.get("DCH_DigiCollection"):
                    cellID = dc_hit.getCellID()
                    superLayer = decoder.get(cellID, "superlayer")
                    layer = decoder.get(cellID, "layer")
                    # phi = decoder.get(cellID, "nphi")
                    # stereo = decoder.get(cellID, "stereosign")
                    # print("SuperLayer: ", superLayer, " Layer: ", layer, " Phi: ", phi, " Stereo: ", stereo)