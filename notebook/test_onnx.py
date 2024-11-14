import torch
import onnxruntime as ort

dic = torch.load(
    "/eos/user/m/mgarciam/EVAL_REPOS/Tracking_wcoc/models/test/showers_df_evaluation/graphs_all_hdb/0_0_0.pt",
    map_location="cpu",
)

pos_hits_xyz = dic["graph"].ndata["pos_hits_xyz"]
hit_type = dic["graph"].ndata["hit_type"].view(-1, 1)
vector = dic["graph"].ndata["vector"]
# input_data = torch.cat((pos_hits_xyz, hit_type, vector), dim=1)


ort.set_default_logger_severity(0)

so = ort.SessionOptions()
so.enable_profiling = True
print(so.inter_op_num_threads)
print(so.intra_op_num_threads)
so.inter_op_num_threads = 1
so.intra_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

# GraphOptimizationLevel::ORT_DISABLE_ALL -> Disables all optimizations
# --> This option works and takes 3h to load
# GraphOptimizationLevel::ORT_ENABLE_BASIC -> Enables basic optimizations
# GraphOptimizationLevel::ORT_ENABLE_EXTENDED -> Enables basic and extended optimizations
# GraphOptimizationLevel::ORT_ENABLE_ALL -> Enables all available optimizations including layout optimizations

print("starting to load")
ort_session = ort.InferenceSession(
    "/eos/user/m/mgarciam/EVAL_REPOS/Tracking_wcoc/models/test/model_multivector_11.onnx",
    # providers=["CPUExecutionProvider"],
    sess_options=so,
)
print("finished loading to load")

# input_data = [
#     torch.randn((30, 3)).cpu().numpy(),
#     torch.randn((30, 1)).cpu().numpy(),
#     torch.randn((30, 3)).cpu().numpy(),
# ]

input_data = [
    pos_hits_xyz.cpu().numpy(),
    hit_type.view(-1, 1).cpu().numpy(),
    vector.cpu().numpy(),
]

# def to_numpy(tensor):
#     for i in range(0,len(input_data)):

#     return (
#         tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
#     )


# compute ONNX Runtime output prediction
ort_inputs = {
    ort_session.get_inputs()[0].name: input_data[0],
    ort_session.get_inputs()[1].name: input_data[1],
    ort_session.get_inputs()[2].name: input_data[2],
}  # to_numpy(input_data)
ort_outs = ort_session.run(None, ort_inputs)

print(ort_outs)
print(ort_outs[0].shape)
