import onnx
import onnx.inliner

print("starting to load")
model_proto = onnx.load(
    "/eos/user/m/mgarciam/EVAL_REPOS/Tracking_wcoc/models/test/model_experimental_2.onnx"
)
inlined = onnx.inliner.inline_local_functions(model_proto)
onnx.save(
    inlined,
    "/eos/user/m/mgarciam/EVAL_REPOS/Tracking_wcoc/models/test/model_inlined_2.onnx",
)
print("finished loading to load")

# then test inlined model
import onnxruntime as ort

so = ort.SessionOptions()
# so.enable_profiling = True
print(so.inter_op_num_threads)
print(so.intra_op_num_threads)
# so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

print("starting to load")
ort_session = ort.InferenceSession(
    "/eos/user/m/mgarciam/EVAL_REPOS/Tracking_wcoc/models/test/model_inlined_2.onnx",
    providers=["CPUExecutionProvider"],
    sess_options=so,
)
print("finished loading to load")


#tried with the model without dynamic inputs 