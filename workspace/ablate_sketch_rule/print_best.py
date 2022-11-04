# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Auto-scheduling a Neural Network for NVIDIA GPU
===============================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_

Auto-tuning for specific devices and workloads is critical for getting the
best performance. This is a tutorial on how to tune a whole neural
network for NVIDIA GPU with the auto-scheduler.

To auto-tune a neural network, we partition the network into small subgraphs and
tune them independently. Each subgraph is treated as one search task.
A task scheduler slices the time and dynamically allocates time resources to
these tasks. The task scheduler predicts the impact of each task on the end-to-end
execution time and prioritizes the one that can reduce the execution time the most.

For each subgraph, we use the compute declaration in :code:`tvm/python/topi` to
get the computational DAG in the tensor expression form.
We then use the auto-scheduler to construct a search space of this DAG and search
for good schedules (low-level optimizations).

Different from the template-based :ref:`autotvm <tutorials-autotvm-sec>` which relies on
manual templates to define the search space, the auto-scheduler does not require any
schedule templates. In other words, the auto-scheduler only uses the compute declarations
in :code:`tvm/python/topi` and does not use existing schedule templates.

Note that this tutorial will not run on Windows or recent versions of macOS. To
get it to run, you will need to wrap the body of this tutorial in a :code:`if
__name__ == "__main__":` block.
"""

import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor
import argparse
import os

#################################################################
# Parse arguments

def parse_args():
    parser = argparse.ArgumentParser("Evaluate tuned result")
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=16,
        help='batch size')
    parser.add_argument(
        '-d',
        '--device_id',
        type=int,
        default=7,
        help='device id to be used'
    )
    parser.add_argument(
        '--tuned_dir',
        default='./result',
        help='dirname of tuned result stored'
    )
    args = parser.parse_args()
    return args

args = parse_args()
print("Arguments: %s" % args)

#################################################################
# Define a Network
# ----------------
# First, we need to define the network with relay frontend API.
# We can load some pre-defined network from :code:`tvm.relay.testing`.
# We can also load models from MXNet, ONNX, PyTorch, and TensorFlow
# (see :ref:`front end tutorials<tutorial-frontend>`).
#
# For convolutional neural networks, although auto-scheduler can work correctly
# with any layout, we found the best performance is typically achieved with NHWC layout.
# We also implemented more optimizations for NHWC layout with the auto-scheduler.
# So it is recommended to convert your models to NHWC layout to use the auto-scheduler.
# You can use :ref:`ConvertLayout <convert-layout-usage>` pass to do the layout conversion in TVM.


def get_network(name, batch_size, layout="NHWC", dtype="float32"):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        assert layout == "NCHW"

        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)

    return mod, params, input_shape, output_shape


# Define the neural network and compilation target
network = "resnet-50"
batch_size = args.batch_size
layout = "NHWC"
target = tvm.target.Target("cuda")
dtype = "float32"

mod, params, input_shape, output_shape = get_network(network, batch_size, layout, dtype=dtype)

#################################################################
# Extract Search Tasks
# --------------------
# Next, we extract the search tasks and their weights from a network.
# The weight of a task is the number of appearances of the task's subgraph
# in the whole network.
# By using the weight, we can approximate the end-to-end latency of the network
# as :code:`sum(latency[t] * weight[t])`, where :code:`latency[t]` is the
# latency of a task and :code:`weight[t]` is the weight of the task.
# The task scheduler will just optimize this objective.

# Extract tasks from the network
print("Extract tasks...")
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
dev = tvm.device(str(target), args.device_id)

def debug_tuned_result(log_file):
    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print("Weight:%f" % task_weights[idx])
        compute_dag = task.compute_dag
        print("DAG------->")
        print(compute_dag)
        #sch, args = task.apply_best(log_file)
        inp, _ = auto_scheduler.load_best_record(log_file, task.workload_key)
        if inp is None:
            print("!!!Can't find tuned schedule, skip")
            continue
            #sch, tensors = compute_dag.apply_steps_from_state(compute_dag.get_init_state())
        else:
            sch, tensors = compute_dag.apply_steps_from_state(inp.state)
        lowered_module = tvm.lower(sch, tensors, simple_mode=True)
        print("TIR------->")
        print(lowered_module)
        print("TIR AST------->")
        print(lowered_module.astext())
        func = tvm.build(sch, tensors, target)
        print("CUDA------->")
        #print(task.print_best(log_file, print_mode="cuda"))
        print(func.imported_modules[0].get_source())
        input_data = []
        for tensor in tensors:
            shape = auto_scheduler.utils.get_const_tuple(tensor.shape)
            xd = tvm.nd.array((np.random.uniform(size=shape)).astype(dtype), device=dev)
            input_data.append(xd)
        evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
        print("Execution time of this task: %.3f ms" % (np.median(evaluator(*input_data).results) * 1000))


#################################################################
# Compile and Evaluate
# --------------------
# After auto-tuning, we can compile the network with the best schedules we found.
# All measurement records are dumped into the log file during auto-tuning,
# so we can read the log file and load the best schedules.

# Compile with the history best
def apply_tuned(log_file):
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, params=params)

    # Create graph executor
    dev = tvm.device(str(target), args.device_id)
    module = graph_executor.GraphModule(lib["default"](dev))
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    module.set_input("data", data_tvm)

    # Evaluate
    print(module.benchmark(dev, repeat=3, min_repeat_ms=500))

if os.path.isdir(args.tuned_dir):
    for root, dirs, files in os.walk(args.tuned_dir):
        for file_name in files:
            print("Apply file: %s" % log_file)
            log_file = os.path.join(root, file_name)
            debug_tuned_result(log_file)
        #apply_tuned(log_file)
else:
    log_file = args.tuned_dir
    print("Apply file: %s" % log_file)
    debug_tuned_result(log_file)

