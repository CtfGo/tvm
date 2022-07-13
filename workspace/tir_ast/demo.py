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
import numpy as np

import tvm
from tvm import te
from tvm.tir import ast_dumper
from tvm.ir.module import IRModule
from tvm.script import tir as T
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

#n = te.var("m")
#A = te.placeholder((n,), name='A')
#B = te.placeholder((n,), name='B')
#C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
#
#s = te.create_schedule([C.op])
##bx, tx = s[C].split(C.op.axis[0], factor=64)
#
#res = tvm.lower(s, [A, B, C], simple_mode=True)
#print("--->Module")
#print(res)
#print("--->PrimFunc.body")
#print(res["main"].body)
#print("--->Dump")
##print(ast_dumper.dump(res["main"].body))
#ast_dumper.dump(res["main"].body, os.path.join("./log", "task_%s.dot" % (0)))

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        # We exchange data between function by handles, which are similar to pointer.
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # Create buffer from handles.
        A = T.match_buffer(a, (8,), dtype="float32")
        B = T.match_buffer(b, (8,), dtype="float32")
        for i in range(8):
            # A block is an abstraction for computation.
            with T.block("B"):
                # Define a spatial block iterator and bind it to value i.
                vi = T.axis.spatial(8, i)
                B[vi] = A[vi] + 1.0


ir_module = MyModule
print("--->Module")
print(ir_module)
print("--->PrimFunc.body")
print(ir_module["main"].body)
ast_dumper.dump(ir_module["main"].body, os.path.join("./ast", "task_%s.dot" % (1)))
