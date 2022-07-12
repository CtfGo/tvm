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

"""IR Ast Dumper"""
import tvm
from . import _ffi_api
from tvm.ir import PrimExpr
from tvm.tir import Stmt

def get_valid_fields(stmt_or_expr):
    result = {}
    for key in dir(stmt_or_expr):
        attr = getattr(stmt_or_expr, key)
        if not key.startswith("__") and (isinstance(attr, PrimExpr) or isinstance(attr, Stmt)):
            result[key] = attr
    return result

def match(fields, child):
    for key, value in fields.items():
        if value is child:
        #if str(value) == str(child):
            return key
    return "None"

def dump(stmt, filename="graph.txt"):
    stack = []
    ast_node = []
    ast_edge = []
    count = [0]
    idx2obj = {}

    def pre_func(stmt):
        node_idx = count[0]
        count[0] += 1
        idx2obj[node_idx] = (stmt, get_valid_fields(stmt))

        ast_node.append([node_idx, stmt])
        if len(stack):
            ast_edge.append([stack[-1], node_idx])

        stack.append(node_idx)

    def post_func(stmt):
        del stack[-1]

    _ffi_api.PrePostOrderVisit(stmt, pre_func, post_func)

    with open(filename, "w") as f:
        f.write("digraph {\n")
        f.write("    node [shape=matrix]\n")
        for node in ast_node:
            ast_type = type(node[1])
            ast_str = str(node[1]).replace("\n", "\\l").replace("\\n", "\\l").replace('"', '\\"')
            f.write("    node%d" % (node[0]))
            f.write("[label=\"%s\n%s\"]" % (ast_type, ast_str))
            f.write(";\n")
        for edge in ast_edge:
            f.write("    node%d -> node%d [label=\"%s\"];\n" % (edge[0], edge[1], match(idx2obj[edge[0]][1], idx2obj[edge[1]][0])))
        f.write("}\n")
