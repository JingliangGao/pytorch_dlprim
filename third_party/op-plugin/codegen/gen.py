from collections import defaultdict
from typing import (List, Dict, Optional, Set, Callable, Any,
                    Union, TypeVar, Iterable, Tuple)
from torchgen.model import NativeFunction, Argument, TensorOptionsArguments
from torchgen.utils import Target, concatMap, context, NamespaceHelper
from torchgen.api.types.signatures import NativeSignature, DispatcherSignature
from torchgen.context import native_function_manager
from codegen.utils import get_torchgen_dir
from torchgen.gen import parse_tags_yaml
import os
import yaml
from pathlib import Path
from codegen.config import AutogradPrivateUse1_Op, AllPrivateUse1_Op, NAMESPACE

SYMINT_SET = set()

def sort_native_yaml(path: str):
    # open yaml file
    with open(path, "r") as f:
        es = yaml.safe_load(f)
    
    # check yaml file if it is empty
    rs = []
    if not es:
        raise AssertionError(f"[ERROR] Context in {path} is empty. ")
    
    # extract context then sort
    all_funcs = []
    if 'official' not in es:
        raise AssertionError("Can't find official in yaml.")
    es_official = es['official']
    if es_official:
        es_official_sorted = sorted(es_official, key=lambda x: x["func"])
        all_funcs += es_official_sorted
        es['official'] = es_official_sorted
    if not isinstance(all_funcs, list):
        raise TypeError("all_funcs must be a list")
    
    # rewrite yaml file
    current_file = Path(__file__).resolve()   
    plugin_path = current_file.parent.parent 
    new_yaml_path = f"{plugin_path}/op_plugin/generate/op_plugin_functions.yaml"
    with open(new_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(es, f, allow_unicode=True, sort_keys=False)
    
    return new_yaml_path
    
    


def parse_native_yaml_struct(
    es: object,
) -> List[NativeFunction]:

    rs: List[NativeFunction] = []
    if not es:
        return rs
    
    if 'official' not in es:
        raise AssertionError("Can't find official in yaml.")
    # if 'symint' not in es:
    #     raise AssertionError("Can't find symint in yaml.")
    # if 'custom' not in es:
    #     raise AssertionError("Can't find custom in yaml.")

    # if es['symint']:
    #     for e in es['symint']:
    #         global SYMINT_SET
    #         SYMINT_SET.add(e['func'].split("(")[0])

    all_funcs = []
    if es['official']:
        all_funcs += es['official']
    # if es['custom']:
    #     all_funcs += es['custom']
    # if ('quant' in es) and es['quant']:
    #     all_funcs += es['quant']

    if not isinstance(all_funcs, list):
        raise TypeError("all_funcs must be a list")

    torchgen_path = get_torchgen_dir()
    tag_path = os.path.join(torchgen_path, 'packaged/ATen/native/tags.yaml')
    valid_tags = parse_tags_yaml(tag_path)

    for e in all_funcs:
        funcs = e.get("func")
        with context(lambda: f"in:\n  {funcs}"):
            func, m = NativeFunction.from_yaml(e, "Location", valid_tags)
            rs.append(func)

    return rs

def gen_function_declaration(
    f: NativeFunction,
    backend_decalarations: Dict,
):
    with native_function_manager(f):
        has_symint = False
        op_name = str(f.func.name.name)
        global SYMINT_SET
        if f.func.is_out_fn():
            op_name += "_out"
        if str(f.func.name) in SYMINT_SET:
            op_name += "_symint"
            has_symint = True

        sig = NativeSignature(f.func, prefix='', symint=has_symint)
        sig_str = f"{sig.decl(name=op_name)};"
        backend_decalarations["op_api"].append(sig_str)

from dataclasses import dataclass
@dataclass(frozen=True)
class SelfArgument:
    argument: Argument


def gen_dispatch_return(
    f: NativeFunction,
) -> List[Optional[str]]:
    ret = []
    with native_function_manager(f):
        has_symint = False
        op_name_with_overload = str(f.func.name)
        op_name = str(f.func.name.name)
        global SYMINT_SET
        if f.func.is_out_fn():
            op_name += "_out"
        if str(f.func.name) in SYMINT_SET:
            op_name += "_symint"
            has_symint = True
     

        sig = NativeSignature(f.func, prefix='', symint=has_symint)
        args_exprs_str = ', '.join(a.name for a in sig.arguments())

        impl_name = op_name
        op_name = "wrapper_" + op_name
        ns = NAMESPACE
        ret.append(f"""{sig.defn(name=op_name)}{{

    // insert profiler anchor
    // at_torch::profiler::NPURecordFunction record("{impl_name}");
    return {ns}::{impl_name}({args_exprs_str});
}}
\n""")

    return ret

def gen_dispatchkey_return(
    f: NativeFunction,
    type: str
) -> List[Optional[str]]:
    privateuse1_ret = set()
    autogradprivateuse1_ret = set()
    with native_function_manager(f):
        has_symint = False
        op_name_with_overload = str(f.func.name)
        op_name = str(f.func.name.name)
        global SYMINT_SET
        if f.func.is_out_fn():
            op_name += "_out"
        if str(f.func.name) in SYMINT_SET:
            op_name += "_symint"
            has_symint = True
     

        impl_name = op_name
        op_name = "wrapper_" + op_name

        aten_name = f"aten::{op_name_with_overload}"
        if aten_name in AllPrivateUse1_Op:
            privateuse1_ret.add(f"""m.impl("{aten_name}", TORCH_FN({op_name}));""")
            autogradprivateuse1_ret.add(f"""m.impl("aten::{op_name_with_overload}", TORCH_FN({op_name}));""")
        elif aten_name in AutogradPrivateUse1_Op:
            autogradprivateuse1_ret.add(f"""m.impl("aten::{op_name_with_overload}", TORCH_FN({op_name}));""")
        else:
            privateuse1_ret.add(f"""m.impl("{aten_name}", TORCH_FN({op_name}));""")

    if type == "privateuse1":
        return sorted(list(privateuse1_ret))
    else:
        return sorted(list(autogradprivateuse1_ret))

def parse_native_yaml(
    path: str,
) -> Tuple[Dict[str, list], List[Optional[str]]]:

    with open(path, "r") as f:
        es = yaml.safe_load(f)
    
    res = parse_native_yaml_struct(es)
    backend_declarations = defaultdict(list)
    for f in res:
        gen_function_declaration(f, backend_declarations)

    dispatch_registrations_body = sorted(set(concatMap(lambda f: gen_dispatch_return(f), res)))
    p_registrations_body = sorted(set(concatMap(lambda f: gen_dispatchkey_return(f, "privateuse1"), res)))
    ap_registrations_body = sorted(set(concatMap(lambda f: gen_dispatchkey_return(f, "autograd_privateuse1"), res)))

    return backend_declarations, dispatch_registrations_body, p_registrations_body, ap_registrations_body