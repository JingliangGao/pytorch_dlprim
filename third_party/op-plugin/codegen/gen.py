from collections import defaultdict
from typing import (List, Dict, Optional, Tuple, overload)
from torchgen.model import NativeFunction, Argument
from torchgen.utils import concatMap, context
from torchgen.api.types.signatures import NativeSignature
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

    all_funcs = []
    if es['official']:
        all_funcs += es['official']


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
    type: str="raw_func",
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

        ns = NAMESPACE
        impl_name = op_name
        if type == "wrap_func":
            overload = f.func.name.overload_name  
            if overload:
                impl_name = "wrapper_" + overload + "_" + impl_name
            else:
                impl_name = "wrapper_" + impl_name  


        ret.append(f"""{sig.defn(name=impl_name)}{{

    /* insert profiler anchor */
    // at_torch::profiler::NPURecordFunction record("{op_name}");
    return at_torch::{ns}::{op_name}({args_exprs_str});
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
     
        ns = NAMESPACE
        impl_name = op_name
        overload = f.func.name.overload_name  
        if overload:
            impl_name = "wrapper_" + overload + "_" + impl_name
        else:
            impl_name = "wrapper_" + impl_name        

        # register to different dispatch key
        aten_name = f"aten::{op_name_with_overload}"
        if aten_name in AllPrivateUse1_Op:
            privateuse1_ret.add(f"""m.impl("{aten_name}", TORCH_FN(at_torch::{ns}::{impl_name}));""")
            autogradprivateuse1_ret.add(f"""m.impl("aten::{op_name_with_overload}", TORCH_FN(at_torch::{ns}::{impl_name}));""")
        elif aten_name in AutogradPrivateUse1_Op:
            autogradprivateuse1_ret.add(f"""m.impl("aten::{op_name_with_overload}", TORCH_FN(at_torch::{ns}::{impl_name}));""")
        else:
            privateuse1_ret.add(f"""m.impl("{aten_name}", TORCH_FN(at_torch::{ns}::{impl_name}));""")

    # return funcs in different dispatch key
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

    dispatch_registrations_body = sorted(set(concatMap(lambda f: gen_dispatch_return(f, "wrap_func"), res)))
    p_registrations_body = sorted(set(concatMap(lambda f: gen_dispatchkey_return(f, "privateuse1"), res)))
    ap_registrations_body = sorted(set(concatMap(lambda f: gen_dispatchkey_return(f, "autograd_privateuse1"), res)))

    return backend_declarations, dispatch_registrations_body, p_registrations_body, ap_registrations_body