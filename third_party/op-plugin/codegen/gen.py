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
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(es, f, allow_unicode=True, sort_keys=False)
    
    return path
    
    


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


def gen_return(
    f: NativeFunction,
    # deprecated_dict: Dict,
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

        # impl_name = f.impl_name
        # if not f.impl_name:
        #     impl_name = op_name
        
#         deprecated_warn = ""
#         if op_name_with_overload in deprecated_dict.keys():
#             deprecated_func = f'torch_npu.{str(f.func.name.name)}'
#             deprecated_replace = deprecated_dict[op_name_with_overload]
#             if deprecated_replace is not None:
#                 deprecated_warn += f'TORCH_WARN_ONCE("{deprecated_func} is deprecated and will be removed in future version. \
# Use {deprecated_replace} instead.");'
#             else:
#                 deprecated_warn += f'TORCH_WARN_ONCE("{deprecated_func} is deprecated and will be removed in future version.");'

        format_check = []
        format_display = []
        place_holder = []
        format_for_args = []
        inputs_list = ""
        is_aclnn_only = "c10_npu::IsAclnnOnly()"
        for a in sig.arguments():
            argument = a.argument
            if isinstance(a.argument, SelfArgument):
                argument = a.argument.argument
            if not isinstance(a.argument, TensorOptionsArguments) and argument.type.is_tensor_like():
                format_for_args.append(
                    f"    bool {a.name}_base_format = at_npu::native::FormatHelper::IsOpInputBaseFormat({a.name});\n")
                format_check.append(f" && {a.name}_base_format")
                format_display.append(f", !{a.name}_base_format")
                place_holder.append(f", {a.name} is internal format: %d")
            inputs_list += f"{a.name}, "
        inputs_list = inputs_list[:-2]
            

        # if "op_api" in f.impl_ns and "acl_op" in f.impl_ns:
        #     if not f.internal_format_opapi:
        #         pass
        #     else:
        #         pass
        # elif "op_api" in f.impl_ns:
        #     ns = f.impl_ns[0]
        #     if f.internal_format_opapi:
        #         pass
        #     else:
        #         pass
        # elif "acl_op" in f.impl_ns:
        #     ns = f.impl_ns[0]
        #     pass
        # if f.sparse is not None:
        #     pass
    return ret


def parse_native_yaml(
    path: str,
) -> Tuple[Dict[str, list], List[Optional[str]]]:

    with open(path, "r") as f:
        es = yaml.safe_load(f)
    
    res = parse_native_yaml_struct(es)
    backend_declarations = defaultdict(list)
    for f in res:
        gen_function_declaration(f, backend_declarations)

    dispatch_registrations_body = sorted(set(concatMap(lambda f: gen_return(f), res)))   # lambda f: gen_return(f, {})

    return backend_declarations, dispatch_registrations_body