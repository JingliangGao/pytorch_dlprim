
from torchgen.model import (
    BackendMetadata,
    DispatchKey,
    NativeFunction,
    FunctionSchema,
    OperatorName,
    DeviceCheckType,
    Location,
    is_structured_dispatch_key,
    DEFAULT_KERNEL_NAMESPACE,
    UfuncKey,
    UfuncInnerLoop,
    Variant,
    STRUCTURED_DISPATCH_KEYS,
    UFUNC_DISPATCH_KEYS,
    Precompute,
)
from torchgen.utils import NamespaceHelper
from typing import Dict, Set, Tuple, Optional

import torchgen

# func: do not do anything in __post_init__
def patch__post_init__(self) -> None:
    pass

# func: parse NativeFunction from yaml
@staticmethod
def patch__from_yaml(
    ei: dict[str, object],
    loc: Location,
    valid_tags: set[str],
    ignore_keys: set[DispatchKey] | None = None,
) -> tuple[NativeFunction, dict[DispatchKey, dict[OperatorName, BackendMetadata]]]:
    """
    Parse a NativeFunction from a dictionary as directly parsed
    from native_functions.yaml
    """
    e = ei.copy()

    funcs = e.pop("func")
    assert isinstance(funcs, str), f"not a str: {funcs}"
    # only support one level of namespace. E.g., aten::add
    namespace_helper = NamespaceHelper.from_namespaced_entity(
        namespaced_entity=funcs, max_level=1
    )
    namespace = namespace_helper.get_cpp_namespace(default="aten")
    func = FunctionSchema.parse(namespace_helper.entity_name)

    cpp_no_default_args_list = e.pop("cpp_no_default_args", [])
    assert isinstance(cpp_no_default_args_list, list)
    cpp_no_default_args = set(cpp_no_default_args_list)

    use_const_ref_for_mutable_tensors = e.pop(
        "use_const_ref_for_mutable_tensors", False
    )
    assert isinstance(use_const_ref_for_mutable_tensors, bool)

    if use_const_ref_for_mutable_tensors:
        assert (
            not func.arguments.out
        ), "see https://github.com/pytorch/pytorch/issues/145522"

    variants_s = e.pop("variants", "function")
    assert isinstance(variants_s, str)
    variants: set[Variant] = set()
    for v in variants_s.split(", "):
        if v == "function":
            variants.add(Variant.function)
        elif v == "method":
            variants.add(Variant.method)
        else:
            raise AssertionError(f"illegal variant {v}")

    manual_kernel_registration = e.pop("manual_kernel_registration", False)
    assert isinstance(
        manual_kernel_registration, bool
    ), f"not a bool: {manual_kernel_registration}"

    manual_cpp_binding = e.pop("manual_cpp_binding", False)
    assert isinstance(manual_cpp_binding, bool), f"not a bool: {manual_cpp_binding}"

    device_guard = e.pop("device_guard", True)
    assert isinstance(device_guard, bool), f"not a bool: {device_guard}"

    device_check_s = e.pop("device_check", None)
    assert device_check_s is None or isinstance(
        device_check_s, str
    ), f"not a str: {device_check_s}"
    assert (
        device_check_s is None or device_check_s in DeviceCheckType.__members__
    ), f"illegal device_check: {device_check_s}"
    device_check: DeviceCheckType
    if device_check_s is None:
        device_check = DeviceCheckType.ExactSame
    else:
        device_check = DeviceCheckType[device_check_s]

    structured = e.pop("structured", False)
    assert isinstance(structured, bool), f"not a bool: {structured}"

    structured_delegate_s = e.pop("structured_delegate", None)
    assert structured_delegate_s is None or isinstance(
        structured_delegate_s, str
    ), f"not a str: {structured_delegate_s}"
    assert structured_delegate_s is None or "::" not in structured_delegate_s, (
        "namespace is not supported in structured delegate,"
        " using the same namespace as the native function"
    )
    structured_delegate: OperatorName | None = None
    if structured_delegate_s is not None:
        structured_delegate = OperatorName.parse(structured_delegate_s)

    structured_inherits = e.pop("structured_inherits", None)
    assert structured_inherits is None or isinstance(
        structured_inherits, str
    ), f"not a str: {structured_inherits}"
    assert structured_inherits is None or "::" not in structured_inherits, (
        "namespace is not supported in structured inherits,"
        " using the same namespace as the native function"
    )

    python_module = e.pop("python_module", None)
    assert python_module is None or isinstance(
        python_module, str
    ), f"not a str: {python_module}"
    assert (
        python_module is None or Variant.method not in variants
    ), "functions in modules cannot be methods"

    category_override = e.pop("category_override", None)
    assert category_override is None or isinstance(
        category_override, str
    ), f"not a str: {category_override}"

    precomputed_dict = e.pop("precomputed", None)
    assert precomputed_dict is None or structured is True
    precomputed = Precompute.parse(precomputed_dict) if precomputed_dict else None

    tags_inp = e.pop("tags", [])
    if isinstance(tags_inp, str):
        tags_inp = [tags_inp]
    assert isinstance(tags_inp, list)

    # All aten ops generated by torchgen receive the pt2_compliant tag.
    if namespace == "aten" and "pt2_compliant_tag" in valid_tags:
        tags_inp.append("pt2_compliant_tag")

    tags: set[str] = set()
    for t in tags_inp:
        assert len(valid_tags) > 0
        # TODO: verify that the tag is valid and has an entry in tags.yaml
        if t in valid_tags:
            tags.add(t)
        else:
            raise AssertionError(f"illegal tag {t}")

    from torchgen.api import cpp

    raw_dispatch = e.pop("dispatch", None)
    assert raw_dispatch is None or isinstance(raw_dispatch, dict), e
    dispatch: dict[DispatchKey, BackendMetadata] = {}
    num_dispatch_keys: int = 0
    if raw_dispatch is not None:
        assert not manual_kernel_registration, (
            "cannot specify both manual_kernel_registration and dispatch; with "
            "manual registration, dispatch has no effect!"
        )
        redundant_composite_implicit_autograd = False
        for ks, v in raw_dispatch.items():
            if ks == "__line__":
                continue  # not worth tracking line numbers for dispatch entries
            assert isinstance(ks, str), f"illegal dispatch key '{ks}' in {raw_dispatch}"
            assert isinstance(v, str), f"illegal dispatch value '{v}' in {raw_dispatch}"
            for k in ks.split(","):
                dispatch_key = DispatchKey.parse(k.strip())
                num_dispatch_keys += 1

                if ignore_keys and dispatch_key in ignore_keys:
                    continue
                assert dispatch_key in dispatch_keys, (
                    f"Dispatch key {dispatch_key} of kernel {v} "
                    "is not a supported dispatch key."
                )
                # We only allow at most 3 levels of namespace for kernels.
                # We will append "native" to a custom kernel namespace.
                namespace_helper = NamespaceHelper.from_namespaced_entity(
                    v, max_level=3
                )
                kernel_namespace = namespace_helper.get_cpp_namespace(default="at")
                # Why is 'structured' included? External backends (e.g.
                # XLA) opt into which ops are structured independently
                # of which in-tree ops are structured
                dispatch[dispatch_key] = BackendMetadata(
                    kernel=namespace_helper.entity_name,
                    structured=structured and is_structured_dispatch_key(dispatch_key),
                    cpp_namespace=(kernel_namespace + "::native"),
                )
                if (
                    dispatch_key is DispatchKey.CompositeImplicitAutograd
                    and v == cpp.name(func)
                ):
                    redundant_composite_implicit_autograd = True

        # We count the number of dispatch keys which have not been ignored to prevent a dispatch table
        # in which all backend keys are ignored but necessarily kept, remaining compositeimplicit,
        # from being treated as redundant.
        assert not (num_dispatch_keys == 1 and redundant_composite_implicit_autograd), (
            "unnecessary dispatch table for this function; just delete the dispatch "
            "key entirely"
        )
        # if a function is a structured delegate, deleting the dispatch
        # table is NOT semantics preserving
        assert (
            structured_delegate
            or dispatch.keys() != {DispatchKey.CompositeImplicitAutograd}
            or dispatch[DispatchKey.CompositeImplicitAutograd].supports_symint()
            or num_dispatch_keys != 1
        ), (
            f"unexpected name for singleton CompositeImplicitAutograd dispatch entry: expected {cpp.name(func)} "
            f"but got {dispatch[DispatchKey.CompositeImplicitAutograd]}.  Rename your implementation to the expected "
            "name, then delete the dispatch table"
        )
    elif not structured and structured_delegate is None:
        name = str(func.name.name)
        # assert not (
        #     name.startswith("new_")
        #     or name.endswith("_like")
        #     # TODO: maybe it's better to test the return
        #     or (
        #         func.arguments.tensor_options
        #         and not func.arguments.has_tensor_arg()
        #     )
        # ), (
        #     f"expected {name} to have a CompositeExplicitAutograd "
        #     "dispatch entry, but there was no dispatch table.  Factory functions "
        #     "should not have implicit dispatch as they should not be decomposed "
        #     "for __torch_dispatch__"
        # )
        dispatch[DispatchKey.CompositeImplicitAutograd] = BackendMetadata(
            cpp.name(func), structured=False, cpp_namespace=DEFAULT_KERNEL_NAMESPACE
        )

    composites_in_dispatch = [
        d
        for d in dispatch
        if d == DispatchKey.CompositeExplicitAutograd
        or d == DispatchKey.CompositeExplicitAutogradNonFunctional
        or d == DispatchKey.CompositeImplicitAutograd
        or d == DispatchKey.CompositeImplicitAutogradNestedTensor
    ]

    assert len(composites_in_dispatch) <= 1 or (
        len(composites_in_dispatch) == 2
        and (
            DispatchKey.CompositeExplicitAutogradNonFunctional
            not in composites_in_dispatch
        )
        and (
            DispatchKey.CompositeImplicitAutogradNestedTensor in composites_in_dispatch
        )
    ), (
        "cannot specify more than one of CompositeExplicitAutograd, CompositeExplicitAutogradNonFunctional, "
        "or CompositeImplicitAutograd on a single kernel; each "
        "strictly subsumes the other.  If you wanted to provide an explicit autograd "
        "implementation, specify CompositeExplicitAutograd; otherwise specify CompositeImplicitAutograd only"
    )

    autogen_str = e.pop("autogen", "")
    assert isinstance(autogen_str, str)
    autogen = (
        []
        if autogen_str == ""
        else [OperatorName.parse(x) for x in autogen_str.split(", ")]
    )

    raw_ufunc_inner_loop = e.pop("ufunc_inner_loop", {})
    ufunc_inner_loop = {}
    if isinstance(raw_ufunc_inner_loop, str):
        ufunc_inner_loop[UfuncKey.Generic] = UfuncInnerLoop.parse(
            raw_ufunc_inner_loop, UfuncKey.Generic
        )
    elif isinstance(raw_ufunc_inner_loop, dict):
        for k, vo in raw_ufunc_inner_loop.items():
            if k == "__line__":
                continue
            assert isinstance(k, str), f"ufunc_inner_loop key is not a str: {k}"
            assert isinstance(vo, str), f"ufunc_inner_loop value is not a str: {v}"
            ufunc_key = UfuncKey.parse(k)
            ufunc_inner_loop[ufunc_key] = UfuncInnerLoop.parse(vo, ufunc_key)
    else:
        raise AssertionError(
            f"ufunc_inner_loop not str or dict: {raw_ufunc_inner_loop}"
        )
    # Program the BackendIndex for the implicit dispatch entry from ufunc
    if ufunc_inner_loop:
        assert structured, "ufunc must be structured"

        # Delay import ufunc here to avoid circular import issue
        # See: https://github.com/pytorch/pytorch/issues/81294
        import torchgen.api.ufunc as ufunc

        for dispatch_key in UFUNC_DISPATCH_KEYS:
            assert (
                dispatch_key not in dispatch
            ), f"ufunc should not have explicit dispatch entry for {dispatch_key}"
            dispatch[dispatch_key] = BackendMetadata(
                kernel=ufunc.schema_kernel_name(func, dispatch_key),
                structured=True,
                cpp_namespace=DEFAULT_KERNEL_NAMESPACE,
            )

    if structured_delegate:
        # Structured functions MUST have a dispatch table
        is_abstract = True
    else:
        is_abstract = (
            dispatch.keys() != {DispatchKey.CompositeImplicitAutograd}
            and dispatch.keys() != {DispatchKey.CompositeImplicitAutogradNestedTensor}
            and dispatch.keys()
            != {
                DispatchKey.CompositeImplicitAutograd,
                DispatchKey.CompositeImplicitAutogradNestedTensor,
            }
        )

    has_composite_implicit_autograd_kernel = (
        DispatchKey.CompositeImplicitAutograd in dispatch
    )
    has_composite_implicit_autograd_nested_tensor_kernel = (
        DispatchKey.CompositeImplicitAutogradNestedTensor in dispatch
    )
    has_composite_explicit_autograd_kernel = (
        DispatchKey.CompositeExplicitAutograd in dispatch
    )
    has_composite_explicit_autograd_non_functional_kernel = (
        DispatchKey.CompositeExplicitAutogradNonFunctional in dispatch
    )

    # We aren't going to store dispatch metadata inline in NativeFunctions;
    # instead it is separately indexed by backend (so other backends can
    # add more dispatch entries after the fact).  Reindex the individual
    # metadata by OperatorName!
    backend_metadata = {k: {func.name: v} for k, v in dispatch.items()}

    # don't care if it exists or not; make it easier to use this function
    # with other yaml parsers that aren't setting __line__ in the dict
    e.pop("__line__", None)
    assert not e, f"leftover entries: {e}"

    # Asserts that we can't do in post_init, because they rely on backend-specific info
    if structured_delegate is not None:
        for key in STRUCTURED_DISPATCH_KEYS:
            assert key not in dispatch, (
                f"if structured_delegate, then must not have {key} in dispatch dictionary "
                "(it is delegated!)"
            )

    return (
        NativeFunction(
            func=func,
            use_const_ref_for_mutable_tensors=use_const_ref_for_mutable_tensors,
            variants=variants,
            structured=structured,
            structured_delegate=structured_delegate,
            structured_inherits=structured_inherits,
            precomputed=precomputed,
            autogen=autogen,
            ufunc_inner_loop=ufunc_inner_loop,
            manual_kernel_registration=manual_kernel_registration,
            manual_cpp_binding=manual_cpp_binding,
            python_module=python_module,
            category_override=category_override,
            device_guard=device_guard,
            device_check=device_check,
            loc=loc,
            cpp_no_default_args=cpp_no_default_args,
            is_abstract=is_abstract,
            has_composite_implicit_autograd_kernel=has_composite_implicit_autograd_kernel,
            has_composite_implicit_autograd_nested_tensor_kernel=has_composite_implicit_autograd_nested_tensor_kernel,
            has_composite_explicit_autograd_kernel=has_composite_explicit_autograd_kernel,
            has_composite_explicit_autograd_non_functional_kernel=has_composite_explicit_autograd_non_functional_kernel,
            tags=tags,
            namespace=namespace,
        ),
        backend_metadata,
    )

# func: apply all patches
def apply_codegen_patches():
    torchgen.model.NativeFunction.__post_init__ = patch__post_init__
    torchgen.model.NativeFunction.from_yaml = patch__from_yaml
