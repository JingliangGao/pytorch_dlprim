AutogradPrivateUse1_Op = [
    "aten::relu",
    "aten::copy_",
    "aten::to.device",
    "aten::to.dtype",
    "aten::to.other",
    "aten::linear",
    "aten::max_pool2d",
    "aten::_has_compatible_shallow_copy_type",
]
AllPrivateUse1_Op = [
    "aten::copy_",
    "aten::to.device",
    "aten::to.dtype",
    "aten::to.other",
]
NAMESPACE = "op_plugin"
