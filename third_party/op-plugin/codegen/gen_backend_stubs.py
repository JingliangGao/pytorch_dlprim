
from codegen.gen import parse_native_yaml, sort_native_yaml
from pathlib import Path
from torchgen.gen import FileManager
from torchgen.utils import concatMap
import argparse
import os

current_file = Path(__file__).resolve()   
plugin_dir = current_file.parent.parent


    

def main() -> None:
    parser = argparse.ArgumentParser(description='Generate backend stub files')
    parser.add_argument(
        '-s',
        '--source_yaml',
        default=f'{plugin_dir}/op_plugin/config/op_plugin_functions.yaml',
        help='path to source yaml file containing operator external definitions')
    parser.add_argument(
        '-o', 
        '--output_dir', 
        default=f'{plugin_dir}/op_plugin/generate/',
        help='output directory')
    options = parser.parse_args()

    source_yaml_path = os.path.realpath(options.source_yaml)
    source_yaml_path = sort_native_yaml(source_yaml_path)
    backend_declarations, dispatch_registrations_body = parse_native_yaml(source_yaml_path)

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(
            install_dir=install_dir, template_dir="codegen/templates", dry_run=False
        )

    fm = make_file_manager(options.output_dir)

    # pytorch_version = os.environ.get('PYTORCH_VERSION').split('.')
    torch_dir = f"v2r9"

    all_functions = sorted(set(concatMap(lambda f: [f],
                                         set(v for sublist in backend_declarations.values() for v in sublist))))

    fm.write_with_template(
        "OpInterface.h",
        "OpInterface.h",
        lambda: {
            "torch_dir": torch_dir,
            "namespace": "op_plugin",
            "declarations": all_functions,
        },
    )

    # header_files = {
    #     "op_api": "OpApiInterface.h",
    #     "acl_op": "AclOpsInterface.h",
    #     "sparse": "SparseOpsInterface.h",
    # }
    # for op_type, file_name in header_files.items():
    #     fm.write_with_template(
    #         file_name,
    #         "Interface.h",
    #         lambda: {
    #             "torch_dir": torch_dir,
    #             "namespace": op_type,
    #             "declarations": backend_declarations[op_type],
    #         },
    #     )

    # fm.write_with_template(
    #     "OpInterface.cpp",
    #     "OpInterface.cpp",
    #     lambda: {
    #         "namespace": "op_plugin",
    #         "declarations": dispatch_registrations_body,
    #     },
    # )


if __name__ == '__main__':
    main()
