
from codegen.gen import parse_native_yaml, sort_native_yaml
from pathlib import Path
from torchgen.gen import FileManager
from torchgen.utils import concatMap
import argparse
import os
from codegen.config import NAMESPACE

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

    # parse native yaml
    source_yaml_path = os.path.realpath(options.source_yaml)
    source_yaml_path = sort_native_yaml(source_yaml_path)
    backend_declarations, dispatch_registrations_body, p_registrations_body, ap_registrations_body = parse_native_yaml(source_yaml_path)
    all_functions = sorted(set(concatMap(lambda f: [f], set(v for sublist in backend_declarations.values() for v in sublist))))
    
    # create FileManager
    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(
            install_dir=install_dir, template_dir="codegen/templates", dry_run=False
        )

    fm = make_file_manager(options.output_dir)

    # write "OpInterface.h" file
    fm.write_with_template(
        "OpInterface.h",
        "OpInterface.h",
        lambda: {
            "namespace": NAMESPACE,
            "declarations": all_functions,
        },
    )
    print(f"[INFO] succeed to generate 'OpInterface.h' in '{options.output_dir}'")

    # write "Register.cpp" file
    fm.write_with_template(
        "RegisterOps.cpp",
        "Register.cpp",        
        lambda: {
            "namespace": NAMESPACE,
            "declarations": dispatch_registrations_body,
            "p_namespace": "aten",
            "p_declarations": p_registrations_body,
            "ap_namespace": "aten",
            "ap_declarations": ap_registrations_body,
        },
    )
    print(f"[INFO] succeed to generate 'RegisterOps.cpp' in '{options.output_dir}'")


if __name__ == '__main__':
    main()
