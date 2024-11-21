import os
import ast
import yaml


def extract_classes(file_path: str):
    """
    Extract the names of all classes in a Python file.

    Parameters:
        file_path (str): Path to the Python file.

    Returns:
        class_names (list): List of class names.
    """
    class_names = []
    with open(file_path, 'r', encoding='utf-8') as f:
        node = ast.parse(f.read(), filename=file_path)
    for child in node.body:
        if isinstance(child, ast.ClassDef):
            class_names.append(child.name)
    return class_names


def generate_api_docs(root_folder: str, output_folder: str, index_file: str, mkdocs_file: str):
    """
    Automatically generate Markdown files for API documentation, update the homepage, and modify mkdocs.yml.

    Parameters:
        root_folder (str): Path to the root directory of the library.
        output_folder (str): Path to the output directory for the generated documentation.
        index_file (str): Path to the homepage file.
        mkdocs_file (str): Path to the mkdocs.yml file.
    """
    nav_structure = {}

    for root, dirs, files in os.walk(root_folder):
        relative_root = os.path.relpath(root, root_folder)  # Get the relative path of the current directory
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                file_path = os.path.join(root, file)
                class_names = extract_classes(file_path)
                if not class_names:
                    print(f"Warning: No class found in {file_path}. Skipping...")
                    continue

                module_path = file_path.replace('/', '.').replace('\\', '.').replace('.py', '')
                # Create a corresponding output directory that mirrors the input directory's structure
                output_dir = os.path.join(output_folder, relative_root, file.replace('.py', ''))
                os.makedirs(output_dir, exist_ok=True)

                # Generate Markdown files for each class
                for class_name in class_names:
                    class_md_path = os.path.join(output_dir, f"{class_name}.md")
                    with open(class_md_path, 'w', encoding='utf-8') as f:
                        f.write(f"# {class_name}\n\n::: {module_path}.{class_name}\n")

                # Update navigation structure
                current_nav = nav_structure
                for part in relative_root.split(os.sep):
                    if part:
                        current_nav = current_nav.setdefault(part, {})
                current_nav.setdefault(file.replace('.py', ''), []).extend([{
                    class_name: os.path.relpath(os.path.join(output_dir, f"{class_name}.md"), output_folder)
                } for class_name in class_names])

    # Generate the content for the homepage
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write("# Python Motion Planning Documentation\n\n")
        def write_nav(current_nav, level=1):
            for category, subcategories in sorted(current_nav.items()):
                f.write(f"{'#' * level} {category.capitalize()}\n\n")
                if isinstance(subcategories, dict):
                    write_nav(subcategories, level + 1)
                else:
                    for item in sorted(subcategories, key=lambda x: list(x.keys())[0]):
                        class_name = list(item.keys())[0]
                        doc_path = item[class_name].replace('\\', '/')
                        f.write(f"- [{class_name}]({doc_path})\n")
                    f.write("\n")
        write_nav(nav_structure)

    # Build the nav section of mkdocs.yml
    nav = [{"Home": "index.md"}]
    def build_nav(current_nav) -> list:
        if isinstance(current_nav, dict):
            return [{category: build_nav(subcategories)} for category, subcategories in sorted(current_nav.items())]
        else:
            return current_nav
    nav.extend(build_nav(nav_structure))

    print("\nGenerated nav for mkdocs.yml:")
    print(yaml.dump({"nav": nav}, allow_unicode=True, sort_keys=False))

    # If mkdocs.yml exists, automatically update its nav section
    if os.path.exists(mkdocs_file):
        with open(mkdocs_file, 'r', encoding='utf-8') as f:
            mkdocs_config = yaml.unsafe_load(f)

        mkdocs_config['nav'] = nav

        with open(mkdocs_file, 'w', encoding='utf-8') as f:
            yaml.dump(mkdocs_config, f, allow_unicode=True, sort_keys=False)


# Example usage
generate_api_docs(
    root_folder='python_motion_planning',  # Code directory
    output_folder='docs/',  # Directory for the generated documentation
    index_file='docs/index.md',  # Path to the homepage file
    mkdocs_file='mkdocs.yml'  # Path to the mkdocs.yml file
)