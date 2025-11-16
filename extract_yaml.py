import pandas as pd
import yaml

def csv_to_yaml(csv_path, yaml_path="dataset.yaml", nrows=500):
    # Read a sample of the CSV for types
    df = pd.read_csv(csv_path, nrows=nrows)
    columns = []
    numerical_columns = []
    for col in df.columns:
        dtype = df[col].dtype
        # Map pandas dtype to YAML schema type
        if pd.api.types.is_integer_dtype(dtype):
            col_type = "int"
            numerical_columns.append(col)
        elif pd.api.types.is_float_dtype(dtype):
            col_type = "float"
            numerical_columns.append(col)
        else:
            col_type = "object"
        columns.append({col: col_type})

    yaml_dict = {
        "columns": columns,
        "numerical_columns": numerical_columns
    }

    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    print(f"YAML written to {yaml_path}")

# Usage:
csv_to_yaml('./data/DementiaData.csv', './data_schema/schema.yaml')
