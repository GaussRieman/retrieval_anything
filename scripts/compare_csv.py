import pandas as pd
import os
import json



def compare_csv(src_file: str, src_col: str,
                dest_file: str, dest_col: str,
                output_file: str):
    """
    Compare values from specific columns of two CSV files line by line.
    Saves a new CSV with the original values and a boolean comparison result.
    
    :param src_file: Path to the source CSV file.
    :param src_col: Column name in the source CSV file to compare.
    :param dest_file: Path to the destination CSV file.
    :param dest_col: Column name in the destination CSV file to compare.
    :param output_file: Path to save the comparison result.
    """
    # Read both CSV files
    src_df = pd.read_csv(src_file)
    dest_df = pd.read_csv(dest_file)

    # Validate columns
    if src_col not in src_df.columns:
        raise ValueError(f"Column '{src_col}' not found in {src_file}")
    if dest_col not in dest_df.columns:
        raise ValueError(f"Column '{dest_col}' not found in {dest_file}")

    # Perform the comparison
    result_df = pd.DataFrame(columns=[src_col, dest_col, "Comparison", "ImgUrl"])
    for i in range(len(src_df)):
        src_value = src_df.at[i, src_col]
        src_url = src_df.at[i, "ImgUrl"]
        
        filtered_df = dest_df[dest_df["ImgUrl"] == src_url]
        dest_value = filtered_df[dest_col].values
        
        dest_value = dest_value[0] if len(dest_value) > 0 else " "
        print("src_value: ", src_value)
        print("dest_value: ", dest_value)
        

        result_df.at[i, "src_value"] = src_value
        result_df.at[i, "dest_value"] = dest_value
        result_df.at[i, "Comparison"] = (src_value == dest_value)
        result_df.at[i, "ImgUrl"] = src_url
    return result_df
    

def process_src(src_file: str):
    df = pd.read_csv(src_file)
    df_new = df[["ImgUrl", "ProductId"]]
    # df_new.to_csv("/datadrive/codes/frank/langchains/retrieval_anything/data/heinz_scene/heinz_src_new.csv", index=False)
    
    type_codes = []
    for _, row in df_new.iterrows():
        print("row: ", type(row))
        category = row["ProductId"]
        # If category has sub string in types, then set the code value
        if 4447496 == category:
            type_codes.append(0)
        elif 4447497 == category:
            type_codes.append(1)
        elif 4447494 == category:
            type_codes.append(2)
        elif 4447498 == category:
            type_codes.append(3)
        elif 4447495 == category:
            type_codes.append(4)
        elif 4447493 == category:
            type_codes.append(5)
        elif 4447499 == category:
            type_codes.append(6)
        else:
            type_codes.append(7)
            
    df_new["TypeCode"] = type_codes
    df_new.to_csv("/datadrive/codes/frank/langchains/retrieval_anything/data/heinz_scene/scene_type.csv", index=False)


def process_dest(dest_file: str):
    csv_path = dest_file
    df = pd.read_csv(csv_path)
    values = []
    
    for _, it in df.iterrows():
        resp = it["response"]
        value = resp[-2]
        # print("value: ", value)
        values.append(value)
    df["TypeCode"] = values
    df.to_csv("/datadrive/codes/frank/langchains/retrieval_anything/data/heinz_scene/Heinz-week-1-result.csv", index=False)
        


if __name__ == "__main__":
    #1. process_src()
    #2. process_dest()
    #3. compare
    src_file = "/datadrive/codes/frank/langchains/retrieval_anything/data/heinz_scene/scene_type.csv"
    dest_file = "/datadrive/codes/frank/langchains/retrieval_anything/data/heinz_scene/Heinz-week-1-result.csv"
    output_file = "/datadrive/codes/frank/langchains/retrieval_anything/data/heinz_scene/heinz_compare.csv"
    src_col = "TypeCode"
    dest_col = "TypeCode"
    
    process_src(src_file)
    process_dest(dest_file)
    
    result_df = compare_csv(src_file, src_col, dest_file, dest_col, output_file)
    result_df.to_csv(output_file, index=False)
    print("Comparison result saved to:", output_file)
    