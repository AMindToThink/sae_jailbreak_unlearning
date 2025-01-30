import pandas as pd
import sys

def clean_refusal_data(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get list of columns except 'Question'
    answer_columns = [col for col in df.columns if col != 'Question']
    
    # For each answer column, remove the question text
    for col in answer_columns:
        df[col] = df[col].apply(lambda x: x.replace(df['Question'][df.index[df[col] == x][0]], '').strip())
    
    return df

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean_refusal_data.py <csv_path>")
        sys.exit(1)
        
    csv_path = sys.argv[1]
    cleaned_df = clean_refusal_data(csv_path)
    output_path = csv_path.replace('.csv', '_cleaned.csv')
    cleaned_df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")