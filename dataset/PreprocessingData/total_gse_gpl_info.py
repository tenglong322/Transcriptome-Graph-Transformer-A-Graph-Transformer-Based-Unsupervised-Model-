import os
import csv
import chardet

folder_path = 'gse_gpl'
output_file = 'gse_gpl.csv'

with open(output_file, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Column1', 'Column2'])
    
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            base_name = os.path.splitext(file_name)[0]
            parts = base_name.split('_')
            try:
                if len(parts) >= 2:
                    file_path = os.path.join(folder_path, file_name)
                    
                    with open(file_path, 'rb') as f:
                        raw_data = f.read(10000) 
                        result = chardet.detect(raw_data)
                        encoding = result['encoding']
                    
                   
                        with open(file_path, 'r', encoding=encoding) as f:
                            lines = f.readlines()
                            if len(lines) > 2:
                                writer.writerow([parts[0], parts[1]])
            except UnicodeDecodeError:
                print(f"Failed to decode {file_name}, skipping.")
